"""GBDT 的 570 列特征矩阵，列主序 [570, N] float32，行序与发布包行序一致。

列布局（与推理包 src/stream_features.py 一致）：
    [0, 323)    原始特征
    [323, 403)  TOP40 × {d1, dev60} 交错：该资产观测序列上的一阶差分 / 60 行滚动均值偏离
    [403]       asset_id
    [404, 564)  四个派生块各 40 列：xzs、xbeta128、xbeta512、xdmdev512
    [564, 570)  clock3：日内相位 1~3 次谐波的 sin/cos

派生块在稠密时间网格 [T, A] 上计算：缺席格与非有限值沿时间前向填充，段首没有
历史的格用该资产的第一个观测值回填；截面统计只用当期在场的资产。
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ..common import FEATURES, Paths, save_npy, scan_labels, write_json

DAY = 1440
EPS = 1e-12
N_BASE = 404


# ---------------------------------------------------------------- 基座 404 列
def base_frame(df: pl.DataFrame, top40: list[str], window: int, min_samples: int) -> pl.DataFrame:
    """在每个资产自己的行序列上算 d1 / dev60（缺失为 null，按 polars 语义跳过）。"""
    d1 = [pl.col(c).diff().over("asset_id").alias(f"{c}_d1") for c in top40]
    dev = [(pl.col(c) - pl.col(c).rolling_mean(window, min_samples=min_samples).over("asset_id"))
           .alias(f"{c}_dev60") for c in top40]
    df = df.with_row_index("_ridx").sort(["asset_id", "time_id"])
    df = df.with_columns(d1 + dev).sort("_ridx").drop("_ridx")
    interleaved = [f"{c}_{s}" for c in top40 for s in ("d1", "dev60")]
    return df.select([*FEATURES, *interleaved, "asset_id"])


# ---------------------------------------------------------------- 稠密面板
def ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    """沿 axis=0 的因果 EWMA，y[0] = x[0]；float64 滤波，输出 float32。"""
    a = 1.0 - 2.0 ** (-1.0 / halflife)
    xt = np.ascontiguousarray(x.T, dtype=np.float64)
    zi = ((1.0 - a) * xt[:, 0])[:, None]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], xt, axis=-1, zi=zi)
    return np.ascontiguousarray(y.T).astype(np.float32, copy=False)


def fill_panel(panel: np.ndarray, present: np.ndarray) -> None:
    """就地：无效格（缺席或非有限）沿时间前向填充，段首用首个有效值回填。"""
    valid = present & np.isfinite(panel)
    n_time = panel.shape[0]
    idx = np.where(valid, np.arange(n_time, dtype=np.int64)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    has_history = idx >= 0
    taken = np.take_along_axis(panel, np.where(has_history, idx, 0), axis=0)
    any_valid = valid.any(axis=0)
    head = panel[np.where(any_valid, valid.argmax(axis=0), 0), np.arange(panel.shape[1])]
    head = np.where(any_valid, head, 0.0).astype(panel.dtype)
    panel[...] = np.where(has_history, taken, head[None, :])
    np.nan_to_num(panel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def cross_section(col: np.ndarray, m: np.ndarray, n: np.ndarray):
    """当期在场资产的截面均值与总体标准差，各 [T, 1] float32。"""
    mean = (col * m).sum(axis=1, keepdims=True) / n
    var = ((col - mean) ** 2 * m).sum(axis=1, keepdims=True) / n
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def xzs(col, mean, std):
    return np.where(std > EPS, (col - mean) / np.maximum(std, EPS), 0.0).astype(np.float32)


def xbeta(col, mean, halflife):
    """对截面均值（市场因子）的 EWMA 滚动 beta。"""
    m = np.broadcast_to(mean, col.shape).astype(np.float32)
    e_m = ewma(m, halflife)
    cov = ewma(col * m, halflife) - ewma(col, halflife) * e_m
    var = np.maximum(ewma(m * m, halflife) - e_m ** 2, 0.0)
    return np.where(var > EPS, cov / np.maximum(var, EPS), 0.0).astype(np.float32)


def xdmdev(col, mean, halflife):
    d = col - mean
    return (d - ewma(d, halflife)).astype(np.float32)


BLOCKS = {
    "xzs": lambda col, mean, std: xzs(col, mean, std),
    "xbeta128": lambda col, mean, std: xbeta(col, mean, 128.0),
    "xbeta512": lambda col, mean, std: xbeta(col, mean, 512.0),
    "xdmdev512": lambda col, mean, std: xdmdev(col, mean, 512.0),
}


def clock3(time_id: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.pi * (time_id % DAY).astype(np.float64) / DAY
    cols = []
    for h in (1, 2, 3):
        cols += [np.sin(h * theta), np.cos(h * theta)]
    return np.stack(cols, axis=1).astype(np.float32)


def n_columns(cfg: dict) -> int:
    return N_BASE + 40 * len(cfg["gbdt"]["blocks"]) + 6


def unused_columns(cfg: dict) -> list[int]:
    g = cfg["gbdt"]
    cols = []
    for i, name in enumerate(g["blocks"]):
        if name in g["unused_blocks"]:
            cols += range(N_BASE + 40 * i, N_BASE + 40 * (i + 1))
    return cols


def used_columns(cfg: dict) -> np.ndarray:
    unused = set(unused_columns(cfg))
    return np.asarray([c for c in range(n_columns(cfg)) if c not in unused], dtype=np.int64)


# ---------------------------------------------------------------- 主流程
def build(cfg: dict, paths: Paths) -> None:
    g = cfg["gbdt"]
    out = paths.stage("gbdt")
    n_assets = cfg["n_assets"]
    top40 = [f"feature_{i:03d}" for i in g["top40"]]
    aux = f"responder_{g['catboost']['aux_responder']:02d}"

    df = scan_labels(paths, ["row_id", "time_id", "asset_id", "weight", "target", aux,
                             *FEATURES]).collect()
    tid = df["time_id"].to_numpy().astype(np.int64)
    aid = df["asset_id"].to_numpy().astype(np.int64)
    save_npy(out / "row_id.npy", df["row_id"].to_numpy().astype(np.int64))
    save_npy(out / "y.npy", df["target"].to_numpy().astype(np.float32))
    save_npy(out / "w.npy", df["weight"].to_numpy().astype(np.float32))
    save_npy(out / "aux.npy", df[aux].to_numpy().astype(np.float32))
    base = base_frame(df.drop(["row_id", "weight", "target", aux]), top40,
                      g["roll_window"], g["roll_min"])
    del df
    x404 = base.to_numpy().astype(np.float32, copy=False)
    del base
    n_rows = len(tid)

    n_cols = n_columns(cfg)
    store = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32,
                                      shape=(n_cols, n_rows))
    for lo in range(0, n_rows, 1_000_000):
        store[:N_BASE, lo:lo + 1_000_000] = x404[lo:lo + 1_000_000].T
    print(f"[gbdt/features] 基座 {n_rows:,}×{N_BASE}", flush=True)

    n_time = int(tid[-1]) + 1
    flat = tid * n_assets + aid
    present = np.zeros(n_time * n_assets, bool)
    present[flat] = True
    present = present.reshape(n_time, n_assets)
    m = present.astype(np.float32)
    n = np.maximum(m.sum(axis=1, keepdims=True), 1.0)

    panels = {}
    for c in g["top40"]:
        p = np.zeros((n_time, n_assets), np.float32)
        p.reshape(-1)[flat] = x404[:, c]
        fill_panel(p, present)
        panels[c] = p
    del x404
    col = N_BASE
    for name in g["blocks"]:
        for c in g["top40"]:
            mean, std = cross_section(panels[c], m, n)
            store[col] = BLOCKS[name](panels[c], mean, std).reshape(-1)[flat]
            col += 1
        print(f"[gbdt/features] {name}", flush=True)
    store[col:col + 6] = clock3(tid).T
    store.flush()
    write_json(out / "meta.json", {"rows": n_rows, "columns": n_cols,
                                   "layout": "column-major [columns, rows]"})
