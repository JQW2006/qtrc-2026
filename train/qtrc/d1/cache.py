"""D1 的输入：冻结标准化统计量 + 稠密网格 [T, A, 325]。

每行 325 列 = 323 个标准化后的特征 + 缺失比例 + 全缺失标志。
标准化：(x - mean) / scale，clip 到 ±5，缺失置 0。
网格里没有观测的 (time_id, asset) 格写成 x=0、两个缺失列=1、y=w=0。
"""
from __future__ import annotations

import numpy as np

from ..common import FEATURES, Paths, save_npy, scan_labels, write_json

RAW_DIM = 323
BASE_DIM = 325


def _blocks(n_rows: int, size: int):
    for start in range(0, n_rows, size):
        yield start, min(start + size, n_rows)


def fit_scaler(paths: Paths, n_rows: int, block_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """前 n_rows 行（即 0630 训练集）的列均值与总体标准差，忽略非有限值。"""
    sums = np.zeros(RAW_DIM, np.float64)
    squares = np.zeros(RAW_DIM, np.float64)
    counts = np.zeros(RAW_DIM, np.int64)
    source = scan_labels(paths, FEATURES)
    for lo, hi in _blocks(n_rows, block_rows):
        values = source.slice(lo, hi - lo).collect().to_numpy().astype(np.float32)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0).astype(np.float64)
        sums += safe.sum(axis=0)
        squares += np.square(safe).sum(axis=0)
        counts += finite.sum(axis=0)
    mean = np.divide(sums, counts, out=np.zeros(RAW_DIM), where=counts > 0)
    second = np.divide(squares, counts, out=np.zeros(RAW_DIM), where=counts > 0)
    scale = np.sqrt(np.maximum(second - mean * mean, 0.0))
    scale[(counts == 0) | ~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def build(cfg: dict, paths: Paths) -> None:
    c = cfg["d1"]
    out = paths.stage("d1")
    day, n_assets = cfg["day_len"], cfg["n_assets"]
    n_time = cfg["days"] * day

    keys = scan_labels(paths, ["time_id", "asset_id", "weight", "target"]).collect()
    tid = keys["time_id"].to_numpy().astype(np.int64)
    aid = keys["asset_id"].to_numpy().astype(np.int64)
    y = keys["target"].to_numpy().astype(np.float32)
    w = keys["weight"].to_numpy().astype(np.float32)
    if np.any(np.diff(tid) < 0) or tid[-1] + 1 != n_time:
        raise RuntimeError("拼接后的 time_id 不是 [0, days*1440) 上的单调序列")

    n_fit = int(np.searchsorted(tid, c["scaler_days"] * day))
    mean, scale = fit_scaler(paths, n_fit, c["block_rows"])
    fit = slice(0, n_fit)
    target_bound = round(float(np.abs(y[fit][w[fit] > 0]).max()), 4)

    x_grid = np.lib.format.open_memmap(out / "grid_x.npy", mode="w+", dtype=np.float16,
                                       shape=(n_time, n_assets, BASE_DIM))
    y_grid = np.zeros((n_time, n_assets), np.float32)
    w_grid = np.zeros((n_time, n_assets), np.float32)
    x_grid[:] = 0
    x_grid[:, :, RAW_DIM:] = 1

    source = scan_labels(paths, FEATURES)
    for lo, hi in _blocks(len(tid), c["block_rows"]):
        values = source.slice(lo, hi - lo).collect().to_numpy().astype(np.float32)
        observed = np.isfinite(values)
        missing = 1.0 - observed.mean(axis=1, dtype=np.float32)
        values[~observed] = np.nan
        values -= mean
        values /= scale
        np.clip(values, -c["clip"], c["clip"], out=values)
        values[~np.isfinite(values)] = 0
        t, a = tid[lo:hi], aid[lo:hi]
        x_grid[t, a, :RAW_DIM] = values.astype(np.float16)
        x_grid[t, a, RAW_DIM] = missing.astype(np.float16)
        x_grid[t, a, RAW_DIM + 1] = (missing == 1).astype(np.float16)
        y_grid[t, a] = y[lo:hi]
        w_grid[t, a] = w[lo:hi]
    x_grid.flush()
    del x_grid

    save_npy(out / "grid_y.npy", y_grid)
    save_npy(out / "grid_w.npy", w_grid)
    np.savez(out / "scaler.npz", mean=mean, scale=scale)
    write_json(out / "stats.json", {"scaler_fit_rows": n_fit, "rows": len(tid),
                                    "time_ids": n_time, "target_bound": target_bound})
    print(f"[d1/cache] {len(tid):,} 行 -> 网格 {n_time:,}×{n_assets}；"
          f"scaler 拟合 {n_fit:,} 行；target_bound={target_bound}", flush=True)


def load(paths: Paths, mmap: bool = True):
    d = paths.work / "d1"
    x = np.load(d / "grid_x.npy", mmap_mode="r" if mmap else None)
    return x, np.load(d / "grid_y.npy"), np.load(d / "grid_w.npy")
