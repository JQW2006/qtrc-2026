"""在全部 768 天上训练 CatBoost（多任务）与 XGBoost，各 3 个种子，导出纯 numpy 树库。"""
from __future__ import annotations

import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from ..common import Paths
from .features import n_columns, unused_columns, used_columns


def load(paths: Paths):
    d = paths.work / "gbdt"
    return (np.load(d / "X.npy", mmap_mode="r"), np.load(d / "y.npy"),
            np.load(d / "w.npy"), d)


# ---------------------------------------------------------------- CatBoost
def standardize(v: np.ndarray, clip: float) -> np.ndarray:
    """按全段 float32 均值/标准差标准化，clip，缺失置 0。"""
    finite = np.isfinite(v)
    m = float(v[finite].mean())
    s = float(v[finite].std())
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    z = np.empty_like(v, dtype=np.float32)
    np.subtract(v, m, out=z, where=finite, casting="unsafe")
    z[~finite] = 0.0
    z /= np.float32(s)
    np.clip(z, -clip, clip, out=z)
    return z


def mtl_label(y: np.ndarray, aux: np.ndarray, lam: float, clip: float) -> np.ndarray:
    """[N, 2]：第 0 维是 target，第 1 维是缩放到 sqrt(λ)·std(target) 的辅助 responder。

    MultiRMSE 的叶值按维独立求解，辅助维只影响分裂选择；推理只取第 0 维。
    """
    scale = np.float32(np.sqrt(lam) * float(np.std(y.astype(np.float64))))
    label = np.empty((len(y), 2), np.float32)
    label[:, 0] = y
    label[:, 1] = standardize(aux, clip) * scale
    return label


def export_catboost(model, n_features: int) -> dict:
    """对称树 → {feat, border, leaves, scale, bias}；多输出时只保留第 0 维叶值。"""
    with tempfile.NamedTemporaryFile(suffix=".json") as fh:
        model.save_model(fh.name, format="json")
        blob = json.loads(Path(fh.name).read_text())
    flat = {int(f["feature_index"]): int(f["flat_feature_index"])
            for f in blob["features_info"].get("float_features", [])}
    trees = blob["oblivious_trees"]
    depth = len(trees[0]["splits"])
    n_out = len(trees[0]["leaf_values"]) >> depth
    feat = np.empty((len(trees), depth), np.int32)
    border = np.empty((len(trees), depth), np.float32)
    leaves = np.empty((len(trees), 1 << depth), np.float64)
    for i, tree in enumerate(trees):
        if len(tree["splits"]) != depth:
            raise ValueError("树深不一致")
        for k, split in enumerate(tree["splits"]):
            feat[i, k] = flat[int(split["float_feature_index"])]
            border[i, k] = np.float32(split["border"])
        # 叶值按叶优先排布：同一叶子的 n_out 个分量相邻
        leaves[i] = np.asarray(tree["leaf_values"], np.float64).reshape(1 << depth, n_out)[:, 0]
    scale, bias = blob.get("scale_and_bias", [1.0, [0.0]])
    bias = bias[0] if isinstance(bias, (list, tuple)) else bias
    return {"kind": np.array("catboost"), "feat": feat, "border": border, "leaves": leaves,
            "scale": np.float64(scale), "bias": np.float64(bias),
            "n_features": np.int64(n_features)}


def train_catboost(cfg: dict, paths: Paths, seeds: list[int]) -> None:
    import pandas as pd
    from catboost import CatBoostRegressor, Pool

    c = cfg["gbdt"]["catboost"]
    store, y, w, work = load(paths)
    label = mtl_label(y, np.load(work / "aux.npy"), c["aux_lambda"], c["aux_clip"])
    t0 = time.time()
    pool = Pool(pd.DataFrame(store[:].T, copy=False), label=label, weight=w)
    pool.quantize(**c["quantize"], task_type="GPU")
    del store, label
    gc.collect()
    print(f"[cat] 量化 {time.time() - t0:.0f}s", flush=True)

    out = paths.stage("gbdt/models")
    paths.model.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        model = CatBoostRegressor(
            loss_function="MultiRMSE", eval_metric="MultiRMSE", **c["params"],
            random_seed=seed, iterations=c["rounds"], task_type="GPU", devices="0",
            ignored_features=unused_columns(cfg), allow_writing_files=False, verbose=0)
        model.fit(pool)
        model.save_model(str(out / f"cat_seed{seed}.cbm"))
        np.savez_compressed(paths.model / f"cat_seed{seed}.npz",
                            **export_catboost(model, n_columns(cfg)))
        print(f"[cat/seed{seed}] {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- XGBoost
def export_xgboost(booster, n_features: int) -> dict:
    """lossguide 树 → 节点数组；叶子自环，补齐区当叶子处理。"""
    learner = json.loads(booster.save_raw("json").decode("utf-8"))["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    base_score = float(str(learner["learner_model_param"]["base_score"]).strip("[]").split(",")[0])
    width = max(len(t["left_children"]) for t in trees)
    shape = (len(trees), width)
    left, right = np.empty(shape, np.int32), np.empty(shape, np.int32)
    sfeat, scond = np.zeros(shape, np.int32), np.zeros(shape, np.float32)
    dleft, value = np.zeros(shape, bool), np.zeros(shape, np.float64)
    is_leaf = np.ones(shape, bool)
    depth = 0
    for i, tree in enumerate(trees):
        lc = np.asarray(tree["left_children"], np.int32)
        rc = np.asarray(tree["right_children"], np.int32)
        n = len(lc)
        nodes = np.arange(width)
        left[i], right[i] = nodes, nodes
        left[i, :n] = np.where(lc < 0, nodes[:n], lc)
        right[i, :n] = np.where(lc < 0, nodes[:n], rc)
        sfeat[i, :n] = tree["split_indices"]
        # 内节点存阈值，叶节点存叶值
        scond[i, :n] = np.asarray(tree["split_conditions"], np.float32)
        value[i, :n] = np.asarray(tree["split_conditions"], np.float64)
        dleft[i, :n] = np.asarray(tree["default_left"], np.int32) != 0
        is_leaf[i, :n] = lc < 0
        depth = max(depth, tree_depth(lc, rc))
    return {"kind": np.array("xgboost"), "left": left, "right": right, "sfeat": sfeat,
            "scond": scond, "dleft": dleft, "value": value, "is_leaf": is_leaf,
            "max_depth": np.int64(depth), "base_score": np.float64(base_score),
            "n_features": np.int64(n_features)}


def tree_depth(lc: np.ndarray, rc: np.ndarray) -> int:
    depth, frontier = 0, [0]
    while frontier:
        nxt = [child for node in frontier for child in (lc[node], rc[node]) if child >= 0]
        if nxt:
            depth += 1
        frontier = nxt
    return depth


def train_xgboost(cfg: dict, paths: Paths, seeds: list[int]) -> None:
    import cupy as cp
    import xgboost as xgb

    c = cfg["gbdt"]["xgboost"]
    store, y, w, _ = load(paths)
    keep = used_columns(cfg)
    n_rows, step = len(y), c["chunk_rows"]
    cp.cuda.Device(0).use()
    cp.cuda.set_pinned_memory_allocator(None)

    class Chunks(xgb.DataIter):
        def __init__(self):
            self.lo = 0
            super().__init__(release_data=True)

        def reset(self):
            self.lo = 0

        def next(self, feed) -> bool:
            if self.lo >= n_rows:
                return False
            hi = min(self.lo + step, n_rows)
            block = cp.asarray(np.asarray(store[keep, self.lo:hi]))
            feed(data=cp.ascontiguousarray(block.T), label=cp.asarray(y[self.lo:hi]),
                 weight=cp.asarray(w[self.lo:hi]))
            self.lo = hi
            return True

    t0 = time.time()
    dtrain = xgb.QuantileDMatrix(Chunks(), max_bin=c["params"]["max_bin"])
    cp.get_default_memory_pool().free_all_blocks()
    print(f"[xgb] 建矩阵 {time.time() - t0:.0f}s", flush=True)

    out = paths.stage("gbdt/models")
    paths.model.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        params = {**c["params"], "seed": seed, "device": "cuda:0"}
        booster = xgb.train(params, dtrain, num_boost_round=c["rounds"])
        booster.save_model(str(out / f"xgb_seed{seed}.json"))
        np.savez_compressed(paths.model / f"xgb_seed{seed}.npz",
                            **export_xgboost(booster, len(keep)))
        print(f"[xgb/seed{seed}] {time.time() - t0:.0f}s", flush=True)
