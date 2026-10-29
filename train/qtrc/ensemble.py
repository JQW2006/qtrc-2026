"""写 model/ensemble.json：融合权重、GBDT 输入列、输出限幅与各权重文件的 md5。

  pred = 0.30·CatBoost + 0.30·XGBoost + 0.15·D1 冻结头 + 0.25·D1 在线头
每一路都是 3 个种子的等权平均；融合权重是固定常数，取自 config.yaml。
"""
from __future__ import annotations

import json

import numpy as np

from .common import FEATURES, Paths, md5, read_json
from .gbdt.features import n_columns, used_columns


def run(cfg: dict, paths: Paths) -> None:
    seeds = cfg["gbdt"]["seeds"]
    cat = cfg["gbdt"]["catboost"]
    digests, shapes = {}, {}
    for s in seeds:
        for engine in ("cat", "xgb"):
            p = paths.model / f"{engine}_seed{s}.npz"
            digests[p.name] = md5(p)
            blob = np.load(p)
            shapes[p.name] = ({"trees": int(blob["feat"].shape[0]),
                               "depth": int(blob["feat"].shape[1])} if engine == "cat"
                              else {"trees": int(blob["left"].shape[0]),
                                    "max_depth": int(blob["max_depth"])})
    payload = {
        "name": "d1-gbdt-ens-mtl-0830",
        "weights": cfg["ensemble"]["weights"],
        "gbdt_seeds": seeds,
        "n_group_columns": n_columns(cfg),
        "xgb_keep_columns": used_columns(cfg).tolist(),
        "target_bound": read_json(paths.work / "d1/stats.json")["target_bound"],
        "max_threads": cfg["ensemble"]["max_threads"],
        "raw_features": FEATURES,
        "provenance": {"train": "train/ 在 public_release_20260630 + public_release_20260823 上重训"},
        "digests": digests,
        "gbdt_shapes": shapes,
        "mtl": {"loss": "MultiRMSE", "aux_responders": [cat["aux_responder"]],
                "lam": cat["aux_lambda"], "clip": cat["aux_clip"], "n_out": 2,
                "arm": "mr_r02_l8"},
    }
    (paths.model / "ensemble.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ensemble] 权重 {payload['weights']} -> {paths.model / 'ensemble.json'}", flush=True)
