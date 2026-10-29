"""把 D1 的训练结果写成推理包读取的 model/d1_model.npz 与 model/d1_meta.json。"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import torch

from ..common import FEATURES, Paths, read_json
from .train import build_model, clock_table

DAY = 1440


def run(cfg: dict, paths: Paths) -> None:
    c = cfg["d1"]
    work = paths.work / "d1"
    seeds = c["train"]["seeds"]

    comps, embs, heads, members = [], [], [], []
    for seed in seeds:
        ck = torch.load(work / "hosts" / f"seed{seed}.pt", map_location="cpu", weights_only=False)
        model = build_model(cfg)
        model.load_state_dict(ck["state_dict"])
        sd = {k: v.numpy().astype(np.float32) for k, v in ck["state_dict"].items()}
        comps.append(model.composite_filters().numpy().astype(np.float32))
        embs.append(sd["asset_embedding.weight"])
        heads.append({"W0": sd["head.1.weight"], "b0": sd["head.1.bias"],
                      "W1": sd["head.5.weight"], "b1": sd["head.5.bias"]})
        members.append({k: ck["info"][k] for k in ("seed", "steps", "swa_samples")})

    scaler = np.load(work / "scaler.npz")
    rec_w = np.load(work / "rec_weight.npy").astype(np.float32)
    table = clock_table(c["model"]["n_clock"]).astype(np.float32)
    payload = {"composite": np.stack(comps), "emb": np.stack(embs),
               "mean": scaler["mean"], "scale": scaler["scale"],
               "clip": np.float32(c["clip"]), "clock_table": table,
               "clock_period": np.int64(table.shape[0]), "rec_weight": rec_w}
    for m, head in enumerate(heads):
        for k, v in head.items():
            payload[f"base_{k}_{m}"] = v
    paths.model.mkdir(parents=True, exist_ok=True)
    np.savez(paths.model / "d1_model.npz", **payload)

    target_bound = read_json(work / "stats.json")["target_bound"]
    model_cfg = c["model"]
    n_lag = model_cfg["seq_len"] + model_cfg["kernel_size"] - 1
    online = {k: v for k, v in c["online"].items() if k != "label_clip"}
    online.update(label_bound=target_bound, label_clip=c["online"]["label_clip"])
    n_chunk = int(np.ceil(online["n_days"] * DAY * cfg["n_assets"] / c["chunk_rows"]))
    units = len(seeds) * n_chunk * int(online["n_steps"])
    meta = {
        "features": FEATURES, "raw_dim": 323, "base_dim": 325,
        "input_dim": 325 + model_cfg["n_clock"], "n_assets": cfg["n_assets"],
        "n_clock": model_cfg["n_clock"], "n_lag": n_lag, "context": n_lag - 1,
        "target_bound": target_bound, "asset_embedding_dim": model_cfg["emb_dim"],
        "name": "d1-online-0825", "kind": "dlinear_d1_online_ens",
        "built": date.today().isoformat(), "arm": "D1_day_tod",
        "n_members": len(seeds), "members": members,
        "train_data": ("public_release_20260630/train + public_release_20260823/train"
                       "（标签回补），time_id 0–1105919，共 768 天"),
        "recipe": "d1_online_0825", "online": online,
        "chunk_rows": c["chunk_rows"], "buf_slack": c["buf_slack"],
        "budget_calls": c["budget_calls"], "n_chunk_per_member": n_chunk,
        "total_units": units,
        "units_per_call": max(1, int(np.ceil(units / c["budget_calls"]))),
        "rec_max_lag": c["reconstructor"]["max_lag"],
    }
    (paths.model / "d1_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"[d1/export] {len(seeds)} 个成员 -> {paths.model}", flush=True)
