"""伪标签重构器：用 f(t), f(t+1), ..., f(t+6) 线性重构 target(t)。

推理期在 time_id T 时用已经看到的 f(T-6..T) 给 T-6 打标签，供天末在线更新使用。
设计矩阵 = 7 帧 × 325 列 + 15 个资产哑变量（兼作截距），岭回归 λ=10，
哑变量同样受惩罚。正规方程在 GPU 上以 float32 分块累加，
求解在 CPU 上以 float64 做特征分解。
"""
from __future__ import annotations

import numpy as np
import torch

from ..common import Paths, save_npy
from . import cache

DAY = 1440


def design(x: torch.Tensor, n: int, lags: list[int]) -> torch.Tensor:
    """x [n+max_lag, A, C] → [n·A, C·len(lags) + A]，行序 (时间, 资产)。"""
    n_assets = x.shape[1]
    feat = torch.cat([x[j:j + n] for j in lags], dim=2).reshape(n * n_assets, -1)
    dummies = torch.eye(n_assets, device=x.device, dtype=x.dtype).repeat(n, 1)
    return torch.cat([feat, dummies], dim=1)


def fit(cfg: dict, paths: Paths, device: str) -> np.ndarray:
    r = cfg["d1"]["reconstructor"]
    dev = torch.device(device)
    x_all, y_all, _ = cache.load(paths)
    end = r["days"] * DAY                       # 只用 0630 训练集
    lags = list(range(r["max_lag"] + 1))
    n_assets, n_col = x_all.shape[1], x_all.shape[2]
    d = n_col * len(lags) + n_assets
    G = torch.zeros((d, d), dtype=torch.float32, device=dev)
    c = torch.zeros(d, dtype=torch.float32, device=dev)
    for day in range(0, r["days"], r["block_days"]):
        lo = day * DAY
        hi = min(day + r["block_days"], r["days"]) * DAY
        if hi + r["max_lag"] > end:             # 取不到 t+max_lag 的块整块丢弃
            break
        xb = torch.from_numpy(np.array(x_all[lo:hi + r["max_lag"]])).to(dev).float()
        yb = torch.from_numpy(np.array(y_all[lo:hi])).to(dev).float()
        n = hi - lo
        idx = torch.arange(0, n, r["stride"], device=dev)
        X = design(xb, n, lags).reshape(n, n_assets, -1)[idx].reshape(-1, d)
        yv = yb[idx].reshape(-1)
        G += X.T @ X
        c += X.T @ yv
    evals, evecs = torch.linalg.eigh(G.double().cpu())
    w = evecs @ ((evecs.T @ c.double().cpu()) / (evals + r["lam"]))
    weight = w.float().numpy()
    save_npy(paths.stage("d1") / "rec_weight.npy", weight)
    print(f"[d1/reconstructor] 维度 {d}，‖w‖={float(np.linalg.norm(weight)):.4f}", flush=True)
    return weight
