"""D1 网络训练：一天一个 batch、AdamW + cosine、跑满步数后取 SWA 平均。

无验证集、无早停：交付的权重就是跑满 `steps` 时的 SWA 平均。
随机性来源与固定方式：
  - 参数初始化、dropout：torch.manual_seed(seed)（CPU 与 CUDA）
  - 每步取哪一天：np.random.default_rng(seed)
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch

from ..common import Paths
from . import cache
from .model import D1

DAY = 1440


def clock_table(n_clock: int) -> np.ndarray:
    """[1440, 4] fp16：日内相位的 sin/cos 一、二次谐波。"""
    if n_clock != 4:
        raise ValueError("只实现 4 列谐波时钟")
    theta = 2.0 * np.pi * np.arange(DAY, dtype=np.float64) / DAY
    cols = [np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta)]
    return np.stack(cols, axis=1).astype(np.float16)


def build_model(cfg: dict) -> D1:
    m = cfg["d1"]["model"]
    return D1(n_channels=325 + m["n_clock"], n_assets=cfg["n_assets"],
              seq_len=m["seq_len"], kernel_size=m["kernel_size"],
              emb_dim=m["emb_dim"], hidden=m["hidden"], dropout=m["dropout"])


class DayBatcher:
    """每步取一段 [context + 1440] 的时间片，计损失的部分恰好从日界开始。"""

    def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, clock: np.ndarray,
                 context: int, seed: int, device: torch.device):
        self.x, self.y, self.w, self.clock = x, y, w, clock
        self.span = context + DAY
        self.first = (-context) % DAY                       # (start + context) % DAY == 0
        self.n_starts = (len(x) - self.span - self.first) // DAY + 1
        self.rng = np.random.default_rng(seed)
        self.device = device

    def _upload(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)[None]).to(self.device)

    def next(self):
        start = self.first + DAY * int(self.rng.integers(0, self.n_starts, size=1)[0])
        stop = start + self.span
        clock = self.clock[np.arange(start, stop) % DAY]
        xb = np.concatenate(
            [self.x[start:stop], np.broadcast_to(clock[:, None, :], (self.span, self.x.shape[1], 4))],
            axis=2)
        # [1, T, A, F] -> [A, T, F]，资产放进 batch 维。这里有意保留 permute + reshape 得到的
        # 内存布局：布局不同时 GPU 选用的算子不同，结果末位会有差异，影响逐位复现。
        xb = self._upload(xb).permute(0, 2, 1, 3).reshape(-1, self.span, xb.shape[2]).float()
        yb = self._upload(self.y[start:stop]).permute(0, 2, 1).reshape(-1, self.span)
        wb = self._upload(self.w[start:stop]).permute(0, 2, 1).reshape(-1, self.span)
        return xb, yb, wb


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(cfg: dict, x, y, w, seed: int, device: torch.device) -> tuple[dict, dict]:
    t = cfg["d1"]["train"]
    seed_everything(seed)
    torch.set_num_threads(4)

    model = build_model(cfg).to(device)
    temporal = model.temporal_parameters()
    others = [p for p in model.parameters() if all(p is not q for q in temporal)]
    optimizer = torch.optim.AdamW(
        [{"params": temporal, "weight_decay": t["weight_decay"]},
         {"params": others, "weight_decay": t["weight_decay"]}], lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t["steps"], eta_min=t["lr"] * t["lr_min_ratio"])
    use_amp = t["amp"] and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

    context = model.context
    batcher = DayBatcher(x, y, w, clock_table(cfg["d1"]["model"]["n_clock"]),
                         context, seed, device)
    asset_id = torch.arange(cfg["n_assets"], device=device)

    swa_start = int(t["swa_start"] * t["steps"])
    swa_sum, swa_n = None, 0
    t0, running = time.time(), 0.0
    for step in range(1, t["steps"] + 1):
        model.train()
        xb, yb, wb = batcher.next()
        wb[:, :context] = 0.0                          # 历史不足的位置不计损失
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            pred = model(xb, asset_id)
            loss = torch.sum(wb * torch.square(yb - pred)) / torch.clamp_min(wb.sum(), 1e-12)
        if not torch.isfinite(loss):
            raise RuntimeError(f"seed={seed} step={step}：loss 非有限")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += float(loss.detach())

        if step >= swa_start and step % t["swa_every"] == 0:
            state = model.state_dict()
            if swa_sum is None:
                swa_sum = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state.items()}
            for k, acc in swa_sum.items():
                acc += state[k].detach().float()
            swa_n += 1
        if step % 2000 == 0:
            print(f"[d1/seed{seed}] step {step:>5d} loss {running / 2000:.6f} "
                  f"swa_n {swa_n} ({time.time() - t0:.0f}s)", flush=True)
            running = 0.0

    final = copy.deepcopy(model).cpu()
    final.load_state_dict({k: (acc / swa_n).cpu() for k, acc in swa_sum.items()})
    info = {"seed": seed, "steps": t["steps"], "swa_samples": swa_n,
            "seconds": round(time.time() - t0, 1)}
    return final.state_dict(), info


def run(cfg: dict, paths: Paths, seeds: list[int], device: str) -> None:
    out = paths.stage("d1/hosts")
    x, y, w = cache.load(paths, mmap=False)
    dev = torch.device(device)
    for seed in seeds:
        state, info = fit(cfg, x, y, w, seed, dev)
        torch.save({"state_dict": state, "info": info}, out / f"seed{seed}.pt")
        print(f"[d1/seed{seed}] 完成，SWA {info['swa_samples']} 个快照，{info['seconds']:.0f}s",
              flush=True)
