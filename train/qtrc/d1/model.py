"""D1：逐通道因果滤波前端 + 资产 embedding + 单隐层 MLP 头。

前端是 DLinear 的因果版本：x 先拆成因果移动平均（trend）与残差（seasonal），
两支各过一个逐通道、长度 L 的时间滤波器后相加。两支线性叠加等价于一个长度
L+k-1 的合成滤波器（`composite_filters`），推理包只存合成后的系数。

参数的创建顺序（embedding → Linear → Linear）决定初始化时随机数的消耗顺序，
调整顺序会改变训练结果。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def causal_moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
    """[B, C, T] 的 trailing 均值，左端 replicate 填充。"""
    padded = F.pad(x, (k - 1, 0), mode="replicate")
    csum = F.pad(torch.cumsum(padded, dim=-1), (1, 0))
    return (csum[..., k:] - csum[..., :-k]) / k


class D1(nn.Module):
    def __init__(self, n_channels: int, n_assets: int, seq_len: int, kernel_size: int,
                 emb_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.seq_len, self.kernel_size = seq_len, kernel_size
        self.seasonal_weight = nn.Parameter(torch.empty(n_channels, 1, seq_len))
        self.trend_weight = nn.Parameter(torch.empty(n_channels, 1, seq_len))
        with torch.no_grad():
            # 起点：两支相同 ⇒ 合成滤波器 = 单位脉冲（只看当前步）
            for weight in (self.seasonal_weight, self.trend_weight):
                weight.zero_()
                weight[..., -1] = 1.0
        self.asset_embedding = nn.Embedding(n_assets, emb_dim)
        # Identity 仅作占位，使隐层、输出层的下标固定为 head.1、head.5，导出时按此读取
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n_channels + emb_dim, hidden), nn.Identity(),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    @property
    def context(self) -> int:
        """预测一个位置需要的历史步数（不含当前步）。"""
        return self.seq_len + self.kernel_size - 2

    def temporal_parameters(self) -> list[nn.Parameter]:
        return [self.seasonal_weight, self.trend_weight]

    def _filter(self, series: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        padded = F.pad(series, (self.seq_len - 1, 0), mode="replicate")
        return F.conv1d(padded, weight, groups=series.shape[1])

    def forward(self, x: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        """x [B, T, C]，asset_id [B] → [B, T]，位置 t 只用 x[:, :t+1]。"""
        series = x.transpose(1, 2)
        trend = causal_moving_average(series, self.kernel_size)
        z = self._filter(series - trend, self.seasonal_weight) + \
            self._filter(trend, self.trend_weight)
        z = z.transpose(1, 2)
        emb = self.asset_embedding(asset_id)
        z = torch.cat([z, emb.unsqueeze(1).expand(-1, z.shape[1], -1)], dim=-1)
        n_batch, n_time = z.shape[:2]
        return self.head(z.reshape(n_batch * n_time, -1)).reshape(n_batch, n_time)

    @torch.no_grad()
    def composite_filters(self) -> torch.Tensor:
        """[C, L+k-1] 的等价单滤波器，最后一列对应当前步。

        z[t] = Σ_i w_s[i]·x[t-L+1+i] + Σ_i (w_t[i]-w_s[i])·MA_k(x)[t-L+1+i]，
        把 MA 展开即得：滞后 L-1-i 处加 w_s[i]，滞后 L-1-i+j (j<k) 处各加 (w_t[i]-w_s[i])/k。
        """
        k, L = self.kernel_size, self.seq_len
        w_t = self.trend_weight[:, 0, :]
        w_s = self.seasonal_weight[:, 0, :]
        comp = torch.zeros(w_t.shape[0], L + k - 1, dtype=w_t.dtype)
        diff = (w_t - w_s) / k
        for i in range(L):
            lag = L - 1 - i
            comp[:, lag] += w_s[:, i]
            comp[:, lag:lag + k] += diff[:, i:i + 1]
        return comp.flip(-1).clone()
