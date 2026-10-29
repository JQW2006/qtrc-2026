"""D1 在线引擎：由 `exp042/derive_d1.py` 从 d1_online_0825_m3/main.py 派生。

请勿手改本文件。改动只有四处（见 derive_d1.py 的模块说明），其余逐字节相同。
"""
import gc
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DAY = 1440
LABEL_LAG = 6
_HEAD_KEYS = ("W0", "b0", "W1", "b1")


class D1Online:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "d1_meta.json").read_text())
        blob = np.load(model_dir / "d1_model.npz")
        self.features = meta["features"]
        self.raw_dim = meta["raw_dim"]
        self.base_dim = meta["base_dim"]
        self.input_dim = meta["input_dim"]
        self.n_assets = int(meta["n_assets"])
        self.n_lag = int(meta["n_lag"])
        self.target_bound = float(meta["target_bound"])
        self.mean, self.scale = blob["mean"], blob["scale"]
        self.clip = float(blob["clip"])
        self.clock_table = blob["clock_table"]
        self.clock_period = int(blob["clock_period"])
        self.n_clock = int(meta["n_clock"])

        # ---- 成员数与逐成员的冻结前端
        self.M = int(meta["n_members"])
        self.composite = blob["composite"]           # [M, F, n_lag]
        self.emb = blob["emb"]                       # [M, A, E]
        if self.composite.shape != (self.M, self.input_dim, self.n_lag):
            raise ValueError(f"合成滤波器形状 {self.composite.shape} 与 meta 不符")
        if self.emb.shape[0] != self.M:
            raise ValueError(f"embedding 有 {self.emb.shape[0]} 份，期望 {self.M}")

        # ---- 逐成员的头：base 冻结 / live 在线 / shadow 是 EMA 后用于预测的
        self.base = [{k: blob[f"base_{k}_{m}"].astype(np.float32)
                      for k in _HEAD_KEYS} for m in range(self.M)]
        self.live = [{k: v.copy() for k, v in p.items()} for p in self.base]
        self.shadow = [{k: v.copy() for k, v in p.items()} for p in self.base]

        # ---- 伪标签重构器（全体共享）
        self.rec_w = blob["rec_weight"].astype(np.float32)
        self.rec_lags = int(meta["rec_max_lag"]) + 1
        if self.rec_lags != LABEL_LAG + 1:
            raise ValueError("重构器的最大 lag 必须等于 LABEL_LAG")
        if self.rec_w.shape[0] != self.base_dim * self.rec_lags + self.n_assets:
            raise ValueError("重构器维度与 meta 不符")

        # ---- 在线超参（全体共享，逐成员相同）
        h = meta["online"]
        self.blend = float(h["blend"])
        self.lr = float(h["lr"])
        self.b1, self.b2 = float(h["beta1"]), float(h["beta2"])
        self.eps = float(h["eps"])
        self.wd = float(h["weight_decay"])
        self.grad_clip = float(h["grad_clip"])
        self.n_steps = int(h["n_steps"])
        self.n_days = int(h["n_days"])
        self.day_decay = float(h["day_decay"])
        self.anchor = float(h["anchor"])
        self.wema = float(h["wema"])
        self.label_bound = float(h["label_bound"])
        self.label_clip = float(h["label_clip"])
        self.chunk_rows = int(meta["chunk_rows"])
        if self.chunk_rows % self.n_assets:
            raise ValueError("chunk_rows 必须是资产数的整数倍")
        self.buf_slack = int(meta.get("buf_slack", 0))
        self.budget_calls = int(meta.get("budget_calls", DAY))

        # ---- 缓冲：共享的一份 + 逐成员的 bz
        self.win = np.zeros((self.n_lag, self.n_assets, self.input_dim), np.float32)
        self.rx = np.zeros((self.rec_lags, self.n_assets, self.base_dim), np.float32)
        self.buf_len = self.n_days * DAY + LABEL_LAG + self.buf_slack
        self.bz = [np.zeros((self.buf_len, self.n_assets, self.input_dim), np.float32)
                   for _ in range(self.M)]
        self.bm = np.zeros((self.buf_len, self.n_assets), np.float32)
        self.by = np.zeros((self.buf_len, self.n_assets), np.float32)
        self.pos = -1
        self.n_seen = 0

        self.absent = np.zeros(self.base_dim, np.float32)
        self.absent[self.raw_dim] = 1.0
        self.absent[self.raw_dim + 1] = 1.0

        self.job = None
        self.all_assets = np.arange(self.n_assets)
        self.last_time_id = None
        # 每步做几个块步：按满负荷（窗口长满）估，保证日内做得完
        n_chunk_full = int(np.ceil(self.n_days * DAY * self.n_assets
                                   / self.chunk_rows))
        total_units = self.M * n_chunk_full * self.n_steps
        self.units_per_call = max(1, int(np.ceil(total_units
                                                 / max(self.budget_calls, 1))))
        z0 = np.zeros((self.n_assets, self.input_dim), np.float32)
        for m in range(self.M):                       # 预热 BLAS
            self._head_m(m, self.shadow[m], z0, self.all_assets)
        gc.collect()
        gc.freeze()
        gc.set_threshold(50_000, 1_000_000, 1_000_000)

    # ------------------------------------------------------------ 特征与前向

    def _features(self, test) -> np.ndarray:
        raw = test[self.features].to_numpy(dtype=np.float32)
        observed = np.isfinite(raw)
        mf = 1.0 - observed.mean(axis=1, dtype=np.float32)
        raw = np.where(observed, raw, np.nan)
        raw -= self.mean
        raw /= self.scale
        np.clip(raw, -self.clip, self.clip, out=raw)
        raw[~np.isfinite(raw)] = 0.0
        out = np.empty((len(raw), self.base_dim), np.float32)
        out[:, :self.raw_dim] = raw
        out[:, self.raw_dim] = mf
        out[:, self.raw_dim + 1] = mf == 1.0
        return out

    def _head_m(self, m: int, p, z: np.ndarray, asset: np.ndarray) -> np.ndarray:
        """z [A, F] -> 拼成员 m 的 embedding -> Linear-ReLU-Linear。"""
        hin = np.concatenate([z, self.emb[m][asset]], axis=1)
        a = np.maximum(hin @ p["W0"].T + p["b0"], 0.0)
        return (a @ p["W1"].T + p["b1"])[:, 0]

    def _window(self) -> np.ndarray:
        if self.n_seen >= self.n_lag or self.n_seen == 0:
            return self.win
        first = self.n_lag - self.n_seen
        window = self.win.copy()
        window[:first] = self.win[first]
        return window

    # ------------------------------------------------------------ 环形缓冲

    def _push(self, base_x: np.ndarray, mask: np.ndarray, time_id: int,
              need_z: bool = True) -> np.ndarray | None:
        """推入一帧，返回 [M, A, F]：每个成员用自己的滤波器滤出的 z。

        `need_z=False` 时**跳过滤波**、往 bz 写零并返回 None。只在补缺席帧
        （`_push_absent`）时这么用，此时该帧的 z 与任何结果无关：

          - 返回值被 `_push_absent` 丢弃，不进预测；
          - 该帧在天末更新里的权重是 `bm[slot]·decay = 0`（缺席 ⇒ mask=0），
            于是 `g = -2·w·(y-out)/total = 0`，四个梯度张量的贡献逐个为零
            （`W1←gᵀa`、`b1←Σg`、`W0←dhᵀhin`、`b0←Σdh`，而 dh 由 g 导出）。
          - 写**零**而不是留旧值：留旧值时 out 可能溢出成 inf，`0×inf = nan`
            会污染整份梯度。写零 ⇒ out=b1 有限 ⇒ g=0。

        为什么需要它：test 里有两处 1445 个 time_id 的间隙，间隙后的第一次
        `predict()` 要一口气补 1445 帧。M=1 时每帧一次 einsum 约 0.14s 还扛得住，
        M=8 时是 8 倍 ⇒ 实测 1.089s，撞穿 0.5s 单步上限（本 exp 首轮全量跑出的
        那 2 次超时就是这两帧）。跳过滤波把这块成本整个消掉。
        """
        self.rx[:-1] = self.rx[1:]
        self.rx[-1] = base_x

        self.win[:-1] = self.win[1:]
        self.win[-1, :, :self.base_dim] = base_x
        if self.n_clock:
            self.win[-1, :, self.base_dim:] = \
                self.clock_table[time_id % self.clock_period]
        self.n_seen += 1

        self.pos = (self.pos + 1) % self.buf_len
        if not need_z:
            for m in range(self.M):
                self.bz[m][self.pos] = 0.0
            self.bm[self.pos] = mask
            return None
        z = np.einsum("daf,mfd->maf", self._window(), self.composite)   # [M, A, F]
        for m in range(self.M):
            self.bz[m][self.pos] = z[m]
        self.bm[self.pos] = mask
        return z

    def _idx(self, back: int) -> int:
        return (self.pos - back) % self.buf_len

    # ------------------------------------------------------------ 伪标签

    def _pseudo_label(self) -> np.ndarray:
        C = self.base_dim
        feat = np.empty((self.n_assets, C * self.rec_lags), np.float32)
        for j in range(self.rec_lags):
            feat[:, j * C:(j + 1) * C] = self.rx[j]
        y = feat @ self.rec_w[:C * self.rec_lags] + self.rec_w[C * self.rec_lags:]
        if self.label_bound > 0:
            np.clip(y, -self.label_bound, self.label_bound, out=y)
        return y

    def _label_lagged_frame(self) -> None:
        if self.n_seen > LABEL_LAG:
            self.by[self._idx(LABEL_LAG)] = self._pseudo_label()

    # ------------------------------------------------------------ 天末更新

    def _open_job(self) -> None:
        """快照更新窗（全体成员共享），并为每个成员建一份优化器状态。"""
        n_rows_time = min(self.n_days * DAY, self.n_seen - LABEL_LAG)
        if n_rows_time <= 0:
            return
        slots = np.asarray([self._idx(LABEL_LAG + k)
                            for k in range(n_rows_time)][::-1])
        age = ((n_rows_time - 1 - np.arange(n_rows_time)) // DAY).astype(np.float32)
        y = self.by[slots]
        w = self.bm[slots] * (self.day_decay ** age)[:, None]
        if self.label_clip > 0:
            w = w * (np.abs(y) <= self.label_clip)
        total = float(w.sum())
        if total <= 0:
            return
        n_chunk = int(np.ceil(n_rows_time * self.n_assets / self.chunk_rows))
        self.job = {
            "slots": slots, "w": w, "y": y, "total": total, "n_chunk": n_chunk,
            "cur": 0,
            "mem": [{"grad": None, "step": 0, "chunk": 0, "t": 0,
                     "m": {k: np.zeros_like(v) for k, v in self.live[i].items()},
                     "v": {k: np.zeros_like(v) for k, v in self.live[i].items()}}
                    for i in range(self.M)]}

    def _accumulate(self, job, i: int) -> None:
        st = job["mem"][i]
        A = self.n_assets
        lo = st["chunk"] * self.chunk_rows
        hi = min(lo + self.chunk_rows, len(job["slots"]) * A)
        t_lo, t_hi = lo // A, hi // A
        slots = job["slots"][t_lo:t_hi]
        nt = len(slots)
        z = self.bz[i][slots].reshape(nt * A, self.input_dim)
        y = job["y"][t_lo:t_hi].reshape(-1)
        w = job["w"][t_lo:t_hi].reshape(-1)
        ai = np.tile(self.all_assets, nt)

        p = self.live[i]
        hin = np.concatenate([z, self.emb[i][ai]], axis=1)
        h = hin @ p["W0"].T + p["b0"]
        a = np.maximum(h, 0.0)
        out = (a @ p["W1"].T + p["b1"])[:, 0]

        g = (-2.0 * w * (y - out) / job["total"]).astype(np.float32)
        gcol = g[:, None]
        da = gcol @ p["W1"]
        dh = np.where(h > 0.0, da, 0.0).astype(np.float32)
        add = {"W1": gcol.T @ a, "b1": np.array([g.sum()], np.float32),
               "W0": dh.T @ hin, "b0": dh.sum(axis=0)}
        if st["grad"] is None:
            st["grad"] = add
        else:
            for k in st["grad"]:
                st["grad"][k] += add[k]

    def _apply_step(self, job, i: int) -> None:
        st = job["mem"][i]
        grad = st["grad"]
        live = self.live[i]
        if self.wd:
            for k in grad:
                grad[k] = grad[k] + self.wd * live[k]
        if self.grad_clip > 0:
            total = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum())
                                      for g in grad.values())))
            if total > self.grad_clip:
                s = np.float32(self.grad_clip / (total + 1e-6))
                for k in grad:
                    grad[k] = grad[k] * s
        st["t"] += 1
        bc1 = 1.0 - self.b1 ** st["t"]
        bc2 = 1.0 - self.b2 ** st["t"]
        for k in live:
            m, v = st["m"][k], st["v"][k]
            m *= self.b1
            m += (1.0 - self.b1) * grad[k]
            v *= self.b2
            v += (1.0 - self.b2) * grad[k] ** 2
            live[k] -= (self.lr * (m / bc1) /
                        (np.sqrt(v / bc2) + self.eps)).astype(np.float32)
        st["grad"] = None
        st["chunk"] = 0
        st["step"] += 1

    def _finish_member(self, i: int) -> None:
        """成员 i 的 n_steps 步做完：锚回 -> EMA 进影子 -> 原子换入。"""
        if self.anchor > 0:
            for k in self.live[i]:
                self.live[i][k] *= (1.0 - self.anchor)
                self.live[i][k] += self.anchor * self.base[i][k]
        for k in self.shadow[i]:
            self.shadow[i][k] *= (1.0 - self.wema)
            self.shadow[i][k] += self.wema * self.live[i][k]

    def _advance_job(self) -> None:
        """做 units_per_call 个块步。成员按顺序推进，做完一个换下一个。"""
        job = self.job
        if job is None:
            return
        for _ in range(self.units_per_call):
            i = job["cur"]
            if i >= self.M:
                self.job = None
                return
            self._accumulate(job, i)
            st = job["mem"][i]
            st["chunk"] += 1
            if st["chunk"] >= job["n_chunk"]:
                self._apply_step(job, i)
                if st["step"] >= self.n_steps:
                    self._finish_member(i)
                    job["cur"] += 1
                    if job["cur"] >= self.M:
                        self.job = None
                        return

    # ------------------------------------------------------------ 协议入口

    def _push_absent(self, time_id: int) -> None:
        self._push(np.broadcast_to(self.absent,
                                   (self.n_assets, self.base_dim)).copy(),
                   np.zeros(self.n_assets, np.float32), time_id, need_z=False)
        self._label_lagged_frame()

    def predict(self, test):
        time_col = test["time_id"]
        time_id = int(time_col.iloc[0]) if hasattr(time_col, "iloc") \
            else int(np.asarray(time_col)[0])

        if self.last_time_id is not None:
            gap = min(max(time_id - self.last_time_id - 1, 0), self.buf_len)
            for k in range(gap, 0, -1):
                self._push_absent(time_id - k)
        self.last_time_id = time_id

        asset = test["asset_id"].to_numpy().astype(np.int64)
        base_x = np.broadcast_to(self.absent,
                                 (self.n_assets, self.base_dim)).copy()
        mask = np.zeros(self.n_assets, np.float32)
        base_x[asset] = self._features(test)
        mask[asset] = 1.0
        z = self._push(base_x, mask, time_id)

        self._label_lagged_frame()

        acc_host = np.zeros(self.n_assets, np.float32)
        acc_online = np.zeros(self.n_assets, np.float32)
        for m in range(self.M):
            acc_online += self._head_m(m, self.shadow[m], z[m], self.all_assets)
            acc_host += self._head_m(m, self.base[m], z[m], self.all_assets)
        host = (acc_host / self.M)[asset].astype(np.float64)
        online = (acc_online / self.M)[asset].astype(np.float64)

        if self.job is None and time_id % DAY == DAY - 1:
            self._open_job()
        self._advance_job()
        # blend 由融合权重吸收；nan_to_num / clip 在组合器里合并之后做一次。
        return host, online
