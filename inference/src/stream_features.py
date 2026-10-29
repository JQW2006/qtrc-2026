"""GBDT 那 490 列特征的**在线**构建：每个 time_id 一步，O(1) 状态，零前瞻。

离线正本是 exp003 `build_cache.py`（404 基座）+ exp006 `build_group.py`（派生块）。
本模块必须在语义上与它们逐格一致——正确性由 `parity_stream.py` 拿真实验证窗对拍，
不靠"读起来像"。以下把每一块的离线语义抄在旁边，因为它们各有各的坑。

列布局（与 exp008 `keep_columns()` 反解出的段一致）
    [0,323)   原始 feature_000..322，直接取自 test
    [323,403) TOP40 × {d1, dev60}，**交错**排列：323+2k = d1, 323+2k+1 = dev60
    [403]     asset_id
    [404,444) xzs@TOP40      当期截面 z-score
    [444,484) xbeta128@TOP40 ← 该臂忽略，恒 0
    [484,524) xbeta512@TOP40 对截面均值的滚动 beta
    [524,564) xdmdev512@TOP40 ← 该臂忽略，恒 0
    [564,570) clock3：sin/cos 的 1~3 次谐波

两套完全不同的"时间轴"，这是本模块最容易搞错的地方
------------------------------------------------------------------
1. **d1 / dev60 走"该资产的观测序列"**（polars `.over("asset_id")`）：
   资产缺席的 time_id 根本不进这条序列。`diff()` 是与该资产**上一行**相减，
   `rolling_mean(60, min_samples=10)` 是该资产**最近 60 行**里的非空值均值。
   源数据里的缺失是 **null 不是 NaN**（已核对 parquet：null_count>0, nan_count=0），
   所以 polars 的语义是"跳过空值、min_samples 数非空个数"，而不是 NaN 传播。
   落到 numpy 缓存里 null 变成 NaN，评测 API 给的也是 NaN——两者一一对应。

2. **xzs / xbeta512 走"稠密时间网格"**（exp006 的 [T, A] 面板）：
   每个 time_id 都是一格，**包括整段缺席的 time_id**；缺席格与非有限格一律
   前向填充（"这一刻没有新观测，状态原样保留"）。所以流里出现 time_id 跳空时，
   必须把跳过的每一格都补进 EWMA 状态，否则相位就对不上了。
   跳空段里全体缺席 ⇒ 截面均值 m = 0（`ctx._n = max(present.sum(), 1)`，
   分子是 `col * present` 全零），可以用一阶 IIR 的闭式解 O(1) 跳过，
   不必真的循环 1445 次。

3. **截面统计只用"在场"资产**，但用的是**面板里的值**——即在场却取到 NaN 的资产，
   进截面时用的是它前向填充后的旧值，而不是被剔除。这两件事的掩码不是同一个。

冷启动与离线的唯一差别
------------------------------------------------------------------
离线 `_ffill_bfill` 在段首用"该资产**第一个**观测值"回填还没出现过的格子——
那是一次跨时间的回看，在线做不到也不该做。本模块的规则是：还没见过任何观测的
资产，面板值取 0（同时 `seen=False`，不进截面统计）。这只影响一个资产在流里
首次出现之前的那些格，私榜的暖机行会把这段吃掉。差异范围由 `parity_stream.py`
量化（验证窗第一帧全员在场时，两者完全一致）。
"""
from __future__ import annotations

import numpy as np

N_ASSETS = 15
N_RAW = 323
N_BASE = 404
N_COLS = 570
DAY = 1440
EPS = 1e-12
ROLL_WINDOW = 60
ROLL_MIN = 10
BETA_HALFLIFE = 512.0

# exp006 `groups.BASE40` = `src.features.F01_TOP40`；同一批 40 列既做 d1/dev60，
# 也做 xzs/xbeta512（已核对 `F01_TOP40 == F02_STABLE[:40]`）。
TOP40 = (15, 39, 36, 14, 219, 157, 34, 50, 151, 127, 13, 308, 93, 187, 287, 305,
         47, 135, 286, 90, 148, 284, 188, 248, 42, 95, 228, 256, 129, 244, 281,
         78, 144, 250, 76, 318, 258, 120, 96, 9)

COL_TS = 323          # [323, 403) d1/dev60 交错
COL_ASSET = 403
COL_XZS = 404         # [404, 444)
COL_XBETA512 = 484    # [484, 524)
COL_CLOCK = 564       # [564, 570)


def halflife_alpha(halflife: float) -> float:
    return float(1.0 - 2.0 ** (-1.0 / halflife))


class FeatureStream:
    """逐 time_id 推进的 570 列构建器。

    用法：每次 `predict(test)` 调用一次 `step(time_id, asset_id, raw)`，
    返回 [n_rows, 570] float32。被忽略的两块（xbeta128 / xdmdev512）恒为 0——
    CatBoost 的 `ignored_features` 保证模型从不在它们上分裂，喂 0 与喂真值等价；
    xgb 那一侧只吃 490 列子集，根本看不到它们。
    """

    def __init__(self, n_assets: int = N_ASSETS):
        self.n_assets = n_assets
        self.top40 = np.asarray(TOP40, dtype=np.int64)
        self.n_top = len(self.top40)
        self.alpha = halflife_alpha(BETA_HALFLIFE)

        # ---- 观测序列状态（d1 / dev60）：按"该资产的行"推进 ----
        self.prev_row = np.full((n_assets, self.n_top), np.nan, np.float32)
        self.ring = np.zeros((n_assets, ROLL_WINDOW, self.n_top), np.float64)
        self.ring_ok = np.zeros((n_assets, ROLL_WINDOW, self.n_top), bool)
        self.ring_pos = np.zeros(n_assets, np.int64)     # 下一个写入位置

        # ---- 稠密网格状态（xzs / xbeta512）：按 time_id 推进 ----
        # 布局是 **[特征, 资产]** 而不是 [资产, 特征]：离线的截面统计是在 [T, A]
        # 上沿**最后一维**（连续的 15 个 float32）求和的，numpy 的归约对连续轴与
        # 跨步轴用不同的分块，末位会差。把资产放最后一维、且保持连续，
        # 求和的内层循环就和离线逐字相同。实测这一条把 xzs 的差从 7e-6 压到 0。
        self.panel = np.zeros((self.n_top, n_assets), np.float32)   # 前向填充后的值
        self.seen = np.zeros((self.n_top, n_assets), bool)
        self.e_col = np.zeros((self.n_top, n_assets), np.float64)
        self.e_m = np.zeros((self.n_top, n_assets), np.float64)
        self.e_cm = np.zeros((self.n_top, n_assets), np.float64)
        self.e_mm = np.zeros((self.n_top, n_assets), np.float64)
        self.ewma_init = False
        self.last_tid: int | None = None

        self._harm = np.arange(1, 4, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def _advance_gap(self, n_steps: int) -> None:
        """把 n_steps 个"全体缺席"的 time_id 一次性推进（闭式解，O(1)）。

        全体缺席 ⇒ 面板值不变（前向填充）、截面均值 m ≡ 0。一阶 IIR
        `y ← a·x + (1−a)·y` 在 x 恒定时的 n 步闭式解是
            y_n = x + (y_0 − x)·(1−a)^n
        m 恒为 0 的三个状态就是纯衰减 `y_n = y_0·(1−a)^n`。
        """
        if n_steps <= 0 or not self.ewma_init:
            return
        decay = (1.0 - self.alpha) ** n_steps
        col = self.panel.astype(np.float64)
        self.e_col = col + (self.e_col - col) * decay
        self.e_m = self.e_m * decay
        self.e_cm = self.e_cm * decay
        self.e_mm = self.e_mm * decay

    # ------------------------------------------------------------------ #
    def step(self, time_id: int, asset_id: np.ndarray, raw: np.ndarray) -> np.ndarray:
        """raw: [n_rows, 323] float32（NaN 表示源数据里的 null）。"""
        time_id = int(time_id)
        asset_id = np.asarray(asset_id, dtype=np.int64)
        raw = np.asarray(raw, dtype=np.float32)
        n = len(asset_id)
        if raw.shape != (n, N_RAW):
            raise ValueError(f"raw 形状 {raw.shape}，期望 ({n}, {N_RAW})")

        # 跳空：离线面板把跳过的 time_id 当成"全体缺席"的格，必须补上
        if self.last_tid is not None and time_id > self.last_tid + 1:
            self._advance_gap(time_id - self.last_tid - 1)
        self.last_tid = time_id

        out = np.zeros((n, N_COLS), dtype=np.float32)
        out[:, :N_RAW] = raw
        out[:, COL_ASSET] = asset_id.astype(np.float32)

        top = raw[:, self.top40]                     # [n, 40]

        # ---------------- 1. d1 / dev60：按该资产的观测序列 ---------------- #
        d1 = top - self.prev_row[asset_id]
        self.prev_row[asset_id] = top

        ok = np.isfinite(top)
        pos = self.ring_pos[asset_id]
        self.ring[asset_id, pos] = np.where(ok, top, 0.0)
        self.ring_ok[asset_id, pos] = ok
        self.ring_pos[asset_id] = (pos + 1) % ROLL_WINDOW

        buf = self.ring[asset_id]                    # [n, 60, 40]
        okb = self.ring_ok[asset_id]
        # 没写过的槽位 ring_ok 恒为 False，所以 cnt 天然只数"最近 60 行里的非空值"，
        # 不需要单独维护"已写入多少行"。
        cnt = okb.sum(axis=1)                        # [n, 40]
        total = np.where(okb, buf, 0.0).sum(axis=1)
        mean = total / np.maximum(cnt, 1)
        dev60 = np.where(cnt >= ROLL_MIN, top - mean.astype(np.float32), np.nan)

        out[:, COL_TS:COL_TS + 2 * self.n_top:2] = d1
        out[:, COL_TS + 1:COL_TS + 2 * self.n_top:2] = dev60

        # ---------------- 2. 面板前向填充（含"在场但取到 NaN"） ---------------- #
        # 面板是 [特征, 资产]，所以这里写入的是 top 的转置。
        fresh = np.zeros((self.n_top, self.n_assets), bool)
        fresh[:, asset_id] = ok.T
        vals = np.zeros((self.n_top, self.n_assets), np.float32)
        vals[:, asset_id] = top.T
        self.panel = np.where(fresh, vals, self.panel)
        self.seen |= fresh

        # ---------------- 3. 当期截面统计（只用在场资产，值取面板） ---------------- #
        present = np.zeros(self.n_assets, bool)
        present[asset_id] = True
        pm = present[None, :].astype(np.float32)
        denom = np.float32(max(int(present.sum()), 1))
        pmean = (self.panel * pm).sum(axis=1, keepdims=True) / denom          # [40, 1]
        pvar = (((self.panel - pmean) ** 2) * pm).sum(axis=1, keepdims=True) / denom
        pstd = np.sqrt(np.maximum(pvar, 0.0)).astype(np.float32)

        xzs_full = np.where(pstd > EPS,
                            (self.panel - pmean) / np.maximum(pstd, EPS), 0.0)
        out[:, COL_XZS:COL_XZS + self.n_top] = \
            xzs_full[:, asset_id].T.astype(np.float32)

        # ---------------- 4. xbeta512 ---------------- #
        # 四个 EWMA 的**输入**在离线侧都是 float32 数组（面板与截面均值都是
        # float32，乘法也在 float32 里做），`ewma()` 才把它们抬到 float64 去滤。
        # 所以这里的乘法必须先在 float32 里算完再抬，不能拿 float64 的 col/m 相乘——
        # 否则 e_cm / e_mm 的输入就和离线差一个舍入，beta 会慢慢分叉。
        col32 = self.panel
        m32 = np.broadcast_to(pmean, col32.shape)
        col = col32.astype(np.float64)
        m = m32.astype(np.float64)
        cm = (col32 * m32).astype(np.float64)
        mm = (m32 * m32).astype(np.float64)
        if not self.ewma_init:
            # 离线 `ewma` 的 zi 让 y[0] = x[0]；在线的第一格照做。
            self.e_col, self.e_m, self.e_cm, self.e_mm = col.copy(), m.copy(), cm, mm
            self.ewma_init = True
        else:
            # 与 `lfilter(b=[a], a=[1, -(1-a)])` 同一种写法：先乘后加。
            # 写成 `e += a*(x-e)` 在代数上等价、在 float64 舍入上不等价。
            a, one_a = self.alpha, 1.0 - self.alpha
            self.e_col = a * col + one_a * self.e_col
            self.e_m = a * m + one_a * self.e_m
            self.e_cm = a * cm + one_a * self.e_cm
            self.e_mm = a * mm + one_a * self.e_mm

        # 离线每个 ewma 的输出在参与算术前已经被 cast 成 float32，必须照做，
        # 否则 cov/var 的舍入路径不同，beta 会在小数末几位分叉。
        e_col = self.e_col.astype(np.float32)
        e_m = self.e_m.astype(np.float32)
        e_cm = self.e_cm.astype(np.float32)
        e_mm = self.e_mm.astype(np.float32)
        cov = e_cm - e_col * e_m
        var = np.maximum(e_mm - e_m * e_m, np.float32(0.0))
        beta = np.where(var > EPS, cov / np.maximum(var, np.float32(EPS)), 0.0)
        out[:, COL_XBETA512:COL_XBETA512 + self.n_top] = \
            beta[:, asset_id].T.astype(np.float32)

        # ---------------- 5. clock3 ---------------- #
        theta = 2.0 * np.pi * float(time_id % DAY) / DAY
        ht = self._harm * theta
        clock = np.empty(6, dtype=np.float32)
        clock[0::2] = np.sin(ht)
        clock[1::2] = np.cos(ht)
        out[:, COL_CLOCK:COL_CLOCK + 6] = clock

        return out
