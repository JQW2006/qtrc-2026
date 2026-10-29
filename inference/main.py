"""d1-gbdt-ens-0827：D1 在线自适应 × CatBoost × XGBoost 的四路线性融合。

    pred = 0.30·cat3 + 0.30·xgb3 + 0.15·d1_host3 + 0.25·d1_online3

每一路都是 3 个种子的等权平均。权重来自 exp042 在 valsys_v2 五折上的终裁
（五折均值 49.77e-4、最差折 36.03e-4，相对现役 d1-online-0825 是 +4.91e-4、
配对 t=5.19、5/5 折全胜；121 个滑动 30 天窗里 98% 为正）。权重是**固定常数**，
不在推理期做任何拟合。

推理协议
--------
评测系统导入本文件的 `class Model`，初始化一次，然后按递增 `time_id` 逐步调用
`predict(test)`；`test` 只含 `row_id,time_id,asset_id,feature_*`。
**零第三方依赖**：只用 numpy——两个 GBDT 都已导出成数组，由 `src/gbdt_numpy.py`
做纯 numpy 推理（对拍见 exp042 `parity_gbdt.py`）。所有路径用
`Path(__file__).resolve().parent` 定位，不含任何绝对路径。

三块的分工
----------
- `src/d1_online.py`：exp039 已在官方 runner 上验收过的 D1 引擎，本包只把它的
  `predict` 从"返回 blend 后的一路"改成"返回 host / online 两路"，好让融合权重
  分别作用。blend 因此被融合权重吸收，不再是一个独立参数。
- `src/stream_features.py`：GBDT 那 490 列的在线构建。它有两条**互不相同**的
  时间轴（该资产的观测序列 vs 稠密 time_id 网格），细节写在该文件的模块说明里。
- `src/gbdt_numpy.py`：遗忘树 / lossguide 树的向量化推理。3 个种子摞成一个树库
  走一遍遍历，比逐个模型调用省两倍多的 Python 开销。

两处协议细节（与现役包一致）
----------------------------
- 私榜按 `time_id` 顺序调用时，部分行只用于恢复内部历史、不计入分数，其 `row_id`
  可能为负；本实现不看 `row_id`，历史照常更新。
- `time_id` 之间可能有间隙（公开 test 里就有两处 1445 帧的空档）。D1 侧按"全体
  缺席"补帧；GBDT 侧的稠密网格状态用一阶 IIR 的闭式解一次跳过，与逐格推进
  逐位相同（`parity_stream.py --gap` 验证）。
"""
import os

# ---------------------------------------------------------------- 线程限制 --
# 赛题 Note 要求"务必在策略模型中手动设置最大线程数"（额度 8 核，本地测耗时 <= 4）。
# **环境变量在这里通常已经太晚**：官方 runner 在 `runner.py` 顶层就 `import numpy`，
# 之后才载入本文件，而 OpenBLAS 是在被载入那一刻读走 `OMP_NUM_THREADS` 的。
# 所以下面这轮 setdefault 只在"本文件先于 numpy 被载入"的场景里有效（例如直接
# `import main`），真正起作用的是 `_limit_threads()` 里对**已载入**的 BLAS 做运行时限流。
# 用 setdefault 而不是覆盖：如果评测方自己设了值，尊重它，不去打架。
MAX_THREADS = 4
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(MAX_THREADS))

import json
from pathlib import Path

import numpy as np

from src.d1_online import D1Online
from src.gbdt_numpy import load as load_trees
from src.stream_features import FeatureStream

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"


def _limit_threads(n: int):
    """对已经载入的 BLAS 做运行时限流。返回 (说明, 需要保活的句柄)。

    两条路，都不是硬依赖——任何一条失败都不影响预测正确性，只影响线程数：

    1. `threadpoolctl.threadpool_limits(n)`：标准做法，直接改运行中的 OpenBLAS/MKL。
       返回的对象必须**保活**，被回收时它会把限制还原回去。
    2. ctypes 兜底：万一容器里没有 threadpoolctl，就直接对 BLAS 动态库调
       `openblas_set_num_threads`。**从 `/proc/self/maps` 找已经载入的那个 so**，
       而不是去 numpy 目录里猜路径——numpy 的 BLAS 装在同级的 `numpy.libs/`
       而不是 `numpy/` 下面，按目录猜会静默落空（我第一版就是这么写的，
       兜底等于没有）。numpy 打包的是 64 位整型接口，符号名带 `64_` 后缀，两个都试。

    本函数**不打印任何东西**：runner 的 JSON 报告走 stdout，往 stdout 写字节
    会把它污染成无法解析。
    """
    notes, keep = [], None
    try:
        import threadpoolctl
        keep = threadpoolctl.threadpool_limits(limits=n)
        notes.append(f"threadpoolctl={n}")
    except Exception as exc:                     # noqa: BLE001
        notes.append(f"threadpoolctl 不可用({type(exc).__name__})")
    try:
        import ctypes
        libs = set()
        try:
            with open("/proc/self/maps", "r") as fh:
                for line in fh:
                    q = line.rstrip().split(" ")[-1]
                    if q.startswith("/") and q.endswith(".so") or ".so." in q:
                        low = q.lower()
                        if "openblas" in low or "libmkl_rt" in low:
                            libs.add(q)
        except OSError:
            pass
        # 符号名随 wheel 而变：numpy 新版打的是 scipy-openblas，符号统一加了 `scipy_`
        # 前缀（`scipy_openblas_set_num_threads64_`）；老版 numpy 打的 `libopenblas64_`
        # 用的是不带前缀的名字。**必须逐个试**，硬编码一两个就会静默落空。
        cands = ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads_64_",
                 "scipy_openblas_set_num_threads", "openblas_set_num_threads64_",
                 "openblas_set_num_threads_64_", "openblas_set_num_threads",
                 "goto_set_num_threads", "MKL_Set_Num_Threads")
        for path in sorted(libs):
            lib = ctypes.CDLL(path)
            for sym in cands:
                fn = getattr(lib, sym, None)
                if fn is not None:
                    fn.argtypes = [ctypes.c_int]
                    fn(n)
                    notes.append(f"{sym}={n}")
                    break
            else:
                # 这条必须记：第一版在"找到库但没匹配到符号"时什么都不记，
                # 是一次**静默失败**——看起来一切正常，其实兜底根本没生效。
                notes.append(f"{Path(path).name} 无已知的 set_num_threads 符号")
        if not libs:
            notes.append("未在 /proc/self/maps 里找到 BLAS")
    except Exception as exc:                     # noqa: BLE001
        notes.append(f"ctypes 兜底失败({type(exc).__name__})")
    return "; ".join(notes), keep


class Model:
    def __init__(self):
        cfg = json.loads((MODEL_DIR / "ensemble.json").read_text())
        self.max_threads = int(cfg.get("max_threads", MAX_THREADS))
        # 句柄挂在实例上保活：threadpoolctl 的限制在对象被回收时会还原
        self.thread_note, self._thread_guard = _limit_threads(self.max_threads)
        self.weights = cfg["weights"]
        self.seeds = cfg["gbdt_seeds"]
        self.keep = np.asarray(cfg["xgb_keep_columns"], dtype=np.int64)
        self.n_group_cols = int(cfg["n_group_columns"])
        self.target_bound = float(cfg["target_bound"])
        self.raw_features = cfg["raw_features"]

        self.d1 = D1Online(MODEL_DIR)
        self.stream = FeatureStream()
        self.cat = load_trees([dict(np.load(MODEL_DIR / f"cat_seed{s}.npz"))
                               for s in self.seeds])
        self.xgb = load_trees([dict(np.load(MODEL_DIR / f"xgb_seed{s}.npz"))
                               for s in self.seeds])
        if self.cat.n_features != self.n_group_cols:
            raise ValueError(f"cat 输入宽度 {self.cat.n_features} != {self.n_group_cols}")
        if self.xgb.n_features != len(self.keep):
            raise ValueError(f"xgb 输入宽度 {self.xgb.n_features} != {len(self.keep)}")

        # 预热：让 BLAS/numpy 的首次分配发生在 __init__ 里而不是第一步预测里
        warm = np.zeros((1, self.n_group_cols), np.float32)
        self.cat.predict(warm)
        self.xgb.predict(warm[:, self.keep])

    def predict(self, test):
        time_col = test["time_id"]
        time_id = int(time_col.iloc[0]) if hasattr(time_col, "iloc") \
            else int(np.asarray(time_col)[0])
        asset = test["asset_id"].to_numpy().astype(np.int64)

        host, online = self.d1.predict(test)

        raw = test[self.raw_features].to_numpy(dtype=np.float32)
        X = self.stream.step(time_id, asset, raw)
        cat = self.cat.predict(X).mean(axis=0)
        xgb = self.xgb.predict(X[:, self.keep]).mean(axis=0)

        w = self.weights
        out = (w["cat"] * cat + w["xgb"] * xgb
               + w["host"] * host + w["online"] * online)
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -self.target_bound, self.target_bound)
