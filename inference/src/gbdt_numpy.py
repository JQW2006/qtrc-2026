"""把 CatBoost / XGBoost 模型导出成纯 numpy 可推理的数组，并提供推理器。

**为什么要这一层**：交付 ZIP 要在官方 JupyterHub 的 base 环境里被导入。现役
d1-online 交付包是零第三方依赖的纯 numpy，加进 catboost/xgboost 就多了两个必须
预装、且模型文件要跨版本可读的依赖。导出成数组以后这两条风险都消失，而且 15 行
输入下比原生 `predict` 更快——原生库的每次调用开销（建 Pool / DMatrix）在这个
规模上是主要成本。

**正确性由对拍保证，不由推理**：`parity_gbdt.py` 拿真实特征矩阵逐行比对
原生 predict 与本模块，阈值按 float32 舍入定。任何一处树结构理解错了，
对拍就会给出**大**的差（叶子值差一个量级），不会是 1e-7 的小差。

---------------------------------------------------------------- CatBoost ----
遗忘树（oblivious）：一棵树的第 k 层对**所有**节点用同一个 (特征, 边界)。
所以一棵深 d 的树只要 d 次比较就得到叶子下标：

    leaf = Σ_k  1[x[f_k] > border_k] · 2^k

`nan_mode=Min` 把 NaN 当作比任何边界都小 ⇒ 位为 0。numpy 里 `x > border`
遇到 NaN 天然给 False，语义正好吻合，不需要额外处理（对拍会验证这一点）。

导出后所有树的 (特征, 边界, 叶值) 摞成 [n_trees, depth] / [n_trees, 2^depth]
两个矩阵，一次向量化就算完全部树，没有 Python 层循环。

----------------------------------------------------------------- XGBoost ----
lossguide 的树不是遗忘树，必须逐节点走。做法是**按层同步遍历所有树**：
维护 [n_rows, n_trees] 的当前节点下标，每轮 gather 一次分裂特征与阈值，
比较后跳到左/右孩子，叶子节点用自环固定住。层数 = 全部树的最大深度（这里 ~15-20），
所以是十几轮 [15, 320] 的 gather，而不是 320×深度 次 Python 迭代。

xgb 的判定是 `x < threshold → 左`（注意是严格小于，与 cat 的 `>` 不对称），
缺失值走 `default_left`。这两条都由对拍验证。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# CatBoost
# --------------------------------------------------------------------------- #
def export_catboost(model, n_features: int) -> dict[str, np.ndarray]:
    """model: 已 fit 的 CatBoostRegressor。返回可直接 np.savez 的数组字典。"""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as fh:
        model.save_model(fh.name, format="json")
        blob = json.loads(Path(fh.name).read_text())

    # float_feature_index（模型内部序）→ flat_feature_index（输入矩阵列号）
    flat_of_float = {}
    for info in blob["features_info"].get("float_features", []):
        flat_of_float[int(info["feature_index"])] = int(info["flat_feature_index"])

    trees = blob["oblivious_trees"]
    depths = {len(t["splits"]) for t in trees}
    if len(depths) != 1:
        raise ValueError(f"树深不一致：{sorted(depths)}；本导出器只支持等深遗忘树")
    depth = depths.pop()

    feat = np.empty((len(trees), depth), dtype=np.int32)
    border = np.empty((len(trees), depth), dtype=np.float32)
    leaves = np.empty((len(trees), 1 << depth), dtype=np.float64)
    for i, tree in enumerate(trees):
        for k, split in enumerate(tree["splits"]):
            if split.get("split_type") != "FloatFeature":
                raise ValueError(f"不支持的分裂类型 {split.get('split_type')}")
            feat[i, k] = flat_of_float[int(split["float_feature_index"])]
            border[i, k] = np.float32(split["border"])
        values = np.asarray(tree["leaf_values"], dtype=np.float64)
        if values.shape != (1 << depth,):
            raise ValueError(f"叶子数 {values.shape} != {1 << depth}")
        leaves[i] = values

    sab = blob.get("scale_and_bias", [1.0, [0.0]])
    scale = float(sab[0])
    bias = float(sab[1][0] if isinstance(sab[1], (list, tuple)) else sab[1])
    if feat.max() >= n_features:
        raise ValueError(f"引用了列 {feat.max()}，超出输入宽度 {n_features}")
    return {"kind": np.array("catboost"), "feat": feat, "border": border,
            "leaves": leaves, "scale": np.float64(scale), "bias": np.float64(bias),
            "n_features": np.int64(n_features)}


class CatBoostNumpy:
    """遗忘树的向量化推理，可一次装多个模型（多种子摞成一个树库走一遍）。"""

    def __init__(self, blobs):
        if isinstance(blobs, dict) or not isinstance(blobs, (list, tuple)):
            blobs = [blobs]
        self.n_models = len(blobs)
        self.n_features = int(blobs[0]["n_features"])
        depths = {int(np.asarray(b["feat"]).shape[1]) for b in blobs}
        if len(depths) != 1:
            raise ValueError(f"树库内树深不一致：{sorted(depths)}")
        self.depth = depths.pop()
        self.scale = np.array([float(b["scale"]) for b in blobs], np.float64)
        self.bias = np.array([float(b["bias"]) for b in blobs], np.float64)

        feats, borders, leaves, self.model_slice, start = [], [], [], [], 0
        for b in blobs:
            if int(b["n_features"]) != self.n_features:
                raise ValueError("树库内各模型输入宽度不一致")
            f = np.asarray(b["feat"], dtype=np.int32)
            feats.append(f)
            borders.append(np.asarray(b["border"], dtype=np.float32))
            leaves.append(np.asarray(b["leaves"], dtype=np.float64))
            self.model_slice.append((start, start + f.shape[0]))
            start += f.shape[0]
        self.n_trees = start
        self._flat_feat = np.concatenate([f.reshape(-1) for f in feats])
        self._flat_border = np.concatenate([b.reshape(-1) for b in borders])
        self._flat_leaves = np.concatenate([v.reshape(-1) for v in leaves])
        self._bits = (1 << np.arange(self.depth, dtype=np.int32))[None, None, :]
        self._tree_offset = (
            np.arange(self.n_trees, dtype=np.int32) << self.depth)[None, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 [n_models, n_rows]。"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"输入形状 {X.shape}，期望 (n, {self.n_features})")
        # NaN 在 `>` 下天然为 False，与 CatBoost 的 nan_mode=Min 语义一致。
        bits = (X[:, self._flat_feat] > self._flat_border).reshape(
            X.shape[0], self.n_trees, self.depth)
        leaf = (bits * self._bits).sum(axis=2)
        vals = self._flat_leaves[leaf + self._tree_offset]
        out = np.empty((self.n_models, X.shape[0]), dtype=np.float64)
        for m, (lo, hi) in enumerate(self.model_slice):
            out[m] = vals[:, lo:hi].sum(axis=1) * self.scale[m] + self.bias[m]
        return out


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def export_xgboost(booster, n_features: int) -> dict[str, np.ndarray]:
    blob = json.loads(booster.save_raw("json").decode("utf-8"))
    learner = blob["learner"]
    gbm = learner["gradient_booster"]["model"]
    trees = gbm["trees"]
    param = learner["learner_model_param"]
    # xgboost 3.x 把 base_score 写成 "[8.34e-5]" 这样的字符串（内部是向量）。
    base_score = float(str(param["base_score"]).strip("[]").split(",")[0])
    if int(param.get("num_target", 1)) != 1:
        raise ValueError("只支持单目标回归")
    tree_info = gbm.get("tree_info", [0] * len(trees))
    if any(t != 0 for t in tree_info):
        raise ValueError("只支持单组树（num_parallel_tree=1, 非多分类）")

    max_nodes = max(len(t["left_children"]) for t in trees)
    n_trees = len(trees)
    left = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    right = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    sfeat = np.zeros((n_trees, max_nodes), dtype=np.int32)
    scond = np.zeros((n_trees, max_nodes), dtype=np.float32)
    dleft = np.zeros((n_trees, max_nodes), dtype=bool)
    value = np.zeros((n_trees, max_nodes), dtype=np.float64)
    depth = 0
    for i, tree in enumerate(trees):
        lc = np.asarray(tree["left_children"], dtype=np.int32)
        rc = np.asarray(tree["right_children"], dtype=np.int32)
        n = len(lc)
        left[i, :n], right[i, :n] = lc, rc
        sfeat[i, :n] = np.asarray(tree["split_indices"], dtype=np.int32)
        # split_conditions 在内节点是阈值、在叶子节点就是叶子值——xgb 的原生布局。
        scond[i, :n] = np.asarray(tree["split_conditions"], dtype=np.float32)
        value[i, :n] = np.asarray(tree["split_conditions"], dtype=np.float64)
        dleft[i, :n] = np.asarray(tree["default_left"], dtype=np.int32) != 0
        # 叶子自环：遍历到叶子后停在原地，多余的层数不会改变结果
        leaf = lc < 0
        left[i, :n] = np.where(leaf, np.arange(n), lc)
        right[i, :n] = np.where(leaf, np.arange(n), rc)
        depth = max(depth, _tree_depth(lc, rc))
        # 补齐区当叶子（值 0）处理，永远走不到
        if n < max_nodes:
            left[i, n:] = np.arange(n, max_nodes)
            right[i, n:] = np.arange(n, max_nodes)
    is_leaf = np.zeros((n_trees, max_nodes), dtype=bool)
    for i, tree in enumerate(trees):
        lc = np.asarray(tree["left_children"], dtype=np.int32)
        is_leaf[i, :len(lc)] = lc < 0
        is_leaf[i, len(lc):] = True
    if sfeat[~is_leaf].size and int(sfeat[~is_leaf].max()) >= n_features:
        raise ValueError(f"引用了列 {sfeat[~is_leaf].max()}，超出输入宽度 {n_features}")
    return {"kind": np.array("xgboost"), "left": left, "right": right,
            "sfeat": sfeat, "scond": scond, "dleft": dleft, "value": value,
            "is_leaf": is_leaf, "max_depth": np.int64(depth),
            "base_score": np.float64(base_score),
            "n_features": np.int64(n_features)}


def _tree_depth(lc: np.ndarray, rc: np.ndarray) -> int:
    depth, frontier = 0, [0]
    while frontier:
        nxt = []
        for node in frontier:
            if lc[node] >= 0:
                nxt += [int(lc[node]), int(rc[node])]
        if nxt:
            depth += 1
        frontier = nxt
    return depth


class XGBoostNumpy:
    """按层同步遍历的向量化推理，可一次装多个模型（多种子共用一次遍历）。

    为什么要为速度费这个事：`max_leaves=512` 的 lossguide 树**深达 57 层**，
    朴素写法就是 57 轮 × 每轮七八次 numpy 调用。15 行输入下每次 numpy 调用的
    固定开销（几微秒）才是主要成本，元素数根本不重要。所以三处都在砍调用次数：

    1. **多模型合并成一个树库**：3 个种子 = 960 棵树一起走，深度只付一遍。
       元素数从 4800 涨到 14400，但每层的调用次数不变——近乎白拿两倍。
    2. **左右孩子合并**成 `child[2*node + go_right]`，两次 gather 变一次。
    3. **活跃集压缩**：57 层里绝大多数状态在前 20 层就到叶子了。一旦活跃比例
       跌破阈值就只对活跃下标做 gather，后面的层几乎不花钱。

    缺失值语义与 xgb 一致：非有限值走 `default_left`，其余按 `x < threshold` 向左。
    """

    def __init__(self, blobs):
        if isinstance(blobs, dict) or not isinstance(blobs, (list, tuple)):
            blobs = [blobs]
        self.n_models = len(blobs)
        self.n_features = int(blobs[0]["n_features"])
        self.base_score = np.array([float(b["base_score"]) for b in blobs], np.float64)
        max_nodes = max(int(np.asarray(b["left"]).shape[1]) for b in blobs)
        self.max_nodes = max_nodes
        self.max_depth = max(int(b["max_depth"]) for b in blobs)

        banks, self.model_slice, start = [], [], 0
        for b in blobs:
            if int(b["n_features"]) != self.n_features:
                raise ValueError("树库内各模型输入宽度不一致")
            n_trees = int(np.asarray(b["left"]).shape[0])
            banks.append(b)
            self.model_slice.append((start, start + n_trees))
            start += n_trees
        self.n_trees = start

        def pad(name, dtype, fill):
            out = np.full((self.n_trees, max_nodes), fill, dtype=dtype)
            for b, (lo, hi) in zip(banks, self.model_slice):
                arr = np.asarray(b[name])
                out[lo:hi, :arr.shape[1]] = arr
                if arr.shape[1] < max_nodes:      # 补齐区自环、当叶子、值为 0
                    tail = np.arange(arr.shape[1], max_nodes)
                    if name in ("left", "right"):
                        out[lo:hi, arr.shape[1]:] = tail[None, :]
            return out

        left = pad("left", np.int32, 0)
        right = pad("right", np.int32, 0)
        self.sfeat = pad("sfeat", np.int32, 0).reshape(-1)
        self.scond = pad("scond", np.float32, 0.0).reshape(-1)
        self.dleft = pad("dleft", bool, False).reshape(-1)
        self.value = pad("value", np.float64, 0.0).reshape(-1)
        self.is_leaf = pad("is_leaf", bool, True).reshape(-1)
        # child[2*flat + go_right]：把左右孩子交错存成一个数组，每层少一次 gather
        self.child = np.empty(self.n_trees * max_nodes * 2, dtype=np.int32)
        self.child[0::2] = left.reshape(-1)
        self.child[1::2] = right.reshape(-1)

        self._off = (np.arange(self.n_trees, dtype=np.int32) * max_nodes)[None, :]
        self._model_id = np.repeat(
            np.arange(self.n_models), [hi - lo for lo, hi in self.model_slice])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 [n_models, n_rows]；单模型时也是二维，调用方自己取 [0]。"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"输入形状 {X.shape}，期望 (n, {self.n_features})")
        n, nf = X.shape
        Xf = np.ascontiguousarray(X).reshape(-1)
        row_off = (np.arange(n, dtype=np.int32) * nf)[:, None]

        node = np.zeros((n, self.n_trees), dtype=np.int32)
        flat = node + self._off                      # [n, n_trees]
        active = np.flatnonzero(~self.is_leaf[flat.reshape(-1)]).astype(np.int32)
        row_off_flat = np.repeat(row_off.reshape(-1), self.n_trees)
        node_flat = node.reshape(-1)
        off_flat = np.broadcast_to(self._off, (n, self.n_trees)).reshape(-1)

        for _ in range(self.max_depth):
            if active.size == 0:
                break
            f = node_flat[active] + off_flat[active]
            val = Xf[row_off_flat[active] + self.sfeat[f]]
            go_right = ~np.where(np.isnan(val), self.dleft[f], val < self.scond[f])
            nxt = self.child[2 * f + go_right]
            node_flat[active] = nxt
            still = ~self.is_leaf[nxt + off_flat[active]]
            if not still.all():
                active = active[still]
        if active.size:
            # 走满 max_depth 还没到叶子 ⇒ 树里有环，或者导出时深度算错了。
            # 不拦的话下面会把内节点的 split_condition 当叶子值读出来——数值合法、
            # 结果全错，是最难查的那种失败。
            raise RuntimeError(f"{active.size} 个状态在 {self.max_depth} 层后仍未到叶子")

        vals = self.value[node_flat + off_flat].reshape(n, self.n_trees)
        out = np.empty((self.n_models, n), dtype=np.float64)
        for m, (lo, hi) in enumerate(self.model_slice):
            out[m] = vals[:, lo:hi].sum(axis=1) + self.base_score[m]
        return out


# --------------------------------------------------------------------------- #
def load(blobs) -> CatBoostNumpy | XGBoostNumpy:
    """blobs 是一个导出字典，或同类模型的列表（列表 = 合成一个树库）。"""
    first = blobs if isinstance(blobs, dict) or not isinstance(blobs, (list, tuple)) \
        else blobs[0]
    kind = str(first["kind"])
    if kind == "catboost":
        return CatBoostNumpy(blobs)
    if kind == "xgboost":
        return XGBoostNumpy(blobs)
    raise ValueError(f"未知 kind {kind}")
