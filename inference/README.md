# d1-gbdt-ens-mtl-0830

`d1-gbdt-ens-0827` 的**单点替换**版：融合权重、xgb、D1 三路一字不动，
只把 CatBoost 那一路换成多任务（MultiRMSE）重训的版本。

    pred = 0.30·cat3 + 0.30·xgb3 + 0.15·d1_host3 + 0.25·d1_online3

## 改了什么

`cat3` 的训练标签从单列 `target` 换成两列：

    label[:, 0] = target
    label[:, 1] = sqrt(8) · std(target) · clip(z(responder_02), ±5)

损失是 CatBoost 的 `MultiRMSE`。它的叶值**按维独立**做加权最小二乘，所以辅助维
只参与**选分裂**、从不进入目标维的叶值——`responder_02`（训练集才有的短期限
responder，h≈3）告诉树"往哪儿分"，`target` 保留全部叶值控制权。
特征一列没动，轮数、超参、量化参数、训练跨度（768 天）、3 个种子全部与 0827 相同；
`responder_*` 只作训练标签，推理时一个字都用不到，官方 API 不返回它们也无所谓。

**推理侧零增量**：导出后的数组格式与单输出模型完全一致（只保留第 0 维），
`src/gbdt_numpy.py` 一个字未改，模型文件大小、单步耗时、常驻内存都不变。

## 依据

valsys_v2 五折（扩展训练 + 60 天 blackout + 30 天验证窗），换成员、权重不动：

| | F1 | F2 | F3 | F4 | F5 | 均值 | 最差折 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0827 | 45.83 | 36.03 | 48.77 | 52.97 | 65.25 | 49.77 | 36.03 |
| 本包 | 46.12 | 37.00 | 49.50 | 53.29 | 65.93 | **50.37** | **37.00** |

+0.60e-4，5/5 折全正；管线对照（同一条重训管线、单输出标签）是 −0.01，干净归零。
单模层面 CatBoost 三种子五折 15 格配对 +2.76e-4、15/15、t=6.11。
权重面在原点附近是平的（任何重调 ≤ +0.07e-4），所以**权重不重调**，
这也是唯一零拟合自由度的方案。详见 `experiments/GBDT/exp010_responder_mtl/RESULTS.md`
与 `experiments/AE_MLP/exp043_mtl_deliver/NOTES.md`。

## 与 0827 的文件级关系

除 `model/cat_seed{42,43,44}.npz`、`model/ensemble.json`、`README.md` 之外，
**每个文件都与 `strategies/d1_gbdt_ens_0827/` 逐字节相同**（`package43.py` 比 md5）。
线程限制、路径定位、零第三方依赖等性质因此原样继承，不再重复论证。

## 目录

    main.py                 组合器（Model 类）
    requirements.txt
    src/d1_online.py        D1 引擎
    src/stream_features.py  GBDT 490 列的在线构建
    src/gbdt_numpy.py       遗忘树 / lossguide 树的纯 numpy 推理
    model/ensemble.json     融合权重与列表
    model/d1_model.npz      D1 权重
    model/d1_meta.json
    model/cat_seed{42,43,44}.npz   ← 本版唯一替换的东西
    model/xgb_seed{42,43,44}.npz
