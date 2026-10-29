# 训练代码

这里的代码只使用两个公开发布包，重新训练 `inference/model/` 中的全部文件。全部步骤跑完后，`<work>/model/` 下会得到
一套同名、同格式的文件，直接替换 `inference/model/` 即可，`inference/` 的代码不需要任何改动。

策略由四路加权融合：

```
pred = 0.30·CatBoost + 0.30·XGBoost + 0.15·D1 冻结头 + 0.25·D1 在线头
```

每一路都是 3 个种子的等权平均。D1 冻结头和在线头共用同一组训练好的网络；在线头在推理期每天用伪标签更新一次
MLP 头，伪标签由重构器给出。各部分的设计见 `report/`。

## 1. 使用的数据

| 发布包                      | 用途                                                                                                                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `public_release_20260630` | `data/train/`（第 0–616 天）：训练全部模型。D1 的标准化统计量、伪标签重构器和 `target_bound` 只用这一段。`data/test/`：只用于 `check` 步核对行对齐，以及 `verify.py` 截取公榜窗口做一致性对照，不参与训练。 |
| `public_release_20260823` | `data/train/`（第 617–767 天，即原公榜测试窗的回补标签）：与上一行拼接，共 768 天，训练全部模型。                                                                                                                   |

没有使用其他数据，也没有使用外部行情。

## 2. 数据读取

- 两个发布包的根目录通过命令行参数 `--data-0630`、`--data-0823` 传入。
- 读取代码在 `qtrc/common.py`。`label_files()` 依次拼接 `0630/data/train/train_partition_000..008.parquet`
  和 `0823/data/train/train_partition_000..002.parquet`，`scan_labels()` 用 polars 惰性读取需要的列。
  0630 训练集的 `time_id` 止于 888479，回补包从 888480 开始，拼接后即按 `time_id` 有序的 768 天；
  `qtrc/d1/cache.py` 读入时会检查这一点。
- 两个包的对齐：回补包的每一行对应 0630 测试集的一行，以 `row_id` 关联。`check` 步（`check_backfill_alignment()`）
  按 `row_id` 排序后逐行比对两边的 `row_id`、`time_id`、`asset_id`，要求完全一致。回补包本身带有特征和标签，
  训练时直接读取，不需要再与 0630 测试集连接。
- 用到的字段：

| 字段                           | 用途                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| `feature_000 … feature_322` | 全部模型的输入                                                                           |
| `time_id`、`asset_id`      | 时间网格定位、时序特征、日内时钟                                                         |
| `target`                     | 全部模型的训练目标                                                                       |
| `weight`                     | 训练样本权重，D1、CatBoost、XGBoost 都使用；`weight=0` 的行不参与计算 `target_bound` |
| `responder_02`               | 仅作 CatBoost 多任务标签的辅助维，推理时不需要                                           |
| `row_id`                     | 只用于对齐核对，不作特征                                                                 |

## 3. 运行

工作目录为 `train/`，配置文件为 `config.yaml`，入口为 `run.py`。下文用 `DATA0630`、`DATA0823`、`WORK` 分别表示
两个发布包的根目录和输出目录。

```bash
pip install -r requirements.txt          # torch 需从 cu124 源安装，见 requirements.txt 顶部注释
A="--data-0630 $DATA0630 --data-0823 $DATA0823 --work $WORK"

python run.py check          $A
python run.py d1-cache       $A
python run.py d1-rec         $A --gpu 0
python run.py d1-host        $A --gpu 0            # 可按种子分卡并行：--seeds 20261201 --gpu 0 / --seeds 20261202 --gpu 1 …
python run.py d1-export      $A
python run.py gbdt-features  $A
python run.py gbdt-cat       $A --gpu 0
python run.py gbdt-xgb       $A --gpu 0
python run.py ensemble       $A
```

| 步骤              | 内容                                              |
| ----------------- | ------------------------------------------------- |
| `check`         | 核对两个发布包的文件与行对齐                      |
| `d1-cache`      | D1 标准化统计量、`target_bound`、768 天稠密网格 |
| `d1-rec`        | 伪标签重构器（岭回归）                            |
| `d1-host`       | D1 网络，每个种子一份                             |
| `d1-export`     | 写出`d1_model.npz`、`d1_meta.json`            |
| `gbdt-features` | GBDT 特征矩阵与标签                               |
| `gbdt-cat`      | CatBoost 多任务模型，3 个种子                     |
| `gbdt-xgb`      | XGBoost 模型，3 个种子                            |
| `ensemble`      | 写出`ensemble.json`                             |

`d1-*` 与 `gbdt-*` 两条线互不依赖，可以同时运行；`ensemble` 需要等两条线都完成。

训练完成后，可以按入围提交说明 §4.2 自检。`verify.py` 截取一段公榜窗口（默认第 738–767 天），用发布包自带的官方
runner 让同一套 `inference/` 代码分别装载原权重和重训权重运行，报告两者预测的相关系数和 MAPE：

```bash
python verify.py --data-0630 $DATA0630 --data-0823 $DATA0823 --model $WORK/model --out $WORK/verify \
                 [--start-day 738 --days 30]
```

### 代码结构

```
run.py                    入口，按步骤分发
verify.py                 §4.2 预测对照
config.yaml               全部超参与手调系数
qtrc/common.py            配置、路径、发布包读取与对齐核对
qtrc/d1/cache.py          标准化统计量、target_bound、稠密网格 [T, 15, 325]
qtrc/d1/model.py          D1 网络：因果线性滤波前端 + 资产嵌入 + MLP 头
qtrc/d1/train.py          D1 训练：一天一个 batch，AdamW + cosine，SWA
qtrc/d1/reconstructor.py  伪标签重构器
qtrc/d1/export.py         导出 d1_model.npz / d1_meta.json
qtrc/gbdt/features.py     GBDT 特征矩阵，与推理包 src/stream_features.py 的在线构建逐格一致
qtrc/gbdt/train.py        CatBoost / XGBoost 训练，并导出为推理包使用的 numpy 格式
qtrc/ensemble.py          写出 ensemble.json
```

## 4. 随机种子

| 随机性来源                     | 固定方式                                                             | 配置项                                             |
| ------------------------------ | -------------------------------------------------------------------- | -------------------------------------------------- |
| D1 参数初始化、dropout         | `torch.manual_seed`（CPU 与 CUDA），同时固定 `random`、`numpy` | `d1.train.seeds`：20261201/02/03，各对应一个成员 |
| D1 每步取哪一天                | `np.random.default_rng`，与上一行同一个种子                        | 同上                                               |
| CatBoost 量化边界的行抽样      | `Pool.quantize(random_seed=42)`                                    | `gbdt.catboost.quantize.random_seed`             |
| CatBoost 行子采样、随机扰动    | `random_seed`                                                      | `gbdt.seeds`：42/43/44，各对应一个成员           |
| XGBoost 行、列子采样           | `seed`                                                             | `gbdt.seeds`：42/43/44                           |
| 重构器、标准化统计量、特征矩阵 | 无随机性                                                             | —                                                 |

## 5. 产物与 inference/ 的对应

| 产物（`<work>/model/`）  | 对应`inference/` 中的路径      | 生成步骤                                                                                                                     |
| -------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `d1_model.npz`           | `model/d1_model.npz`           | `d1-export`：汇总标准化统计量与截断值（`d1-cache`）、重构器系数（`d1-rec`）、网络权重（`d1-host`），时钟表由代码生成 |
| `d1_meta.json`           | `model/d1_meta.json`           | `d1-export`：网络结构常数、在线更新超参、`target_bound`                                                                  |
| `cat_seed{42,43,44}.npz` | `model/cat_seed{42,43,44}.npz` | `gbdt-cat`                                                                                                                 |
| `xgb_seed{42,43,44}.npz` | `model/xgb_seed{42,43,44}.npz` | `gbdt-xgb`                                                                                                                 |
| `ensemble.json`          | `model/ensemble.json`          | `ensemble`：融合权重、XGBoost 输入列、`target_bound`                                                                     |

中间表位于 `<work>/d1/` 和 `<work>/gbdt/`，都由上面的步骤从发布包原始文件生成，不需要另外准备。
`<work>/d1/hosts/seed*.pt` 是 D1 的 torch 检查点，`<work>/gbdt/models/` 是 CatBoost / XGBoost 的原生模型文件，
二者不直接进入推理包。

## 6. 手调系数的来源

所有手调系数都写在 `config.yaml` 中，选定它们所用的样本都在两个发布包之内：

- **在 5 折扩展窗口验证上选定**：融合权重 `ensemble.weights`（0.30/0.30/0.15/0.25）、D1 在线更新超参 `d1.online.*`、
  CatBoost 辅助标签权重 λ=8、GBDT 轮数（CatBoost 500，XGBoost 320）、XGBoost 的树形参数、派生特征块的取舍。
  5 个验证窗各 30 天，依次为第 618–647、648–677、678–707、708–737、738–767 天，全部落在 0823 回补包内；
  每折只用验证窗开始前 60 天以前的数据训练。验证方式详见 `report/md/valsys.md`。
- **在 0630 训练集内部的时间切分上选定**：D1 的结构与训练超参（L=16、k=5、隐层 128、dropout 0.55、学习率、
  24000 步、SWA）、CatBoost 的树参数（深度 8、学习率 0.03、l2 100 等）、`top40` 基特征列表、重构器的 λ=10 与
  滞后 0–6。其中 D1 部分之后又在上述 5 折上复核过。
- **由数据计算、不是手调的**：`target_bound`（0630 训练集中 `weight>0` 的行上 |target| 的最大值，保留 4 位小数）、
  标准化统计量、重构器系数。这些由代码计算，不写在配置里。

## 7. 运行时间、磁盘与硬件

在单张 RTX 4090 的机器上实测，各步骤的大致用时如下：

| 步骤                                   | 用时                         |
| -------------------------------------- | ---------------------------- |
| `check`、`d1-export`、`ensemble` | 均不到 1 min                 |
| `d1-cache`                           | 约 15 min                    |
| `d1-rec`                             | 约 2 min                     |
| `d1-host`                            | 约 6 min（每个种子约 2 min） |
| `gbdt-features`                      | 约 20 min                    |
| `gbdt-cat`                           | 约 8 min                     |
| `gbdt-xgb`                           | 约 6 min                     |
| `verify.py`（可选）                  | 每个窗口约 10 min            |

`d1-*` 与 `gbdt-features` 可以同时运行，全部训练步骤合计约 40 min。

- 磁盘：`<work>` 约 50 GB，最终产物约 17 MB。
- 内存：峰值约 90 GB，出现在 `gbdt-features`。建议内存不少于 128 GB。
- GPU：显存 24 GB 的 NVIDIA GPU 即可。D1 训练使用 bf16 混合精度，不支持 bf16 的 GPU 会自动改用 fp16。

## 8. 复现结果

以下结果来自在 RTX 4090 上按第 3 节的命令从零完整运行一遍。

### 产物比对

| 产物                                              | 与`inference/model/` 中原文件的比较                                                              |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `cat_seed{42,43,44}.npz`                        | 3 个文件全部与原文件完全相同                                                                       |
| D1 标准化统计量、截断值、时钟表、`target_bound` | 逐位相同                                                                                           |
| D1 重构器系数                                     | 差异在 1e-8 量级                                                                                   |
| `d1_meta.json`                                  | 除生成日期外相同                                                                                   |
| `ensemble.json`                                 | 融合权重、输入列、`target_bound` 相同；只有不参与推理的记录信息（如 XGBoost 权重文件的 md5）不同 |
| D1 网络权重、`xgb_seed*.npz`                    | 不完全相同，原因见下                                                                               |

- **D1 网络**：原权重在 A100 上训练。同一块 GPU 上重复训练，结果逐位相同；但不同型号的 GPU 选用的底层计算方式不同，
  末位舍入有差异，经过 24000 步训练后被放大，因此在 4090 上重训的权重与原权重不完全相同。在 A100 上重训预期可以逐位复现，
  由于手头没有 A100，这一点未能验证。
- **XGBoost**：GPU 上的 `hist` 训练本身不能逐位复现，同一台机器、同一种子连续训练两次，得到的树就已经不同。

### §4.2 一致性自检

用 `verify.py` 在公榜测试窗的前 30 天和后 30 天各跑一次。每个窗口都从第一天开始连续运行，D1 在线更新的状态也从窗口第一天开始累积。

| 窗口           |    行数 | corr(ŷ, ŷ') | MAPE(ŷ, ŷ') | max\|ŷ−ŷ'\| | 原权重 R² | 重训权重 R² |
| -------------- | ------: | ------------: | ------------: | -------------: | ---------: | -----------: |
| 第 617–646 天 | 647,925 |        0.9972 |         0.707 |          0.048 |   0.015820 |     0.015772 |
| 第 738–767 天 | 626,143 |        0.9974 |         0.823 |          0.055 |   0.015333 |     0.015365 |

R² 用 0823 回补标签按加权零均值 R² 计算。这两段数据都在训练集内，R² 只说明重训权重与原权重表现相当，不代表样本外成绩。

按四路拆开（第 738–767 天）：

| 分支                   |   corr | MAPE | max\|Δ\| |
| ---------------------- | -----: | ---: | --------: |
| CatBoost（3 种子平均） | 1.0000 |    0 |         0 |
| XGBoost（3 种子平均）  | 0.9911 | 2.38 |     0.049 |
| D1 冻结                | 0.9965 | 1.37 |     0.042 |
| D1 在线                | 0.9929 | 1.61 |     0.198 |
| 最终预测               | 0.9974 | 0.82 |     0.055 |

两个窗口的相关系数都满足 ≥ 0.99。MAPE 不满足 ≤ 0.001，差异全部来自 XGBoost 和 D1 两路，CatBoost 一路的预测与原权重完全相同。
MAPE 按逐点相对误差计算，而最终预测的标准差只有约 0.056，大量预测值接近 0，在这些点上 0.01 量级的绝对差就会变成数倍的
相对误差。只要 GPU 训练不能逐位复现，MAPE 就会落在这个量级。
