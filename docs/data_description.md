# 比赛文档\_数据说明

## 数据概述（Description）

本次比赛数据为匿名化多标的时序数据。每一行对应一个匿名标的在一个匿名时点的状态。数据通过 `time_id` 表示时间顺序，通过 `asset_id` 区分不同匿名标的。

训练数据包含索引字段、匿名特征、样本权重、辅助 responder 和最终目标 `target`。测试数据仅包含预测时可见的信息，不包含 responder 和 `target`。

数据采用列式存储格式，以支持高效读取和大规模建模。

### 数据规模与比赛阶段 \(Data Scales and Competition Phases\)

本次比赛数据覆盖较长的连续历史区间，包含十余个匿名标的、300 余个匿名特征和数十个辅助 responder。训练数据整体为千万行至数千万行量级，并可能按连续 `time_id` 切分为多个 Parquet 分区文件。

| 数据集         | 字段内容                                                                                      | 说明                                 |
| -------------- | --------------------------------------------------------------------------------------------- | ------------------------------------ |
| 训练数据       | `row_id`、`time_id`、`asset_id`、`weight`、`feature_*`、`responder_*`、`target` | 用于模型训练、验证与调参             |
| 公榜测试数据   | `row_id`、`time_id`、`asset_id`、`feature_*`                                          | 用于公开排行榜评分，规模小于训练数据 |
| 私有评测数据   | 顺序释放的 `feature_*` 与索引字段                                                           | 用于最终排名，不会一次性完整暴露     |
| 示例与接口数据 | 示例提交、示例代码、Time\-Series API 本地验证工具                                             | 用于本地调试与提交格式检查           |

---

## 文件说明（Files）

### `train.parquet` / `train_partition_*.parquet`

训练数据文件。训练集可能被切分为多个连续分区。每个分区包含一段连续的 `time_id`，但分区边界不对应真实世界中的任何公开时间含义。

主要字段包括：

```Plain
row_id
time_id
asset_id
weight
feature_000
feature_001
...
feature_xxx
responder_00
responder_01
...
responder_xx
target
```

---

### `test.parquet` / `test_partition_*.parquet`

测试阶段使用的数据文件。该文件仅包含模型预测时可见的信息。

主要字段包括：

```Plain
row_id
time_id
asset_id
feature_000
feature_001
...
feature_xxx
```

测试文件中不包含：

```Plain
weight
responder_*
target
```

---

### `sample_submission.csv`

提交文件示例。

```Plain
row_id,target
0,0.012345
1,-0.008765
2,0.000123
```

提交文件必须包含相同列名，并覆盖测试集中的所有 `row_id`。发布包中的 `sample_submission.csv` 使用随机数作为示例预测值，仅用于展示提交格式，不代表推荐模型输出。

---

### `examples/`

示例代码目录，用于展示数据读取、基础预处理、训练验证切分、模型训练和本地推理流程。

该目录包含：

- `data_io/`：使用 `pandas` 或 `polars` 读取发布数据的示例；
- `random_strategy/`：随机基线策略示例；
- `linear_window_strategy/`：简单线性模型和滑动窗口状态示例。

---

### `timeseries_api/`

顺序推理 API 示例目录，用于帮助参赛者在本地模拟最终评测流程。

该目录包含：

- `example_main.py`：最小可运行的 `Model` 类示例；
- `main.py`：与 `example_main.py` 相同的默认示例策略；
- `runner.py`：本地顺序推理 runner；
- `run_timeseries_api.py`：命令行入口；
- `README.md`：本地运行说明。

本地 runner 的输入为发布包中的 `test_partition_*.parquet`。runner 会按递增 `time_id` 逐步向参赛者 `Model.predict(test)` 提供当前截面数据，并生成与公榜提交格式一致的：

```Plain
row_id,target
```

runner 同时会在标准输出中给出 JSON 运行报告，其中包含 `status`、`rows`、`messages` 和 `timing`。`timing` 用于检查初始化耗时、预测耗时、调用次数和超时情况。

正式私榜评测会使用相同的顺序推理语义，但评测端不会向参赛者暴露未来 feature、未公开标签、`weight`、`responder_*` 或 `target`。

---

## 字段说明（Fields）

### `row_id`

样本唯一标识。提交文件通过 `row_id` 与测试样本进行匹配。

---

### `time_id`

匿名时间索引，表示样本在全局时序中的先后关系。较大的 `time_id` 表示更靠后的样本。

`time_id` 不代表真实时间信息，也不包含可映射到外部世界的日历含义。

---

### `asset_id`

匿名标的编号。相同 `asset_id` 表示同一匿名标的的连续样本。

`asset_id` 本身不包含真实标的名称、类别、规模、排序或其他可解释含义。

---

### `weight`

样本权重，仅在训练数据中提供。

`weight` 用于最终评分中的加权零均值 $R^2$。该字段综合反映样本的成交活跃度、流动性条件、交易摩擦以及样本在整体评估中的相对重要性。参赛者可以将其用于模型训练、验证评估或预测后处理。

---

### `feature_000` 至 `feature_xxx`

匿名数值特征。所有特征均由当前及历史可见信息构造，不包含未来目标信息。

特征可能覆盖多种市场状态和统计结构，包括但不限于价格变化、成交状态、波动结构、流动性状态、路径形态以及跨标的关系等。具体字段含义、计算方式和窗口设置不会公开。

---

### `responder_00` 至 `responder_xx`

训练阶段提供的辅助响应变量。Responder 是基于未来不可见区间构造的一组匿名目标，覆盖多个预测窗口和多个市场响应维度。

这些 responder 可能反映不同类型的未来状态，例如收益类、风险类、路径类以及流动性摩擦类等。

在公榜测试阶段，测试数据不提供 `responder_*`。在公榜截止后会发布标签回补数据，该部分数据将作为扩展训练数据使用，具体以 `赛题说明` 中 Timeline 部分为准。

私有评测数据不提供 `responder_*`。

---

### `target`

最终计分目标。

`target` 是一个连续数值，表示对应匿名标的在当前时点之后某一预测窗口内的风险调整表现。最终评分以参赛者对 `target` 的预测为核心，并结合 `weight` 进行加权计算。

训练数据提供 `target`。公榜测试数据和私有评测数据不提供 `target`。公榜截止后的标签回补安排以 `赛题说明` 中 Timeline 部分为准。

---

## 数据格式（Data Format）

数据采用 Parquet 列式格式存储，并使用压缩以降低磁盘占用和读取成本。训练数据可能按连续时序分区存放。

参赛者可以使用 `polars`、`pandas`、`pyarrow` 或其他支持 Parquet 的工具读取数据。对于较大规模训练任务，建议使用分区读取或惰性加载方式。

赛题发布时会提供数据读写示例代码。

---

## 时序说明（Time\-Series Rules）

本赛题是严格时序预测任务。训练、公榜测试和私有评测数据均按照匿名时间顺序组织。

参赛者在建模、验证、特征处理和标准化过程中，不得使用未来信息。任何通过未来特征、未来分布、未来缺失模式、未来 responder 或未来目标影响当前预测的行为，均视为未来信息泄露。

最终评测的顺序推理规则详见 `赛题说明` 中的代码要求部分。
