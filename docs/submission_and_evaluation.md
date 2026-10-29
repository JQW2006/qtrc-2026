# 关于策略最终交付与评估的说明

## 策略交付

### 适用范围

本文档用于指导参赛队伍在赛事提供的 JupyterHub 环境中完成策略开发、依赖安装、提交打包和提交前检查，同时规定评测系统执行策略时使用的统一环境与目录结构。

### 运行环境约定

每支队伍拥有独立且持久化的 Home 目录和 Conda base 环境：

| **用途**   | **JupyterHub 容器内路径** |
| ---------------- | ------------------------------- |
| 用户 Home        | /home/jovyan                    |
| Python base 环境 | /opt/conda                      |
| Python 解释器    | /opt/conda/bin/python           |

本次策略提交统一使用 JupyterHub 默认的 Conda base 环境。评测系统**不查找、不激活**用户自行创建的 Conda 环境。

参赛队伍不需要执行也不应该执行 `conda init`、`conda create` 或 `conda activate` 等命令。登录 JupyterHub 后可直接使用 base 环境。

### 安装 Python 依赖

如策略需要额外的 Python 包，应在 JupyterHub Terminal 中执行：

```Bash
python -m pip install <package-name>
```

必须使用 `python -m pip`，不要使用 `sudo pip`，也不要依赖用户自行创建的 Conda 环境。

安装前后可通过以下命令确认正在使用默认环境：

```Bash
python -c 'import sys; print(sys.executable); print(sys.prefix)'
```

```Bash
python -m pip -V
```

正确结果应指向 `/opt/conda/bin/python` 与 `/opt/conda`。`python -m pip -V` 的路径也应落在 `/opt/conda` 下，而不是 `/opt/conda/envs/<name>`。

用户在 `/opt/conda` 中安装的软件包会保存到其独立的持久化目录，不会因 Notebook Server 重启而丢失，也不会影响其他队伍。

### 最终交付要求

参赛队伍须提交一个 ZIP 压缩包，并满足以下要求：

1. ZIP 根目录必须包含 `main.py`。
2. ZIP 应包含策略运行所需的源码、配置文件和模型文件。
3. ZIP 必须包含 `requirements.txt`，用于记录 Python 包及版本。
4. ZIP 不得包含完整 Conda 环境、Python 解释器、系统库或缓存目录。
5. 策略不得在评测过程中联网下载或安装依赖。
6. `main.py` 必须定义 `class Model`，并能够在 JupyterHub 默认 Python 环境中被评测系统导入；不是以 `python main.py` 作为脚本入口。
7. 所有文件路径应使用相对路径，不得写死 `/home/jovyan` 或队伍专属绝对路径，否则会出现调用错误。

推荐的参考压缩包结构：

```Plain
submission.zip
├── main.py
├── requirements.txt
├── config.yaml
├── model/
│   └── [model.pkl]
└── src/
    ├── __init__.py
    └── [strategy.py]
```

以下内容（以及其余无关文件）不应放入压缩包：

```Plain
conda/
env/
venv/
__pycache__/
.ipynb_checkpoints/
.git/
原始训练数据
与策略执行无关的大文件
```

本文所称“包含相关依赖”是指：

- ZIP 包含项目源码、配置、模型和 `requirements.txt`。
- Python 第三方包须提前安装到该队伍的 JupyterHub 默认 base 环境。
- **不要求、也不允许**将完整 Conda 环境打入 ZIP。

### 提交文件准备

在用户 Home 下建立独立的提交目录：

```Bash
mkdir -p ~/submit
```

将配置、源码和模型文件一并放入 `~/submit`，保证入口名称为 `~/submit/main.py`，然后记录当前 Python 依赖：

```Bash
cd ~/submit
python -m pip freeze > requirements.txt
```

未创建该目录的用户，主办方会在评测前代为创建并放入压缩包内所有文件。为保证下述自检运行，强烈建议用户自行创建并进行文件一致性核对。

### 提交前自检

#### 确认解释器

```Bash
python -c 'import sys; print(sys.executable); print(sys.prefix)'
```

输出必须指向 `/opt/conda`，不得指向 `/opt/conda/envs/<name>`。

#### 检查依赖状态

```Bash
python -m pip check
```

预期输出

```Bash
No broken requirements found.
```

#### 提交目录运行

评测系统会导入 `~/submit/main.py` 中的 `class Model`，初始化一次 `Model()`，再按递增 `time_id` 调用 `predict(test)`。`test` 仅含 `row_id,time_id,asset_id,feature_*`。本地请用公开发布包中的 Time-Series API runner 做同样检查；在公开包根目录执行，`--strategy-dir` 必须指向即将提交的 `~/submit`：

```Bash
python timeseries_api/run_timeseries_api.py \
  --data-root /path/to/public_release_20260630/data \
  --strategy-dir ~/submit \
  --output /path/to/submit_selfcheck.csv \
  --model-init-timeout-seconds 180
```

`--data-root` 指向主公开包的 `data/`（含 `manifest.json` 与 `test/`）。公开 test 全量约三百余万行，跑完较慢是预期行为。

通过标准：

- JSON 中 `status` 为 `"ok"`。若为 `"error"` 且含 `model_init_failed`，该次私榜评测的预测按全零计。
- `messages` 为空。出现 `predict_exception`、长度不符或非有限预测时，对应 `time_id` 已被填 `0.0`，`status` 仍可能是 `"ok"`。
- 关注 `timing` 中的 `model_init_seconds`、`predict_total_seconds`、`predict_calls`、`mean_predict_seconds`、`total_seconds`。初始化须小于 180 s；单步平均应对照赛题说明中的 50 ms。本地测耗时请将 LightGBM / PyTorch 等线程数设为 `<= 4`。
- `--output` 为两列 `row_id,target`，`row_id` 覆盖公开 test 全部样本（可与 `data/sample_submission.csv` 比对）。

**重要注意：**

`row_id` 只是样本唯一行标，供系统把预测与标签对齐。不要把它的数值当作数组下标、模型特征或训练标签。私榜按 `time_id` 顺序调用 `predict` 时，部分行只用于恢复 `Model` 内部历史、不计入分数，这些行的 `row_id` 可能为负；不要因此跳过历史更新。需要滚动窗口时，请按 `asset_id` 在内存中维护状态，参见公开包 `examples/`。

权重与辅助模块须用 `Path(__file__).resolve().parent` 定位，不要写死 `/home/jovyan`，否则可能造成评测系统读取失败。

#### 用标签回补复算公榜测试分数

标签回补包发布后，公榜提交与更新关闭。标签回补包 `public_release_20260823` 的 `data/train` 与主公开包 `test` **按 `row_id` 逐行对齐**，并补上 `weight`、`responder_*`、`target`。可用 runner 在原 test 上生成的预测 CSV（或当时的公榜提交文件）在本地复算公开 test 分数，公式与赛题说明中的加权零均值 $R^2$ 一致：

```Python
from pathlib import Path

import numpy as np
import pandas as pd

pred = pd.read_csv("/path/to/submit_selfcheck.csv")  # 列：row_id,target
labels = pd.concat(
    [
        pd.read_parquet(path, columns=["row_id", "weight", "target"])
        for path in sorted(Path("/path/to/public_release_20260823/data/train").glob("*.parquet"))
    ],
    ignore_index=True,
)
merged = labels.merge(
    pred.rename(columns={"target": "prediction"}),
    on="row_id",
    how="left",
)
prediction = pd.to_numeric(merged["prediction"], errors="coerce")
merged["prediction"] = prediction.replace([np.inf, -np.inf], np.nan).fillna(0.0)

y = merged["target"].to_numpy(dtype=np.float64)
yhat = merged["prediction"].to_numpy(dtype=np.float64)
w = merged["weight"].to_numpy(dtype=np.float64)
score = 1.0 - np.sum(w * (y - yhat) ** 2) / np.sum(w * y * y)
print(float(score), int(len(merged)))
```

缺行或非有限预测按 0 计入，多余 `row_id` 会被忽略，与公榜评分处理一致。

#### 创建压缩包

在 `~/submit` 目录中执行：

```Bash
cd ~/submit
zip -r submission.zip main.py requirements.txt [config.yaml] [src] [model]
```

如果策略不包含某个示例目录或文件，应从命令中删除对应名称。

检查 ZIP 内容：

```Bash
unzip -l submission.zip
```

必须确认 `main.py` 位于 ZIP 根目录，而不是嵌套在另一层目录中。

### 参赛队伍简要检查清单

提交前请逐项确认：

* [ ] 使用的是 `/opt/conda/bin/python`。
* [ ] 额外依赖通过 `python -m pip install` 安装。
* [ ] `python -m pip check` 未报告错误。
* [ ] ZIP 根目录包含 `main.py`。
* [ ] ZIP 包含源码、配置、模型和 `requirements.txt`。
* [ ] ZIP 不包含 Conda 环境、缓存和原始训练数据。
* [ ] 策略不依赖 Notebook 状态、手工环境变量或用户专属绝对路径。
* [ ] `~/submit` 已创建，内容与 ZIP 一致。
* [ ] 在公开包根目录用 `timeseries_api/run_timeseries_api.py` 对 `~/submit` 跑通，JSON `status` 为 `ok`，输出 `row_id` 与公开 `test` 一致。
* [ ] 计算核数设置正确（本地测时 `<= 4`），运行时间检验通过。
* [ ] 已用标签回补包按 `row_id` 对齐，复算过公开 test 的评估指标。

## 策略评估

实盘评估阶段，全部有效提交按赛题说明中的**加权零均值 $R^2$**（Weighted Zero-Mean $R^2$）计私榜成绩，公式与公榜一致。

最终名次仅在**入围答辩的 $N$ 支队伍**内部合成（$N$ 由主办方公布）：私榜 80%，评委 20%。未入围队伍不进入该合成，也不参加答辩排名。

### 私榜分数

记队伍 $i$ 在实盘评估结束时、私榜累计计分样本上的加权零均值 $R^2$ 为 $R^2_i$。推理失败、超时或非法输出按规则填 0 后照常计入。入围答辩名单按 $R^2_i$ 从高到低取前 $N$ 名。

### 最终合成分（仅入围队伍）

以下步骤只用这 $N$ 支入围队伍的分数。若某一步全体取值相同（例如 MAD 为 0，或某评委给所有队同一分数），对应标准分取 0。

#### 1. 私榜稳健标准化

入围队伍内部用中位数与绝对中位差（MAD）标准化私榜 $R^2$，减轻个别极端分数对尺度的影响。

$$
\mathrm{Median}_{LB} = \mathrm{Median}(R^2_1,\ldots,R^2_N)
$$

$$
\mathrm{MAD} = \mathrm{Median}\bigl(\lvert R^2_i - \mathrm{Median}_{LB}\rvert\bigr)
$$

$$
\sigma_{\mathrm{robust}} = 1.4826 \times \mathrm{MAD}
$$

$$
Z_{LB,i} = \frac{R^2_i - \mathrm{Median}_{LB}}{\sigma_{\mathrm{robust}}}
$$

#### 2. 评委评分标准化

入围队伍在答辩环节由 $K$ 位评委打分。原始分数量表以答辩通知为准。先按评委消去个人尺度差异，再在入围队伍内对齐。

对评委 $k$ 的全部入围打分求均值 $\mu_k$ 与标准差 $\sigma_k$，队伍 $i$ 在该评委处为

$$
z_{i,k} = \frac{\mathrm{Score}_{i,k} - \mu_k}{\sigma_k}.
$$

综合主观分取各评委标准分的平均：

$$
J_i = \frac{1}{K}\sum_{k=1}^{K} z_{i,k}.
$$

再对 $\{J_i\}$ 求均值 $\mu_J$ 与标准差 $\sigma_J$：

$$
Z_{\mathrm{Judge},i} = \frac{J_i - \mu_J}{\sigma_J}.
$$

#### 3. 融合与排名

$$
Z_{\mathrm{Final},i} = 0.8\, Z_{LB,i} + 0.2\, Z_{\mathrm{Judge},i}.
$$

按 $Z_{\mathrm{Final},i}$ **降序**得到最终名次，不再换算为百分制。私榜上差距较大时，评委权重通常不足以改变先后；$R^2$ 接近时，评委可按策略思路、完成度与答辩表现调整相对名次。
