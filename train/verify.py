"""按公告 §4.2 自检：同一套 inference/ 代码分别装原权重与重训权重，比较预测。

  python verify.py --data-0630 <dir> --data-0823 <dir> --model <work>/model --out <dir>
                   [--start-day 738 --days 30]

做法：
  1. 从 0630 公榜测试特征里截出连续的 [start_day, start_day+days) 天，写成官方 runner
     可读的数据目录；
  2. 复制两份 inference/：一份保持原样，另一份只把 model/ 下的文件换成重训产物；
  3. 用 0630 发布包自带的官方 runner 分别跑完（限 4 线程），得到 ŷ 与 ŷ'；
  4. 报告 corr(ŷ, ŷ') 与 MAPE(ŷ, ŷ')，并用 0823 回补标签算两边的加权零均值 R²。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
DAY = 1440


def mape(a: np.ndarray, b: np.ndarray) -> float:
    """公告定义：只在 a≠0 的位置平均 |(b-a)/a|；a=0 处要求 b=0，否则记为无穷。"""
    zero = a == 0
    if np.any(b[zero] != 0):
        return float("inf")
    if zero.all():
        return 0.0
    return float(np.mean(np.abs((b[~zero] - a[~zero]) / a[~zero])))


def weighted_r2(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return 1.0 - float(np.sum(w * (y - p) ** 2) / np.sum(w * y ** 2))


def window_data(data_0630: Path, lo: int, hi: int, out: Path) -> int:
    files = sorted((data_0630 / "data/test").glob("test_partition_*.parquet"))
    frame = pl.scan_parquet(files).filter(pl.col("time_id").is_between(lo, hi - 1)).collect()
    (out / "test").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out / "test/window.parquet")
    (out / "manifest.json").write_text(json.dumps({"files": {"test": ["test/window.parquet"]}}))
    return len(frame)


def strategy_copy(src: Path, dst: Path, model: Path | None) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    if model is not None:
        for f in sorted((dst / "model").iterdir()):
            replacement = model / f.name
            if not replacement.exists():
                raise FileNotFoundError(f"重训产物缺少 {f.name}")
            shutil.copyfile(replacement, f)


def run_runner(runner: Path, data: Path, strategy: Path, output: Path) -> subprocess.Popen:
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    log = output.with_suffix(".log").open("w")
    return subprocess.Popen([sys.executable, str(runner), "--data-root", str(data),
                             "--strategy-dir", str(strategy), "--output", str(output)],
                            stdout=log, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-0630", type=Path, required=True)
    ap.add_argument("--data-0823", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="重训产物目录（<work>/model）")
    ap.add_argument("--inference", type=Path, default=HERE.parent / "inference")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-day", type=int, default=738, help="公榜测试窗是第 617–767 天")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = args.start_day * DAY, (args.start_day + args.days) * DAY
    n = window_data(args.data_0630, lo, hi, out / "data")
    print(f"窗口 time_id [{lo}, {hi})，{n:,} 行", flush=True)

    strategy_copy(args.inference, out / "original", None)
    strategy_copy(args.inference, out / "retrained", args.model.resolve())
    runner = args.data_0630 / "timeseries_api/run_timeseries_api.py"
    jobs = [run_runner(runner, out / "data", out / name, out / f"{name}.csv")
            for name in ("original", "retrained")]
    if any(job.wait() for job in jobs):
        raise SystemExit("runner 失败，见 out/*.log")

    a = pd.read_csv(out / "original.csv")
    b = pd.read_csv(out / "retrained.csv")
    if not a["row_id"].equals(b["row_id"]):
        raise SystemExit("两次运行的 row_id 序列不同")
    ya, yb = a["target"].to_numpy(), b["target"].to_numpy()
    labels = (pl.scan_parquet(sorted((args.data_0823 / "data/train").glob("*.parquet")))
              .select(["row_id", "weight", "target"]).collect().to_pandas()
              .set_index("row_id").loc[a["row_id"]])
    y, w = labels["target"].to_numpy(), labels["weight"].to_numpy()

    constant = np.ptp(ya) == 0 or np.ptp(yb) == 0
    report = {
        "window": {"start_day": args.start_day, "days": args.days, "rows": int(len(a))},
        "corr": None if constant else float(np.corrcoef(ya, yb)[0, 1]),
        "mape": mape(ya, yb),
        "max_abs_diff": float(np.max(np.abs(ya - yb))),
        "r2_original": weighted_r2(y, ya, w),
        "r2_retrained": weighted_r2(y, yb, w),
    }
    report["pass"] = bool((constant or report["corr"] >= 0.99) and report["mape"] <= 1e-3)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
