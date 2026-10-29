"""训练入口。每个步骤读上一步写在 --work 下的中间表，产物落在 <work>/model/。

  python run.py --data-0630 <dir> --data-0823 <dir> --work <dir> <step> [--gpu N] [--seeds ...]

步骤（依赖顺序）：
  check          核对两个发布包：文件齐全、回补行按 row_id 对齐 0630 测试行
  d1-cache       D1 标准化统计量、target_bound、768 天稠密网格
  d1-rec         伪标签重构器（岭回归）
  d1-host        D1 网络（冻结头与在线头共用），每个种子一份，可分卡并行
  d1-export      d1_model.npz + d1_meta.json
  gbdt-features  570 列特征矩阵与标签
  gbdt-cat       CatBoost 多任务 × 3 种子
  gbdt-xgb       XGBoost × 3 种子
  ensemble       ensemble.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["check", "d1-cache", "d1-rec", "d1-host", "d1-export",
         "gbdt-features", "gbdt-cat", "gbdt-xgb", "ensemble"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--data-0630", type=Path, required=True)
    ap.add_argument("--data-0823", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--gpu", type=int, default=0, help="使用的 GPU 编号")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="只训练这些种子（默认取配置里的全部）")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.insert(0, str(HERE))
    from qtrc.common import Paths, check_backfill_alignment, load_config

    cfg = load_config(args.config)
    paths = Paths(args.data_0630.resolve(), args.data_0823.resolve(), args.work.resolve())
    paths.work.mkdir(parents=True, exist_ok=True)

    if args.step == "check":
        n = check_backfill_alignment(paths)
        print(f"[check] 0823 回补包 {n:,} 行与 0630 测试行逐行对齐")
    elif args.step == "d1-cache":
        from qtrc.d1 import cache
        cache.build(cfg, paths)
    elif args.step == "d1-rec":
        from qtrc.d1 import reconstructor
        reconstructor.fit(cfg, paths, "cuda:0")
    elif args.step == "d1-host":
        from qtrc.d1 import train
        train.run(cfg, paths, args.seeds or cfg["d1"]["train"]["seeds"], "cuda:0")
    elif args.step == "d1-export":
        from qtrc.d1 import export
        export.run(cfg, paths)
    elif args.step == "gbdt-features":
        from qtrc.gbdt import features
        features.build(cfg, paths)
    elif args.step == "gbdt-cat":
        from qtrc.gbdt import train
        train.train_catboost(cfg, paths, args.seeds or cfg["gbdt"]["seeds"])
    elif args.step == "gbdt-xgb":
        from qtrc.gbdt import train
        train.train_xgboost(cfg, paths, args.seeds or cfg["gbdt"]["seeds"])
    elif args.step == "ensemble":
        from qtrc import ensemble
        ensemble.run(cfg, paths)


if __name__ == "__main__":
    main()
