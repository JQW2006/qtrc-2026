"""配置、路径与两个发布包的读取。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

FEATURES = [f"feature_{i:03d}" for i in range(323)]


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Paths:
    data_0630: Path      # public_release_20260630 根目录
    data_0823: Path      # public_release_20260823 根目录
    work: Path           # 中间表与产物

    @property
    def model(self) -> Path:
        """与 inference/model/ 同名同布局的产物目录。"""
        return self.work / "model"

    def stage(self, name: str) -> Path:
        path = self.work / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def label_files(paths: Paths) -> list[Path]:
    """带标签的全部行：0630 训练集在前，0823 回补包（原公榜测试窗）在后。

    两者 time_id 首尾相接（0630 训练集止于 888479，回补包始于 888480），
    按此顺序拼接即为全局 time_id 有序的 768 天。
    """
    main = sorted((paths.data_0630 / "data/train").glob("train_partition_*.parquet"))
    backfill = sorted((paths.data_0823 / "data/train").glob("train_partition_*.parquet"))
    if len(main) != 9 or len(backfill) != 3:
        raise FileNotFoundError(f"发布包文件数不符：0630 train {len(main)}，0823 train {len(backfill)}")
    return main + backfill


def scan_labels(paths: Paths, columns: list[str]) -> pl.LazyFrame:
    return pl.scan_parquet(label_files(paths)).select(columns)


def check_backfill_alignment(paths: Paths) -> int:
    """0823 回补行与 0630 测试行按 row_id 一一对应，且 time_id / asset_id 相同。"""
    test = sorted((paths.data_0630 / "data/test").glob("test_partition_*.parquet"))
    backfill = sorted((paths.data_0823 / "data/train").glob("train_partition_*.parquet"))
    keys = ["row_id", "time_id", "asset_id"]
    a = pl.scan_parquet(test).select(keys).collect().sort("row_id")
    b = pl.scan_parquet(backfill).select(keys).collect().sort("row_id")
    if not a.equals(b):
        raise RuntimeError("0823 回补包与 0630 测试集的 row_id/time_id/asset_id 对不上")
    return len(a)


def write_json(path: Path, payload: dict, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)
