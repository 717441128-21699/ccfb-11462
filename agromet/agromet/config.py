"""全局配置: 目录、目标网格、内存与分块参数。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


# 项目根目录 (agromet/ 的上一级)
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


@dataclass
class TargetGrid:
    """统一目标经纬网格 (等经纬度, 单调递增)."""

    lat_min: float = 29.85
    lat_max: float = 31.15
    lon_min: float = 102.85
    lon_max: float = 104.15
    resolution: float = 0.1  # 度

    @property
    def lats(self):
        import numpy as np

        return np.arange(self.lat_min, self.lat_max + 0.5 * self.resolution,
                         self.resolution)

    @property
    def lons(self):
        import numpy as np

        return np.arange(self.lon_min, self.lon_max + 0.5 * self.resolution,
                         self.resolution)


@dataclass
class Config:
    data_root: Path = DATA_ROOT
    raw_dir: Path = DATA_ROOT / "raw"
    normalized_dir: Path = DATA_ROOT / "normalized"
    daily_dir: Path = DATA_ROOT / "daily"
    stats_dir: Path = DATA_ROOT / "stats"
    weights_dir: Path = DATA_ROOT / "weights"
    lineage_dir: Path = DATA_ROOT / "lineage"
    claims_dir: Path = DATA_ROOT / "claims"
    boundary_dir: Path = DATA_ROOT / "boundaries"
    tasks_dir: Path = DATA_ROOT / "tasks"
    target_grid: TargetGrid = field(default_factory=TargetGrid)
    # 内存约束: 单 worker 峰值上限 (字节), 默认 4 GiB
    memory_limit: int = 4 * 1024**3
    # 时间分块: 每次处理的天数 (逐月)
    time_chunk_days: int = 31
    dask_threads: int = 2
    # 单块格点数上限, 超出则进一步切分空间块
    spatial_chunk: int = 256 * 256

    def ensure_dirs(self) -> None:
        for p in [
            self.raw_dir, self.normalized_dir, self.daily_dir, self.stats_dir,
            self.weights_dir, self.lineage_dir, self.claims_dir,
            self.boundary_dir, self.tasks_dir,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = v
            elif isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        if path is None:
            path = ROOT / "config" / "config.yaml"
        path = Path(path)
        if not path.exists():
            cfg = cls()
            cfg.ensure_dirs()
            return cfg
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tg = raw.pop("target_grid", None)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        for key in ("data_root", "raw_dir", "normalized_dir", "daily_dir",
                    "stats_dir", "weights_dir", "lineage_dir", "claims_dir",
                    "boundary_dir", "tasks_dir"):
            if key in kwargs:
                kwargs[key] = Path(kwargs[key])
        cfg = cls(**kwargs)
        if tg:
            cfg.target_grid = TargetGrid(**tg)
        cfg.ensure_dirs()
        return cfg
