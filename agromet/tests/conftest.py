"""共享 pytest fixtures: 在临时目录生成裁剪版合成数据 (1 月 + 3 月)。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agromet.config import Config, TargetGrid
from agromet import synth


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Config:
    root = tmp_path_factory.mktemp("agromet_data")
    c = Config(
        data_root=root, raw_dir=root / "raw", normalized_dir=root / "normalized",
        daily_dir=root / "daily", stats_dir=root / "stats",
        weights_dir=root / "weights", lineage_dir=root / "lineage",
        claims_dir=root / "claims", boundary_dir=root / "boundaries",
        tasks_dir=root / "tasks",
        target_grid=TargetGrid(
            lat_min=29.95, lat_max=31.05, lon_min=102.95, lon_max=104.05,
            resolution=0.1),
        memory_limit=4 * 1024**3, time_chunk_days=31,
    )
    c.ensure_dirs()
    synth.generate_all(c, months=[1, 3])
    return c


@pytest.fixture(scope="session")
def manifest(cfg) -> dict:
    import json
    return json.loads((cfg.raw_dir / "truth_manifest.json").read_text(encoding="utf-8"))
