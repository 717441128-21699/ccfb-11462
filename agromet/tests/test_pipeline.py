"""需求 7 + 总体验收: 分块管线、4G 内存约束、埋点事件时空核对。"""
import gc
import json

import numpy as np
import pandas as pd
import psutil
import pytest

from agromet.pipeline import AgrometPipeline
from agromet.synth import DROUGHT, FROST

SOURCES = ["source_A_lambert", "source_B_irregular", "source_C_rotated"]


@pytest.mark.parametrize("stem", SOURCES)
def test_pipeline_end_to_end(cfg, stem):
    pipe = AgrometPipeline(cfg)
    src = next(p for p in cfg.raw_dir.iterdir() if p.stem == stem)
    task = pipe.run_source(src, method="linear")
    assert task.status == "done", task.error
    assert task.chunks == 2  # 1 月 + 3 月, 每月一块
    # 输出齐全
    for out in task.outputs:
        assert __import__("pathlib").Path(out).exists()


@pytest.mark.parametrize("stem", SOURCES)
def test_memory_under_4g(cfg, stem):
    pipe = AgrometPipeline(cfg)
    src = next(p for p in cfg.raw_dir.iterdir() if p.stem == stem)
    task = pipe.run_source(src, method="cubic")
    peak = max(m["rss_mb"] for m in task.memory_trace)
    assert peak < 4096, f"峰值内存 {peak}MB 超过 4GiB"
    # 所有采样点都在守卫阈值内
    assert all(m["ok"] for m in task.memory_trace)
    # 进程级实时 RSS 复核
    assert psutil.Process().memory_info().rss / 1024**2 < 4096


@pytest.mark.parametrize("stem", SOURCES)
def test_drought_event_matches_truth(cfg, stem):
    pipe = AgrometPipeline(cfg)
    events_path = cfg.stats_dir / f"{stem}_linear_events.parquet"
    task = pipe.run_source(
        next(p for p in cfg.raw_dir.iterdir() if p.stem == stem), method="linear")
    import pandas as pd
    df = pd.read_parquet(events_path)
    drought = df[(df["indicator_id"] == "drought_dry_spell")
                 & (df["zone_id"].isin(DROUGHT.counties))]
    assert not drought.empty, "未在埋点区县检出旱情"
    # 至少一个事件完整覆盖埋点窗口 3/01-3/25
    def covers(row):
        return pd.Timestamp(row.start_date) <= pd.Timestamp(DROUGHT.start) and \
               pd.Timestamp(row.end_date) >= pd.Timestamp(DROUGHT.end)
    hit = drought[drought.apply(covers, axis=1)]
    assert not hit.empty, f"旱情窗口未对上: {drought[['start_date','end_date']].to_dict('records')}"
    assert hit["duration_days"].max() >= 25
    # 其他区县不应出现同等强度的旱情事件 (3 月数据里唯一的 25 天干旱盒在 C03)
    other = df[(df["indicator_id"] == "drought_dry_spell")
               & (~df["zone_id"].isin(DROUGHT.counties))]
    assert (other["duration_days"] < 25).all()


@pytest.mark.parametrize("stem", SOURCES)
def test_frost_event_matches_truth(cfg, stem):
    pipe = AgrometPipeline(cfg)
    pipe.run_source(next(p for p in cfg.raw_dir.iterdir()
                         if p.stem == stem), method="linear")
    df = pd.read_parquet(cfg.stats_dir / f"{stem}_linear_events.parquet")
    frost = df[(df["indicator_id"] == "frost_streak")
               & (df["zone_id"].isin(FROST.counties))]
    assert not frost.empty, "未在埋点区县检出霜冻过程"
    def covers(row):
        return pd.Timestamp(row.start_date) <= pd.Timestamp(FROST.start) and \
               pd.Timestamp(row.end_date) >= pd.Timestamp(FROST.end)
    hit = frost[frost.apply(covers, axis=1)]
    assert not hit.empty, f"霜冻窗口未对上: {frost[['start_date','end_date']].to_dict('records')}"
    assert hit["duration_days"].max() >= 3


def test_daily_outputs_have_correct_aggregation(cfg):
    """管线产出的日尺度文件: 降水非负且温度极值关系自洽。"""
    import xarray as xr
    pipe = AgrometPipeline(cfg)
    pipe.run_source(next(p for p in cfg.raw_dir.iterdir()
                         if p.stem == "source_B_irregular"), method="linear")
    ds = xr.open_dataset(cfg.daily_dir / "source_B_irregular_linear_daily.nc")
    assert (ds["precip"] >= 0).all()
    # 边缘格点可能为 NaN, 忽略缺测后温度极值关系必须自洽
    diff = (ds["tmax"] - ds["tmin"]).values
    assert np.nanmin(diff) >= -1e-9
    assert set(ds["n_hours"].values) <= {24}
    ds.close()


def test_task_persistence(cfg):
    pipe = AgrometPipeline(cfg)
    pipe.run_source(next(p for p in cfg.raw_dir.iterdir()
                         if p.stem == "source_A_lambert"), method="nearest")
    tasks = pipe.list_tasks()
    assert any(t["status"] == "done" for t in tasks)
