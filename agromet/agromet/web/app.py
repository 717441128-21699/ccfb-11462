"""农险气象网页后端 (FastAPI)。

五个板块:
  /api/datasets   数据集
  /api/tasks      处理任务
  /api/zones      区县统计
  /api/hazards    灾害判定
  /api/claims     理赔查询 (含格点级溯源)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config
from ..pipeline import AgrometPipeline, scan_raw_dir
from ..io import open_dataset
from ..indicators import load_indicators, IndicatorSpec
from ..zones import load_zones, Zone
from ..lineage import trace_parcel, save_report
from ..regrid import RegridWeights
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="农险气象格点数据处理平台", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _cfg() -> Config:
    return Config.load()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# 1. 数据集
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
def list_datasets():
    cfg = _cfg()
    entries = scan_raw_dir(cfg)
    return [e.summary | {"dataset_id": e.dataset_id, "path": e.path,
                         "fmt": e.fmt, "grid_type": e.grid_type,
                         "time_step_hours": e.time_step_hours}
            for e in entries]


@app.get("/api/datasets/{dataset_id}")
def dataset_detail(dataset_id: str):
    cfg = _cfg()
    for p in cfg.raw_dir.iterdir():
        if p.stem == dataset_id:
            nd = open_dataset(p, chunks={})
            summary = nd.summary()
            nd.ds.close()
            return summary
    raise HTTPException(404, "数据集不存在")


# ---------------------------------------------------------------------------
# 2. 处理任务
# ---------------------------------------------------------------------------

class TaskCreate(BaseModel):
    dataset_id: str
    method: str = "linear"


@app.post("/api/tasks")
def create_task(body: TaskCreate):
    cfg = _cfg()
    target = None
    for p in cfg.raw_dir.iterdir():
        if p.stem == body.dataset_id:
            target = p
            break
    if target is None:
        raise HTTPException(404, "数据集不存在")
    pipe = AgrometPipeline(cfg)
    task = pipe.run_source(target, method=body.method)
    return json.loads(json.dumps(task.__dict__, ensure_ascii=False))


@app.get("/api/tasks")
def list_tasks():
    return AgrometPipeline(_cfg()).list_tasks()


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str):
    p = _cfg().tasks_dir / f"{task_id}.json"
    if not p.exists():
        raise HTTPException(404, "任务不存在")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 3. 区县统计
# ---------------------------------------------------------------------------

@app.get("/api/zones/boundaries")
def zone_boundaries():
    cfg = _cfg()
    p = cfg.boundary_dir / "counties.geojson"
    if not p.exists():
        raise HTTPException(404, "边界文件不存在, 请先运行 synth")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/zones/stats")
def zone_stats(dataset: str, method: str = "linear",
               variable: str = "tmean", date: str | None = None):
    cfg = _cfg()
    p = cfg.stats_dir / f"{dataset}_{method}_zonal.parquet"
    if not p.exists():
        raise HTTPException(404, "统计结果不存在, 请先运行处理任务")
    df = pd.read_parquet(p)
    df = df[df["variable"] == variable]
    if date:
        df = df[df["time"].dt.strftime("%Y-%m-%d") == date]
    df["time"] = df["time"].dt.strftime("%Y-%m-%d")
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in list(rec.items()):
            if isinstance(v, float) and not np.isfinite(v):
                rec[k] = None
    return records


@app.get("/api/zones/timeseries")
def zone_timeseries(dataset: str, method: str, zone_id: str,
                    variable: str = "tmean", stat: str = "mean"):
    cfg = _cfg()
    p = cfg.stats_dir / f"{dataset}_{method}_zonal.parquet"
    if not p.exists():
        raise HTTPException(404, "统计结果不存在")
    df = pd.read_parquet(p)
    df = df[(df["zone_id"] == zone_id) & (df["variable"] == variable)
            & (df["stat"] == stat)].sort_values("time")
    return [{"date": t.strftime("%Y-%m-%d"), "value": None if pd.isna(v) else float(v)}
            for t, v in zip(df["time"], df["value"])]


# ---------------------------------------------------------------------------
# 4. 灾害判定
# ---------------------------------------------------------------------------

@app.get("/api/hazards/indicators")
def indicators():
    path = ROOT / "config" / "indicators.yaml"
    specs = load_indicators(path)
    import dataclasses
    return [dataclasses.asdict(s) for s in specs]


@app.get("/api/hazards/events")
def hazard_events(dataset: str, method: str = "linear",
                  zone_id: str | None = None,
                  indicator_id: str | None = None):
    cfg = _cfg()
    p = cfg.stats_dir / f"{dataset}_{method}_events.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    if zone_id:
        df = df[df["zone_id"] == zone_id]
    if indicator_id:
        df = df[df["indicator_id"] == indicator_id]
    records = df.to_dict(orient="records")
    # NaN/Inf 不是合法 JSON, 统一转 None
    for rec in records:
        for k, v in list(rec.items()):
            if isinstance(v, float) and not np.isfinite(v):
                rec[k] = None
    return records


@app.get("/api/hazards/grid")
def hazard_grid(dataset: str, method: str, indicator_id: str,
                date: str | None = None):
    """返回某日指标命中网格 (用于地图着色)。"""
    cfg = _cfg()
    daily_path = cfg.daily_dir / f"{dataset}_{method}_daily.nc"
    if not daily_path.exists():
        raise HTTPException(404, "日尺度网格不存在")
    specs = {s.indicator_id: s for s in
             load_indicators(ROOT / "config" / "indicators.yaml")}
    if indicator_id not in specs:
        raise HTTPException(404, "指标不存在")
    ds = xr.open_dataset(daily_path)
    spec = specs[indicator_id]
    if spec.variable not in ds.data_vars:
        ds.close()
        raise HTTPException(400, f"日数据缺少变量 {spec.variable}")
    if date and "time" in ds.dims:
        ds = ds.sel(time=pd.Timestamp(date), method="nearest")
    from ..indicators import evaluate_indicator
    hit = evaluate_indicator(ds, spec)
    if "time" in hit.dims:
        hit = hit.isel(time=0)
    payload = {
        "lat": hit["lat"].values.tolist(),
        "lon": hit["lon"].values.tolist(),
        "values": np.nan_to_num(hit.values, nan=0).tolist(),
    }
    ds.close()
    return payload


# ---------------------------------------------------------------------------
# 5. 理赔查询 / 溯源
# ---------------------------------------------------------------------------

class ClaimRequest(BaseModel):
    lon: float
    lat: float
    start_date: str
    end_date: str
    variable: str = "tmin"
    sources: list[str] | None = None  # 默认全部已处理源
    name: str = "理赔点"


@app.post("/api/claims/trace")
def claim_trace(body: ClaimRequest):
    cfg = _cfg()
    # 找到所有可用数据源及其重采样权重
    weight_files = list(cfg.weights_dir.glob("regrid_*.json"))
    if not weight_files:
        raise HTTPException(400, "尚无重采样权重, 请先运行处理任务")
    weights_by_source, ds_by_source = {}, {}
    for meta_path in weight_files:
        name = meta_path.stem  # regrid_<stem>_<method>
        w = RegridWeights.load(cfg.weights_dir, name)
        # 从权重名反推数据源文件名
        parts = name.split("_")
        method = parts[-1]
        stem = "_".join(parts[1:-1])
        if body.sources and stem not in body.sources:
            continue
        src_file = next((p for p in cfg.raw_dir.iterdir() if p.stem == stem), None)
        if src_file is None:
            continue
        nd = open_dataset(src_file, chunks={})
        if body.variable not in nd.ds.data_vars:
            nd.ds.close()
            continue
        dates = pd.date_range(body.start_date, body.end_date, freq="1D")
        ds_by_source[stem] = nd.ds.sel(time=slice(dates[0], dates[-1] + pd.Timedelta(hours=23))).load()
        weights_by_source[stem] = w
    if not weights_by_source:
        raise HTTPException(400, "没有包含该变量的已处理数据源")

    parcel = Zone(zone_id=body.name, name=body.name,
                  geometry=Point(body.lon, body.lat), level="parcel")
    dates = pd.date_range(body.start_date, body.end_date, freq="1D")
    report = trace_parcel(
        parcel, dates, weights_by_source, ds_by_source, body.variable)
    from ..lineage import save_report, _sanitize
    report = _sanitize(report)
    save_report(report, cfg.claims_dir,
                f"claim_{body.lat:.3f}_{body.lon:.3f}_{body.start_date}")
    return JSONResponse(report)


@app.get("/api/claims")
def list_claims():
    cfg = _cfg()
    out = []
    for p in sorted(cfg.claims_dir.glob("*.json")):
        out.append({"file": p.stem,
                    "updated": pd.Timestamp(p.stat().st_mtime, unit="s").isoformat()})
    return out


@app.get("/api/claims/{name}")
def get_claim(name: str):
    p = _cfg().claims_dir / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, "溯源报告不存在")
    return json.loads(p.read_text(encoding="utf-8"))
