"""处理链路编排: 读取 -> 重采样 -> 逐日 -> 分区统计 -> 灾害判定。

内存控制策略 (上限 4 GiB):
  * 数据源按"月"切片逐个处理, 不把全年场同时读入内存;
  * Dask 线程数受限, 单块空间大小受限 (spatial_chunk);
  * MemoryGuard 在每个阶段前后采样 RSS, 超限主动释放并报错定位;
  * 重采样/分区权重只构建一次并缓存复用。
"""
from __future__ import annotations

import dataclasses
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from .config import Config
from .io import open_dataset, NormalizedDataset
from .regrid import (
    SourceGrid, describe_source_grid, build_weights, RegridWeights,
)
from .resample import daily_dataset, validate_daily
from .zones import (
    load_zones, build_zonal_weights, ZonalWeights, zonal_statistics, Zone,
)
from .indicators import (
    IndicatorSpec, load_indicators, evaluate_indicator, extract_events,
)


# ---------------------------------------------------------------------------
# 数据集注册表
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    dataset_id: str
    path: str
    fmt: str
    variables: list[str]
    grid_type: str
    time_step_hours: float | None
    summary: dict


def scan_raw_dir(cfg: Config) -> list[DatasetEntry]:
    """扫描 raw 目录, 打开每个文件读元信息 (不加载全部数据)。"""
    entries = []
    for p in sorted(cfg.raw_dir.iterdir()):
        if p.suffix.lower() in (".nc", ".nc4", ".grib", ".grib2",
                                ".grb", ".grb2"):
            nd = open_dataset(p, chunks={})
            entries.append(DatasetEntry(
                dataset_id=p.stem, path=str(p), fmt=nd.fmt,
                variables=list(nd.ds.data_vars),
                grid_type=nd.grid_type,
                time_step_hours=nd.time_step_hours,
                summary=nd.summary(),
            ))
            nd.ds.close()
    return entries


# ---------------------------------------------------------------------------
# 内存守卫
# ---------------------------------------------------------------------------

class MemoryGuard:
    def __init__(self, limit_bytes: int, label: str = "pipeline"):
        self.limit = limit_bytes
        self.label = label
        self.proc = psutil.Process()
        self.peak = 0

    def rss(self) -> int:
        return self.proc.memory_info().rss

    def sample(self, stage: str) -> dict:
        gc.collect()
        rss = self.rss()
        self.peak = max(self.peak, rss)
        info = {"stage": stage, "rss_mb": round(rss / 1024**2, 1),
                "limit_mb": round(self.limit / 1024**2, 1),
                "peak_mb": round(self.peak / 1024**2, 1),
                "ok": rss <= self.limit}
        if rss > self.limit:
            raise MemoryError(
                f"[{self.label}] 阶段 {stage} 内存 {info['rss_mb']}MB "
                f"超过上限 {info['limit_mb']}MB")
        return info


# ---------------------------------------------------------------------------
# 任务状态
# ---------------------------------------------------------------------------

def _compare(values: np.ndarray, comparator: str, threshold: float) -> np.ndarray:
    table = {
        "<": lambda: values < threshold,
        "<=": lambda: values <= threshold,
        ">": lambda: values > threshold,
        ">=": lambda: values >= threshold,
        "==": lambda: values == threshold,
        "!=": lambda: values != threshold,
    }
    if comparator not in table:
        raise ValueError(f"不支持的比较符: {comparator}")
    return table[comparator]()

@dataclass
class TaskRecord:
    task_id: str
    dataset_id: str
    method: str
    status: str = "pending"          # pending|running|done|failed
    created_at: str = ""
    finished_at: str = ""
    chunks: int = 0
    memory_trace: list[dict] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    error: str = ""

    def save(self, tasks_dir: Path) -> None:
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / f"{self.task_id}.json").write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8")


# ---------------------------------------------------------------------------
# 管线
# ---------------------------------------------------------------------------

class AgrometPipeline:
    def __init__(self, cfg: Config | None = None,
                 indicators_path: str | Path | None = None):
        self.cfg = cfg or Config.load()
        self.cfg.ensure_dirs()
        ind_path = indicators_path or Path(__file__).resolve().parents[1] / \
            "config" / "indicators.yaml"
        self.indicators: list[IndicatorSpec] = load_indicators(ind_path)
        self.guard = MemoryGuard(self.cfg.memory_limit)
        self._zones: list[Zone] | None = None
        self._zonal_w: ZonalWeights | None = None

    # ----- 权重准备 -----
    def prepare_zones(self) -> ZonalWeights:
        if self._zonal_w is not None:
            return self._zonal_w
        geojson = self.cfg.boundary_dir / "counties.geojson"
        self._zones = [z for z in load_zones(geojson) if z.level == "county"]
        tg = self.cfg.target_grid
        cache = self.cfg.weights_dir / "zonal.json"
        if cache.exists():
            self._zonal_w = ZonalWeights.load(self.cfg.weights_dir, "zonal")
        else:
            self._zonal_w = build_zonal_weights(
                self._zones, tg.lats, tg.lons)
            self._zonal_w.save(self.cfg.weights_dir, "zonal")
        return self._zonal_w

    def prepare_regrid(self, nd: NormalizedDataset, method: str) -> RegridWeights:
        name = f"regrid_{Path(nd.source_path).stem}_{method}"
        meta = self.cfg.weights_dir / f"{name}.json"
        if meta.exists():
            return RegridWeights.load(self.cfg.weights_dir, name)
        src = describe_source_grid(nd)
        w = build_weights(src, self.cfg.target_grid, method=method)
        w.save(self.cfg.weights_dir, name)
        return w

    # ----- 单数据源全流程 -----
    def run_source(self, dataset_path: str | Path, method: str = "linear",
                   task_id: str | None = None) -> TaskRecord:
        dataset_path = Path(dataset_path)
        task_id = task_id or f"task_{dataset_path.stem}_{method}_{int(time.time())}"
        task = TaskRecord(
            task_id=task_id, dataset_id=dataset_path.stem, method=method,
            status="running", created_at=pd.Timestamp.now().isoformat(timespec="seconds"))
        task.memory_trace.append(self.guard.sample("start"))
        try:
            nd = open_dataset(dataset_path, chunks={"time": self.cfg.time_chunk_days * 24})
            weights = self.prepare_regrid(nd, method)
            zonal_w = self.prepare_zones()
            zones = [z for z in self._zones if z.zone_id in zonal_w.zone_ids]

            times = pd.DatetimeIndex(nd.ds["time"].values)
            month_starts = pd.date_range(
                times[0].to_period("M").to_timestamp(),
                times[-1], freq="MS")
            stats_frames, event_rows = [], []
            daily_out = self.cfg.daily_dir / f"{dataset_path.stem}_{method}_daily.nc"
            first_write = True

            for ms in month_starts:
                me = ms + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
                chunk = nd.ds.sel(time=slice(ms, me)).load()
                if chunk.sizes.get("time", 0) == 0:
                    # 该月无数据 (裁剪时段/缺月), 跳过而非报错
                    continue
                task.memory_trace.append(
                    self.guard.sample(f"load-{ms.strftime('%Y%m')}"))

                # 重采样每个变量
                rg_vars = {}
                for var in chunk.data_vars:
                    rg_vars[var] = self._regrid_chunk(chunk[var], weights, var)
                rg_ds = xr.Dataset(rg_vars)
                task.memory_trace.append(
                    self.guard.sample(f"regrid-{ms.strftime('%Y%m')}"))

                # 逐小时 -> 逐日
                daily = daily_dataset(rg_ds)
                task.memory_trace.append(
                    self.guard.sample(f"daily-{ms.strftime('%Y%m')}"))
                self._append_netcdf(daily, daily_out, first_write)
                first_write = False

                # 分区统计: 温度类取面积加权平均; 降水同时存平均强度(mm)
                # 与面积加权总量(mm·m²), 分别服务灾害阈值与总量分析
                for var in daily.data_vars:
                    if var == "n_hours":
                        continue
                    stats = ["sum", "mean"] if var == "precip" else ["mean"]
                    for stat in stats:
                        stats_frames.append(zonal_statistics(
                            daily[var], zonal_w, zones, stat=stat))

                # 灾害指标 (网格 + 区县级事件); aggregate 类不产生逐日事件
                for spec in self.indicators:
                    if spec.variable not in daily.data_vars:
                        continue
                    if spec.type == "aggregate":
                        continue
                    grid_hit = evaluate_indicator(daily, spec)
                    event_rows.extend(self._events_for_zones(
                        grid_hit, daily, spec, zonal_w, zones))

                task.chunks += 1
                del chunk, rg_ds, daily
                gc.collect()

            stats_df = pd.concat(stats_frames, ignore_index=True)
            stats_path = self.cfg.stats_dir / f"{dataset_path.stem}_{method}_zonal.parquet"
            stats_df.to_parquet(stats_path, index=False)

            events_df = pd.DataFrame(event_rows)
            events_path = self.cfg.stats_dir / f"{dataset_path.stem}_{method}_events.parquet"
            if not events_df.empty:
                events_df.to_parquet(events_path, index=False)

            task.outputs = [str(daily_out), str(stats_path), str(events_path)]
            task.status = "done"
            task.finished_at = pd.Timestamp.now().isoformat(timespec="seconds")
            task.memory_trace.append(self.guard.sample("end"))
            nd.ds.close()
        except Exception as exc:  # noqa: BLE001
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
            task.finished_at = pd.Timestamp.now().isoformat(timespec="seconds")
        task.save(self.cfg.tasks_dir)
        return task

    def _regrid_chunk(self, da: xr.DataArray, weights: RegridWeights,
                      var_name: str) -> xr.DataArray:
        from .regrid import regrid_dataarray
        return regrid_dataarray(da.load(), weights, var_name)

    def _append_netcdf(self, daily: xr.Dataset, path: Path, first: bool) -> None:
        if first:
            daily.to_netcdf(path, engine="netcdf4", mode="w")
        else:
            # 读回合并再写, 避免 netCDF4 沿 unlimited dim 追加的编码问题
            existing = xr.open_dataset(path)
            merged = xr.concat([existing.load(), daily], dim="time")
            existing.close()
            merged.to_netcdf(path, engine="netcdf4", mode="w")
            merged.close()

    def _events_for_zones(self, grid_hit: xr.DataArray, daily: xr.Dataset,
                          spec: IndicatorSpec, zonal_w: ZonalWeights,
                          zones: list[Zone]) -> list[dict]:
        """把网格命中序列聚合成区县级事件列表。

        事件阈值 (如降水 < 0.1mm) 是强度量纲, 因此这里始终用面积加权
        **平均** (mm/degC), 与存储用的体积总量 (sum) 解耦。
        """
        rows = []
        # 区县级命中: 面积加权平均后按指标阈值重新判定
        zone_mean = zonal_statistics(
            daily[spec.variable], zonal_w, zones, stat="mean")
        for zid in zonal_w.zone_ids:
            sub = zone_mean[zone_mean.zone_id == zid].set_index("time")["value"]
            sub.index = pd.DatetimeIndex(sub.index)
            if spec.condition:
                # 区县级复合条件: 目前 condition 多变量在网格层已判定,
                # 区县层退化为对主变量的阈值判断 (condition 内主变量比较)
                hit = pd.Series(
                    _compare(sub.values, spec.comparator, spec.threshold)
                    if spec.comparator else np.ones(len(sub), dtype=bool),
                    index=sub.index, dtype=bool)
            else:
                hit = pd.Series(_compare(sub.values, spec.comparator, spec.threshold),
                                index=sub.index, dtype=bool)
            if spec.type == "threshold_count":
                events = extract_events(hit.astype("int8"), spec, sub)
            elif spec.type == "consecutive":
                counter = hit.astype(int).groupby(
                    (~hit).cumsum()).cumsum().astype("int16")
                events = extract_events(counter, spec, sub)
            else:
                continue
            for ev in events:
                rows.append({
                    "task_dataset": daily.attrs.get("source", ""),
                    "zone_id": zid,
                    **dataclasses.asdict(ev),
                })
        return rows

    # ----- 任务清单读取 -----
    def list_tasks(self) -> list[dict]:
        out = []
        for p in sorted(self.cfg.tasks_dir.glob("*.json")):
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return out
