"""理赔判定的格点级溯源。

一条理赔判定的完整证据链:
  地块 (经纬度/多边形)
    -> 覆盖的目标网格点 + 面积占比 (zones.parcel_cells)
    -> 每个目标点由哪些源格点以何种权重插值而来 (regrid.source_contributors)
    -> 这些源格点在判定日期的原始物理值 (从标准化后的源数据切片)
    -> 指标命中结果与公式参数

所有材料落盘为 JSON, 网页"理赔查询"页直接展示。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

from .regrid import RegridWeights, source_contributors
from .zones import parcel_cells, Zone


def _slice_source_values(nd_ds: xr.Dataset, var: str,
                         flat_index: int,
                         source_shape: tuple[int, int]) -> list[dict]:
    """取源数据某格点在已加载时间切片内的全部小时级原始值。"""
    j, i = divmod(flat_index, source_shape[1])
    sub = nd_ds[var]
    if sub.ndim == 3:
        vals = sub.isel({sub.dims[1]: j, sub.dims[2]: i}).values
        times = sub["time"].values
    else:
        vals = sub.values
        times = sub["time"].values
    return [{"time": pd.Timestamp(t).isoformat(),
             "value": None if np.isnan(v) else float(v)}
            for t, v in zip(times, vals)]


def trace_parcel(
    parcel: Zone,
    dates: pd.DatetimeIndex,
    weights_by_source: dict[str, RegridWeights],
    source_datasets: dict[str, xr.Dataset],
    variable: str,
    indicator_spec: dict | None = None,
) -> dict:
    """构建单个地块/单点在给定日期段的溯源报告。

    weights_by_source : {数据源 id: RegridWeights}
    source_datasets   : {数据源 id: 标准化后的 xr.Dataset (含 time 维)}
    """
    # 同一套目标网格, 取任一权重对象的目标坐标即可
    any_w = next(iter(weights_by_source.values()))
    cells = parcel_cells(parcel.geometry, any_w.target_lats, any_w.target_lons)

    sources_report = {}
    for src_id, w in weights_by_source.items():
        ds = source_datasets[src_id]
        cell_details = []
        for cell in cells:
            t_idx = cell["cell_index"]
            contributors = source_contributors(w, t_idx)
            for c in contributors:
                raw_series = _slice_source_values(
                    ds, variable, c["source_flat_index"], w.source_shape)
                c_detail = dict(c)
                c_detail["raw_values"] = raw_series
                cell_details.append({
                    "target_cell": {k: cell[k] for k in
                                    ("cell_index", "ji", "lat", "lon",
                                     "area_fraction")},
                    "source": c_detail,
                })
        # 目标网格值 (重采样后, 小时级), 便于和源值加权核对
        target_series = None
        if variable in ds.data_vars:
            from .regrid import regrid_dataarray
            rg = regrid_dataarray(ds[variable], w, variable)
            target_series = [
                {"time": pd.Timestamp(t).isoformat(),
                 "values": np.round(rg.sel(time=t).values, 4).tolist()}
                for t in rg["time"].values
            ]
        sources_report[src_id] = {
            "grid_type": w.source_kind,
            "method": w.method,
            "target_series": target_series,
            "contributors": cell_details,
        }

    report = {
        "parcel": {
            "id": parcel.zone_id, "name": parcel.name,
            "geometry": parcel.geometry.__geo_interface__,
        },
        "variable": variable,
        "dates": [pd.Timestamp(d).date().isoformat() for d in dates],
        "indicator": indicator_spec,
        "target_cells": cells,
        "sources": sources_report,
    }
    return report


def save_report(report: dict, out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(_sanitize(report), ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化: {type(o)}")


def _sanitize(obj):
    """递归把 NaN/Inf 替换为 None, 保证输出合法 JSON。"""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return v if np.isfinite(v) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
