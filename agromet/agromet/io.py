"""NetCDF / GRIB 读取与标准化。

输出 :class:`NormalizedDataset`, 统一约定:
  * 标准变量 tmin/tmax/temp/precip, 单位 degC / mm(每时间步累计)
  * 坐标名统一为 time / lat / lon; 二维投影网格保留 (y, x) 维并带 2D lat/lon
  * 经度归一到 [-180, 180), 纬度升序, 时间为 UTC datetime64
  * 附带数据源元数据与 (投影网格的) pyproj CRS
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .units import (
    CANONICAL_DESC, CANONICAL_UNITS, recognize_unit, recognize_variable,
    convert_values,
)

GRIB_SUFFIXES = {".grib", ".grib2", ".grb", ".grb2", ".grib1"}
NETCDF_SUFFIXES = {".nc", ".nc4", ".netcdf"}


@dataclass
class VariableAudit:
    """单个变量的识别/换算审计记录。"""
    original_name: str
    canonical_name: str | None
    original_unit: str | None
    unit_kind: str
    converted_unit: str | None
    note: str = ""


@dataclass
class NormalizedDataset:
    ds: xr.Dataset
    source_path: str
    fmt: str                         # "netcdf" | "grib"
    crs: Any = None                  # pyproj.CRS 或 None (经纬网)
    grid_type: str = "regular_ll"    # regular_ll | projected_2d
    audits: list[VariableAudit] = field(default_factory=list)
    time_step_hours: float | None = None

    def summary(self) -> dict:
        d = self.ds
        return {
            "source_path": self.source_path,
            "format": self.fmt,
            "grid_type": self.grid_type,
            "crs": str(self.crs) if self.crs else "EPSG:4326",
            "dims": dict(d.sizes),
            "variables": list(d.data_vars),
            "coords": list(d.coords),
            "time_start": str(pd.Timestamp(d.time.values[0])) if "time" in d.coords else None,
            "time_end": str(pd.Timestamp(d.time.values[-1])) if "time" in d.coords else None,
            "time_step_hours": self.time_step_hours,
            "audits": [dataclasses.asdict(a) for a in self.audits],
        }


# ---------------------------------------------------------------------------
# 坐标标准化
# ---------------------------------------------------------------------------

_LAT_NAMES = ("lat", "latitude", "y_lat", "xlat", "xlats", "nav_lat", "olat", "纬度")
_LON_NAMES = ("lon", "long", "longitude", "xlon", "xlons", "nav_lon", "olon", "经度")
_TIME_NAMES = ("time", "datetime", "date", "valid_time", "t")


def _find_coord(names: tuple[str, ...], coords) -> str | None:
    norm = {str(c).strip().lower(): c for c in coords}
    for n in names:
        if n in norm:
            return norm[n]
    # 模糊包含
    for c in coords:
        cl = str(c).strip().lower()
        if any(n in cl for n in names):
            return c
    return None


def _fix_lon_180(lon: np.ndarray) -> np.ndarray:
    return np.where(lon > 180.0, lon - 360.0, lon)


def _coords_in_degrees(da: xr.DataArray, default: bool = True) -> bool:
    u = str(da.attrs.get("units", "")).lower()
    if "radian" in u:
        return False
    return default


def _angle_to_degrees(values: np.ndarray, da: xr.DataArray) -> np.ndarray:
    u = str(da.attrs.get("units", "")).lower()
    if "radian" in u:
        return np.degrees(values)
    # 个别数据把度放大 100/1000 倍存储 (整数化)
    scale = da.attrs.get("scale_factor")
    return values * scale if scale else values


def standardize_coords(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """把各种坐标命名/方向/单位整理成 time/lat/lon 约定。返回 (ds, info)。"""
    info: dict[str, Any] = {"two_d": False}
    renames: dict[str, str] = {}

    lat_name = _find_coord(_LAT_NAMES, ds.coords)
    lon_name = _find_coord(_LON_NAMES, ds.coords)
    time_name = _find_coord(_TIME_NAMES, ds.coords)

    if lat_name and lat_name != "lat":
        renames[lat_name] = "lat"
    if lon_name and lon_name != "lon":
        renames[lon_name] = "lon"
    if time_name and time_name != "time":
        renames[time_name] = "time"
    if renames:
        ds = ds.rename(renames)

    # ---- 时间 ----
    if "time" in ds.coords and not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = _decode_time(ds)

    # ---- 纬度/经度 ----
    if "lat" in ds.coords and "lon" in ds.coords:
        lat = ds["lat"]
        lon = ds["lon"]
        lat_vals = _angle_to_degrees(lat.values.astype(float), lat)
        lon_vals = _angle_to_degrees(lon.values.astype(float), lon)
        lon_vals = _fix_lon_180(lon_vals)
        info["two_d"] = lat.ndim == 2
        if info["two_d"]:
            # 二维坐标: 保留 (y, x) 维, 仅做单位/经度换算, 不排序
            ds = ds.assign_coords(
                lat=(lat.dims, lat_vals, {**lat.attrs,
                                          "units": "degrees_north"}),
                lon=(lon.dims, lon_vals, {**lon.attrs,
                                          "units": "degrees_east"}))
        else:
            ds = ds.assign_coords(lat=("lat", lat_vals, lat.attrs),
                                  lon=("lon", lon_vals, lon.attrs))
            ds["lat"].attrs.setdefault("units", "degrees_north")
            ds["lon"].attrs.setdefault("units", "degrees_east")
            # 纬度统一升序; 经度排序
            if ds["lat"].values[0] > ds["lat"].values[-1]:
                ds = ds.isel(lat=slice(None, None, -1))
            order = np.argsort(ds["lon"].values)
            if not np.array_equal(order, np.arange(ds.sizes["lon"])):
                ds = ds.isel(lon=order)
    return ds, info


def _decode_time(ds: xr.Dataset) -> xr.Dataset:
    """容错时间解码: CF 字符串失败时按常见 'hours/days since' 手动解。"""
    t = ds["time"]
    units = str(t.attrs.get("units", ""))
    calendar = t.attrs.get("calendar", "standard")
    m = __import__("re").match(
        r"(seconds|minutes|hours|days)\s+since\s+(.+)", units.strip())
    try:
        if m:
            unit, since = m.group(1), m.group(2).strip()
            origin = pd.Timestamp(since)
            delta = pd.to_timedelta(t.values.astype(float), unit=unit)
            ds = ds.assign_coords(time=origin + delta)
        else:
            # 最后尝试: 当字符串日期解析
            ds = ds.assign_coords(time=pd.to_datetime(t.values))
    except Exception:
        ds = ds.assign_coords(time=pd.to_datetime(t.values.astype(str)))
    ds["time"].encoding["calendar"] = calendar
    return ds


# ---------------------------------------------------------------------------
# 变量标准化
# ---------------------------------------------------------------------------

def _grib_fallback(attrs: dict) -> tuple[str, str | None] | None:
    """cfgrib 对部分参数给不出 shortName 时, 按 GRIB2 discipline/category/number 识别。

    返回 (标准变量名, 推断单位); 推断单位用于弥补 GRIB_units=unknown 的情况。
    """
    d = attrs.get("GRIB_discipline")
    cat = attrs.get("GRIB_parameterCategory")
    num = attrs.get("GRIB_parameterNumber")
    if d is None or cat is None:
        return None
    if d == 0 and cat == 0 and num == 0:
        return "temp", "K"
    if d == 0 and cat == 0 and num == 11:
        return "tmax", "K"
    if d == 0 and cat == 0 and num == 12:
        return "tmin", "K"
    if d == 0 and cat == 1:               # 水文/降水类
        return "precip", "kg m-2 s-1"
    return None


def standardize_variables(
    ds: xr.Dataset, time_step_hours: float | None
) -> tuple[xr.Dataset, list[VariableAudit]]:
    audits: list[VariableAudit] = []
    new_vars: dict[str, xr.DataArray] = {}
    for name, da in list(ds.data_vars.items()):
        fallback = _grib_fallback(da.attrs)
        std_name = recognize_variable(name, da.attrs)
        inferred_unit = fallback[1] if fallback else None
        if std_name is None and fallback is not None:
            std_name = fallback[0]
        raw_unit = da.attrs.get("units")
        # GRIB 给出的单位为 unknown/缺失时采用兜底推断单位
        if (raw_unit is None or str(raw_unit).lower() == "unknown") and inferred_unit:
            raw_unit = inferred_unit
        spec = recognize_unit(raw_unit)
        audit = VariableAudit(
            original_name=str(name),
            canonical_name=std_name,
            original_unit=str(raw_unit) if raw_unit is not None else None,
            unit_kind=spec.kind,
            converted_unit=None,
        )
        if std_name is None:
            audit.note = "unrecognized variable, dropped"
            audits.append(audit)
            continue

        values = da.astype(float)
        if spec.kind == "unknown":
            # 无单位信息时按变量类型给默认假设并记录
            if std_name == "precip":
                audit.note = "no unit info, assumed mm per time step"
                spec.kind = "precip_amount"
            elif std_name in ("tmin", "tmax", "temp"):
                # 数值范围启发: 全在 200~330 之间视为开尔文
                sample = float(np.nanmin(values.values[:5])) if values.size else 0.0
                if sample > 180:
                    spec = recognize_unit("K")
                    audit.note = "no unit info, inferred kelvin from value range"
                else:
                    audit.note = "no unit info, assumed degC"
                    spec.kind = "temperature"
        if spec.kind != "unknown":
            values = xr.apply_ufunc(convert_values, values, kwargs={"spec": spec},
                                    dask="parallelized")
        # 降水通量 -> 该时间步累计量 (mm)
        if std_name == "precip" and spec.kind == "precip_rate":
            if not time_step_hours:
                raise ValueError(
                    f"降水通量 {name} 需要时间步长才能积分, 但无法推断 time_step_hours")
            values = values * time_step_hours
            audit.note = f"flux integrated over {time_step_hours}h to mm"
        values.attrs = {
            "units": CANONICAL_UNITS[std_name],
            "long_name": CANONICAL_DESC[std_name],
            "source_variable": str(name),
            "source_units": str(raw_unit) if raw_unit is not None else "",
        }
        audit.converted_unit = CANONICAL_UNITS[std_name]
        audits.append(audit)
        if std_name in new_vars:
            # 同名标准变量冲突时保留先出现的, 并记录
            audits[-1].note = (audit.note + " | duplicate canonical name, kept first").strip(" |")
            continue
        new_vars[std_name] = values
    clean = ds[[]].assign(**new_vars)
    clean.attrs.update({k: v for k, v in ds.attrs.items() if _jsonable(v)})
    return clean, audits


def _jsonable(v) -> bool:
    return isinstance(v, (str, int, float, bool, np.generic))


# ---------------------------------------------------------------------------
# 投影 CRS 推断
# ---------------------------------------------------------------------------

def _extract_crs(ds: xr.Dataset):
    """从 CF grid_mapping / GRIB attrs 推断 pyproj.CRS。"""
    try:
        from pyproj import CRS
    except Exception:
        return None
    # CF grid_mapping 变量
    for var in ds.data_vars.values():
        gm = var.attrs.get("grid_mapping")
        if gm and gm in ds.variables:
            try:
                return CRS.from_cf(ds[gm].attrs)
            except Exception:
                pass
    for coord in ds.coords.values():
        gm = coord.attrs.get("grid_mapping")
        if gm and gm in ds.variables:
            try:
                return CRS.from_cf(ds[gm].attrs)
            except Exception:
                pass
    # GRIB 网格定义: 模板 30 = Lambert, 或 gridType=lambert (cfgrib 把属性挂在变量上)
    gdt = ds.attrs.get("GRIB_gridDefinitionTemplateNumber")
    for var in ds.data_vars.values():
        if gdt is None:
            gdt = var.attrs.get("GRIB_gridDefinitionTemplateNumber")
        if str(var.attrs.get("GRIB_gridType", "")).lower() == "lambert":
            a = var.attrs
            return CRS.from_dict({
                "proj": "lcc",
                "lat_1": a.get("GRIB_Latin1InDegrees"),
                "lat_2": a.get("GRIB_Latin2InDegrees"),
                "lat_0": a.get("GRIB_LaDInDegrees"),
                "lon_0": a.get("GRIB_LoVInDegrees"),
                "R": 6371200.0,
            })
    if gdt == 30:
        return CRS.from_dict({
            "proj": "lcc",
            "lat_1": ds.attrs.get("GRIB_Latin1InDegrees"),
            "lat_2": ds.attrs.get("GRIB_Latin2InDegrees"),
            "lat_0": ds.attrs.get("GRIB_LaDInDegrees"),
            "lon_0": ds.attrs.get("GRIB_LoVInDegrees"),
            "R": 6371200.0,
        })
    return None


def _infer_time_step(ds: xr.Dataset) -> float | None:
    if "time" not in ds.coords or ds.sizes.get("time", 0) < 2:
        return None
    diffs = np.diff(ds["time"].values).astype("timedelta64[s]").astype(float)
    med = float(np.median(diffs))
    return med / 3600.0 if med > 0 else None


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def open_dataset(path: str | Path, fmt: str | None = None,
                 chunks: dict | None = None) -> NormalizedDataset:
    """读取单个 NetCDF/GRIB 文件并标准化。

    chunks: 透传给 xarray 的 dask 分块 (如 {"time": 24}); None 表示懒加载默认块。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if fmt is None:
        if suffix in GRIB_SUFFIXES:
            fmt = "grib"
        elif suffix in NETCDF_SUFFIXES:
            fmt = "netcdf"
        else:
            raise ValueError(f"无法从后缀判断格式: {path.name}")

    if fmt == "grib":
        raw = _open_grib(path, chunks)
    else:
        raw = _open_netcdf(path, chunks)

    raw, coord_info = standardize_coords(raw)
    crs = _extract_crs(raw)
    grid_type = "projected_2d" if coord_info["two_d"] else "regular_ll"
    dt = _infer_time_step(raw)
    clean, audits = standardize_variables(raw, dt)

    return NormalizedDataset(
        ds=clean, source_path=str(path), fmt=fmt, crs=crs,
        grid_type=grid_type, audits=audits, time_step_hours=dt,
    )


def _open_netcdf(path: Path, chunks: dict | None) -> xr.Dataset:
    return xr.open_dataset(path, chunks=chunks, decode_times=True)


def _open_grib(path: Path, chunks: dict | None) -> xr.Dataset:
    """cfgrib 可能把一个文件拆成多个超立方体, 合并之。"""
    import cfgrib

    datasets = cfgrib.open_datasets(
        str(path), chunks=chunks,
        backend_kwargs={
            "indexpath": "",
            "read_keys": ["discipline", "parameterCategory", "parameterNumber"],
        },
    )
    if len(datasets) == 1:
        return datasets[0]
    # 合并: 不同变量拼到一起; 坐标冲突以后者为准
    merged = xr.merge(datasets, compat="override")
    return merged


def open_many(paths: list[str | Path], chunks: dict | None = None
              ) -> list[NormalizedDataset]:
    return [open_dataset(p, chunks=chunks) for p in paths]
