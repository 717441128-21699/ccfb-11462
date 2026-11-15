"""合成测试格点场: 一年逐小时, 三套异构源, 内置已知旱情/霜冻。

三套源 (分辨率/投影/时间步/属性规范程度各不相同):
  A. GRIB2 兰伯特投影, 20km, 变量 t2m/tp (GRIB 标准 discipline/category)
  B. NetCDF 经纬网格 0.25°, 中文名/奇葩单位 (开尔文、kg m-2 s-1)
  C. NetCDF 旋转极 2D 坐标 0.2°, 变量 X_TEMP/X_RAIN, 单位属性缺失

埋入的"真值"事件同时写入 truth_manifest.json, 供测试比对。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from .config import Config

# ---------------------------------------------------------------------------
# 真值事件定义 (测试断言以此为准)
# ---------------------------------------------------------------------------

@dataclass
class TruthEvent:
    hazard: str               # drought | frost
    event_id: str
    start: str                # 日期 (含)
    end: str                  # 日期 (含)
    bbox: tuple               # [lat_min, lat_max, lon_min, lon_max]
    counties: list[str]
    detail: str


# 研究区域
LAT_MIN, LAT_MAX = 29.9, 31.1
LON_MIN, LON_MAX = 102.9, 104.1
YEAR = 2023

# 四个矩形区县, 铺满研究区 2×2
COUNTIES = [
    ("C01", "北区县", (30.5, LAT_MAX, LON_MIN, 103.5)),
    ("C02", "东北县", (30.5, LAT_MAX, 103.5, LON_MAX)),
    ("C03", "西南县", (LAT_MIN, 30.5, LON_MIN, 103.5)),
    ("C04", "东南县", (LAT_MIN, 30.5, 103.5, LON_MAX)),
]

# 旱情: 西南县 C03, 3 月 1-25 日连续 25 天无雨
DROUGHT = TruthEvent(
    hazard="drought", event_id="D1",
    start=f"{YEAR}-03-01", end=f"{YEAR}-03-25",
    bbox=(LAT_MIN, 30.5, LON_MIN, 103.5),
    counties=["C03"],
    detail="25个连续无雨日, 区域内日降水累计均 < 0.1mm",
)
# 霜冻: 北区县 C01, 1 月 10-12 日连续 3 天最低温 < -2℃
FROST = TruthEvent(
    hazard="frost", event_id="F1",
    start=f"{YEAR}-01-10", end=f"{YEAR}-01-12",
    bbox=(30.5, LAT_MAX, LON_MIN, 103.5),
    counties=["C01"],
    detail="连续3天日最低气温低于-2℃, 峰值约-5℃",
)

SEED = 20231115


def _hourly_index(year: int, months: list[int] | None = None) -> pd.DatetimeIndex:
    full = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00",
                         freq="1h", tz=None)
    if months is None:
        return full
    return full[full.month.isin(months)]


def _event_mask_hours(times: pd.DatetimeIndex, ev: TruthEvent):
    start = pd.Timestamp(ev.start)
    end = pd.Timestamp(ev.end) + pd.Timedelta(hours=23)
    return (times >= start) & (times <= end)


def _spatial_mask(lat2d, lon2d, bbox, buffer_deg: float = 0.0):
    la0, la1, lo0, lo1 = bbox
    return ((lat2d >= la0 - buffer_deg) & (lat2d <= la1 + buffer_deg)
            & (lon2d >= lo0 - buffer_deg) & (lon2d <= lo1 + buffer_deg))


def _base_fields(times: pd.DatetimeIndex, lat2d, lon2d, rng: np.random.Generator):
    """生成背景场: 温度 (年/日周期+纬度梯度) 与降水 (晴雨随机+午后对流)。"""
    nt = times.size
    doy = times.dayofyear.to_numpy()
    hour = times.hour.to_numpy()
    # 年周期: 最冷月 1 月, 最热月 7 月; 基准 19℃, 振幅 8℃
    # 取该参数使 1 月背景最低温稳定在 0℃ 以上, 仅霜冻注入窗口出现零下
    seasonal = 19.0 - 8.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
    diurnal = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)  # 午后最高
    lat_grad = -1.2 * (lat2d - 30.5)                        # 越北越冷
    temp = (seasonal[:, None, None] + diurnal[:, None, None]
            + lat_grad[None, :, :]
            + rng.normal(0, 0.6, size=(nt,) + lat2d.shape))

    # 降水: 每小时约 6% 概率有雨, 午后略多, 夏季更强
    p_rain = 0.06 + 0.05 * (hour == 15)
    rain_flag = rng.random((nt,) + lat2d.shape) < p_rain[:, None, None]
    rate = rng.gamma(shape=1.5, scale=2.5, size=(nt,) + lat2d.shape)
    season_boost = 1.0 + 1.2 * np.maximum(
        0, np.cos(2 * np.pi * (doy - 200) / 365.0))
    precip = rain_flag * rate * season_boost[:, None, None]
    return temp.astype(np.float32), precip.astype(np.float32)


def _inject_events(times, lat2d, lon2d, temp, precip, rng):
    # 旱情: 时空盒外扩缓冲, 覆盖区县边界外的粗网格格点, 避免插值把相邻
    # 雨区降水扩散进判定区县。逐小时降水 <= 0.002mm, 使日累计
    # <= ~0.05mm, 稳定低于 0.1mm 无雨阈值。
    d_t = _event_mask_hours(times, DROUGHT)
    d_s = _spatial_mask(lat2d, lon2d, DROUGHT.bbox, buffer_deg=0.3)
    precip[d_t[:, None, None] & d_s[None, :, :]] = (
        rng.uniform(0.0, 0.002, size=int((d_t[:, None, None] & d_s[None]).sum()))
        .astype(np.float32))

    # 霜冻: 三天内夜间(18时-次日09时)温度压到 -5℃ 附近, 白天仍低 (<0)。
    # 霜冻盒东界即 C01/C02 分界线, 仅向外扩纬度方向与西侧, 东侧不扩,
    # 避免低温越过 103.5 经线污染邻县 C02。
    f_t = _event_mask_hours(times, FROST)
    la0, la1, lo0, lo1 = FROST.bbox
    f_s = ((lat2d >= la0 - 0.3) & (lat2d <= la1 + 0.3)
           & (lon2d >= lo0 - 0.3) & (lon2d <= lo1))
    night = np.asarray((times.hour >= 18) | (times.hour <= 9))
    target_t = np.where(night, -5.0, -1.2)
    delta = target_t[:, None, None] - temp
    mask3d = f_t[:, None, None] & f_s[None, :, :]
    temp[mask3d] = (temp + delta * mask3d)[mask3d].astype(np.float32)
    return temp, precip


# ---------------------------------------------------------------------------
# 区县边界 GeoJSON
# ---------------------------------------------------------------------------

def write_boundaries(cfg: Config) -> Path:
    import json
    features = []
    for code, name, (la0, la1, lo0, lo1) in COUNTIES:
        features.append({
            "type": "Feature",
            "properties": {"code": code, "name": name, "level": "county"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0],
                ]],
            },
        })
    # 一个示例地块 (点缓冲的小矩形), 落在霜冻区
    parcel = {
        "type": "Feature",
        "properties": {"code": "P01", "name": "示范地块P01", "level": "parcel"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [103.05, 30.75], [103.20, 30.75], [103.20, 30.90],
            [103.05, 30.90], [103.05, 30.75]]]},
    }
    features.append(parcel)
    out = cfg.boundary_dir / "counties.geojson"
    out.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 源 A: GRIB2 兰伯特
# ---------------------------------------------------------------------------

LAMBERT_CRS = CRS.from_proj4(
    "+proj=lcc +lat_1=25 +lat_2=40 +lat_0=30 +lon_0=103.5 "
    "+R=6371200 +units=m +no_defs")


def _lambert_grid():
    tr = Transformer.from_crs(CRS.from_epsg(4326), LAMBERT_CRS, always_xy=True)
    x0, y0 = tr.transform(LON_MIN, LAT_MIN)
    xs = np.arange(x0, x0 + 20_000.0 * 14, 20_000.0)   # ~14×12
    ys = np.arange(y0, y0 + 20_000.0 * 12, 20_000.0)
    xx, yy = np.meshgrid(xs, ys)
    tr_inv = Transformer.from_crs(LAMBERT_CRS, CRS.from_epsg(4326), always_xy=True)
    lon2d, lat2d = tr_inv.transform(xx, yy)
    keep = (lat2d >= LAT_MIN) & (lat2d <= LAT_MAX) & \
           (lon2d >= LON_MIN) & (lon2d <= LON_MAX)
    # 保留完整矩形但裁掉完全离域的行列
    row_ok = keep.any(axis=1)
    col_ok = keep.any(axis=0)
    return lat2d[row_ok][:, col_ok], lon2d[row_ok][:, col_ok], xs[col_ok], ys[row_ok]


def write_grib_source(cfg: Config, times, lat2d, lon2d, temp, precip) -> Path:
    from eccodes import (codes_grib_new_from_samples, codes_set,
                          codes_set_array, codes_write, codes_release)

    out = cfg.raw_dir / "source_A_lambert.grib2"
    rng_lag = np.random.default_rng(SEED + 1)
    ny, nx = lat2d.shape
    with open(out, "wb") as fh:
        for t_idx, ts in enumerate(times):
            for kind in ("t", "p"):
                msg = codes_grib_new_from_samples("regular_ll_sfc_grib2")
                codes_set(msg, "gridDefinitionTemplateNumber", 30)
                codes_set(msg, "Nx", nx)
                codes_set(msg, "Ny", ny)
                codes_set(msg, "shapeOfTheEarth", 6)
                codes_set(msg, "scaleFactorOfRadiusOfSphericalEarth", 0)
                codes_set(msg, "scaledValueOfRadiusOfSphericalEarth", 6371200)
                codes_set(msg, "Latin1InDegrees", 25.0)
                codes_set(msg, "Latin2InDegrees", 40.0)
                codes_set(msg, "LaDInDegrees", 30.0)
                codes_set(msg, "LoVInDegrees", 103.5)
                codes_set(msg, "DxInMetres", 20000.0)
                codes_set(msg, "DyInMetres", 20000.0)
                codes_set(msg, "latitudeOfFirstGridPointInDegrees",
                          float(lat2d[0, 0]))
                codes_set(msg, "longitudeOfFirstGridPointInDegrees",
                          float(lon2d[0, 0]))
                codes_set(msg, "dataDate", int(ts.strftime("%Y%m%d")))
                codes_set(msg, "dataTime", int(ts.strftime("%H%M")))
                if kind == "t":
                    # GRIB2 discipline 0 / category 0 / number 0 = T, 编码单位 K
                    codes_set(msg, "discipline", 0)
                    codes_set(msg, "parameterCategory", 0)
                    codes_set(msg, "parameterNumber", 0)
                    values = (temp[t_idx].ravel() + 273.15).astype(np.float32)
                else:
                    # 0/1/8 = total precipitation rate (kg m-2 s-1)
                    codes_set(msg, "discipline", 0)
                    codes_set(msg, "parameterCategory", 1)
                    codes_set(msg, "parameterNumber", 8)
                    # mm/h -> kg m-2 s-1 (1 mm/h = 1/3600 kg m-2 s-1)
                    values = (precip[t_idx].ravel() / 3600.0).astype(np.float32)
                codes_set_array(msg, "values", np.ascontiguousarray(values))
                codes_write(msg, fh)
                codes_release(msg)
    return out


# ---------------------------------------------------------------------------
# 源 B: 不规范 NetCDF (中文名/开尔文/通量单位)
# ---------------------------------------------------------------------------

def write_netcdf_irregular(cfg: Config, times, rng) -> Path:
    # 0.25° 粗网格, 起点偏移半个步长使格点尽量落在区县中心而非县界上,
    # 减轻跨边界格点在分区统计中的干湿/冷暖混合
    lat = np.arange(LAT_MIN + 0.125, LAT_MAX, 0.25)
    lon = np.arange(LON_MIN + 0.125, LON_MAX, 0.25)
    lon2, lat2 = np.meshgrid(lon, lat)
    temp, precip = _base_fields(times, lat2, lon2, rng)
    temp, precip = _inject_events(times, lat2, lon2, temp, precip, rng)

    ds = xr.Dataset(
        {
            "气温": (("time", "lat", "lon"), temp + 273.15,
                    {"units": "K", "long_name": "近地面气温"}),
            "降雨量": (("time", "lat", "lon"),
                     (precip / 3600.0).astype(np.float32),
                     {"units": "kg m-2 s-1", "long_name": "降水率"}),
        },
        coords={
            "time": times,
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"source": "synthetic-irregular", "note": "属性不规范测试源"},
    )
    out = cfg.raw_dir / "source_B_irregular.nc"
    ds.to_netcdf(out, engine="netcdf4")
    return out


# ---------------------------------------------------------------------------
# 源 C: 旋转极 2D 坐标 NetCDF, 单位缺失
# ---------------------------------------------------------------------------

ROT_POLE = {
    "grid_north_pole_latitude": 35.0,
    "grid_north_pole_longitude": 15.0,
}


def write_netcdf_rotated(cfg: Config, times, rng) -> Path:
    rot_crs = CRS.from_cf({
        "grid_mapping_name": "rotated_latitude_longitude",
        **{k: float(v) for k, v in ROT_POLE.items()},
    })
    # 先把研究区中心反算到旋转坐标, 围绕中心布规则网格, 保证覆盖全域
    tr_to_rot = Transformer.from_crs(CRS.from_epsg(4326), rot_crs, always_xy=True)
    clon0, clat0 = tr_to_rot.transform(
        (LON_MIN + LON_MAX) / 2, (LAT_MIN + LAT_MAX) / 2)
    rlat = np.arange(clat0 - 7.0, clat0 + 7.01, 0.2)
    rlon = np.arange(clon0 - 7.0, clon0 + 7.01, 0.2)
    rlon2, rlat2 = np.meshgrid(rlon, rlat)
    tr = Transformer.from_crs(rot_crs, CRS.from_epsg(4326), always_xy=True)
    lon2d, lat2d = tr.transform(rlon2, rlat2)
    domain = (lat2d >= LAT_MIN - 0.05) & (lat2d <= LAT_MAX + 0.05) & \
             (lon2d >= LON_MIN - 0.05) & (lon2d <= LON_MAX + 0.05)
    row_ok, col_ok = domain.any(axis=1), domain.any(axis=0)
    rlat2, rlon2 = rlat2[row_ok][:, col_ok], rlon2[row_ok][:, col_ok]
    lat2d, lon2d = lat2d[row_ok][:, col_ok], lon2d[row_ok][:, col_ok]
    assert lat2d.size > 0, "旋转极网格未覆盖研究区"

    temp, precip = _base_fields(times, lat2d, lon2d, rng)
    temp, precip = _inject_events(times, lat2d, lon2d, temp, precip, rng)

    ds = xr.Dataset(
        {
            "X_TEMP": (("time", "y", "x"), temp),   # 无 units 属性
            "X_RAIN": (("time", "y", "x"), precip),  # 无 units 属性
        },
        coords={
            "time": times,
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
            "rlat": (("y",), rlat2[:, 0]),
            "rlon": (("x",), rlon2[0, :]),            "rotated_pole": ([], np.array(0, dtype="int32"),
                             {"grid_mapping_name": "rotated_latitude_longitude",
                              **ROT_POLE}),
        },
        attrs={"source": "synthetic-rotated"},
    )
    ds["X_TEMP"].attrs["grid_mapping"] = "rotated_pole"
    ds["X_RAIN"].attrs["grid_mapping"] = "rotated_pole"
    out = cfg.raw_dir / "source_C_rotated.nc"
    ds.to_netcdf(out, engine="netcdf4")
    return out


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------

def generate_all(cfg: Config | None = None,
                 months: list[int] | None = None) -> dict:
    """生成合成数据。months=None 为全年; 测试可传 [1, 3] 只生成 1/3 月。"""
    cfg = cfg or Config.load()
    cfg.ensure_dirs()
    rng = np.random.default_rng(SEED)
    times = _hourly_index(YEAR, months)

    boundary = write_boundaries(cfg)

    # A: Lambert GRIB
    lat2d, lon2d, _, _ = _lambert_grid()
    temp_a, precip_a = _base_fields(times, lat2d, lon2d, rng)
    temp_a, precip_a = _inject_events(times, lat2d, lon2d, temp_a, precip_a, rng)
    grib_path = write_grib_source(cfg, times, lat2d, lon2d, temp_a, precip_a)

    # B/C 各自独立随机流, 保证可复现
    nc_b = write_netcdf_irregular(cfg, times, np.random.default_rng(SEED + 2))
    nc_c = write_netcdf_rotated(cfg, times, np.random.default_rng(SEED + 3))

    manifest = {
        "year": YEAR,
        "seed": SEED,
        "months": months or list(range(1, 13)),
        "domain": {"lat_min": LAT_MIN, "lat_max": LAT_MAX,
                   "lon_min": LON_MIN, "lon_max": LON_MAX},
        "sources": [
            {"id": "A", "path": str(grib_path), "format": "grib2",
             "projection": "Lambert conformal conic", "resolution_m": 20000,
             "grid_shape": list(lat2d.shape), "step_hours": 1},
            {"id": "B", "path": str(nc_b), "format": "netcdf4",
             "projection": "regular latlon", "resolution_deg": 0.25,
             "step_hours": 1, "quirks": "中文变量名, K, kg m-2 s-1"},
            {"id": "C", "path": str(nc_c), "format": "netcdf4",
             "projection": "rotated pole 2D", "resolution_deg": 0.2,
             "step_hours": 1, "quirks": "无units属性, 2D坐标, 自定义变量名"},
        ],
        "counties": [{"code": c, "name": n, "bbox": list(b)}
                     for c, n, b in COUNTIES],
        "events": [asdict(DROUGHT), asdict(FROST)],
    }
    out = cfg.raw_dir / "truth_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return manifest
