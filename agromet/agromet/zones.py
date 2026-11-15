"""区县/地块面积加权分区统计。

目标格网是等经纬度网格, 真实面积随纬度变化, 因此权重不是简单的
"格点计数", 而是每个目标格网单元与行政边界相交部分在等积投影
下的面积占比。跨边界格点按相交面积比例计入, 满足"跨格点按占比算"。
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union, transform
from pyproj import CRS, Transformer

from .regrid import _cell_edges, _equal_area_crs, RegridWeights


@dataclasses.dataclass
class Zone:
    zone_id: str
    name: str
    geometry: object          # shapely, EPSG:4326
    level: str = "county"     # county | parcel
    props: dict = dataclasses.field(default_factory=dict)


def load_zones(geojson_path: str | Path,
               id_field: str = "code",
               name_field: str = "name",
               level: str = "county") -> list[Zone]:
    """读 GeoJSON (FeatureCollection) 为 Zone 列表。"""
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    zones = []
    for feat in data["features"]:
        props = feat.get("properties", {}) or {}
        geom = shape(feat["geometry"])
        zones.append(Zone(
            zone_id=str(props.get(id_field, props.get("id", len(zones)))),
            name=str(props.get(name_field, props.get("name", f"zone_{len(zones)}"))),
            geometry=geom, level=level, props=props,
        ))
    return zones


def zones_to_geojson(zones: list[Zone]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"code": z.zone_id, "name": z.name,
                            "level": z.level, **z.props},
             "geometry": mapping(z.geometry)}
            for z in zones
        ],
    }


@dataclasses.dataclass
class ZonalWeights:
    """zone × target-cell 的面积权重 (行归一化)."""
    W: sp.csr_matrix
    zone_ids: list[str]
    target_lats: np.ndarray
    target_lons: np.ndarray
    cell_areas: np.ndarray          # 每个目标格网单元的真实面积 m2
    coverage: np.ndarray            # 每个区县覆盖到的格点数

    def save(self, directory: str | Path, name: str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / f"{name}.npz", self.W)
        (directory / f"{name}.json").write_text(json.dumps({
            "zone_ids": self.zone_ids,
            "target_lats": self.target_lats.tolist(),
            "target_lons": self.target_lons.tolist(),
            "cell_areas": self.cell_areas.tolist(),
            "coverage": self.coverage.tolist(),
        }, ensure_ascii=False), encoding="utf-8")
        return directory / f"{name}.npz"

    @classmethod
    def load(cls, directory: str | Path, name: str) -> "ZonalWeights":
        directory = Path(directory)
        meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        return cls(
            W=sp.load_npz(directory / f"{name}.npz"),
            zone_ids=meta["zone_ids"],
            target_lats=np.array(meta["target_lats"]),
            target_lons=np.array(meta["target_lons"]),
            cell_areas=np.array(meta["cell_areas"]),
            coverage=np.array(meta["coverage"]),
        )


def _target_cell_polygons(lats, lons, area_crs):
    lat_edges = _cell_edges(lats)
    lon_edges = _cell_edges(lons)
    tr = Transformer.from_crs(CRS.from_epsg(4326), area_crs, always_xy=True)
    polys, areas = [], []
    for j in range(lats.size):
        for i in range(lons.size):
            pll = box(lon_edges[i], lat_edges[j],
                      lon_edges[i + 1], lat_edges[j + 1])
            x, y = tr.transform(*pll.exterior.xy)
            pa = box(min(x), min(y), max(x), max(y))
            polys.append(pa)
            areas.append(pa.area)
    return polys, np.array(areas)


def build_zonal_weights(zones: list[Zone],
                        target_lats: np.ndarray,
                        target_lons: np.ndarray) -> ZonalWeights:
    """构建区县 × 目标格点 的面积占比矩阵。"""
    lon0 = float(np.mean(target_lons))
    lat0 = float(np.mean(target_lats))
    area_crs = _equal_area_crs(lon0, lat0)
    cell_polys, cell_areas = _target_cell_polygons(
        target_lats, target_lons, area_crs)
    tr = Transformer.from_crs(CRS.from_epsg(4326), area_crs, always_xy=True)

    rows, cols, vals = [], [], []
    coverage = np.zeros(len(zones), dtype=int)
    for z_idx, zone in enumerate(zones):
        geom_a = transform(tr.transform, zone.geometry)
        for c, pc in enumerate(cell_polys):
            inter = geom_a.intersection(pc).area
            if inter <= 0:
                continue
            rows.append(z_idx)
            cols.append(c)
            # 权重 = 该格点被区县覆盖的面积 / 区县内总面积
            vals.append(inter)
            coverage[z_idx] += 1
    raw = sp.csr_matrix(
        (vals, (rows, cols)), shape=(len(zones), cell_areas.size))
    row_sum = np.asarray(raw.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    W = sp.diags(1.0 / row_sum) @ raw
    return ZonalWeights(W=W.tocsr(), zone_ids=[z.zone_id for z in zones],
                        target_lats=np.asarray(target_lats),
                        target_lons=np.asarray(target_lons),
                        cell_areas=cell_areas, coverage=coverage)


def zonal_statistics(da: xr.DataArray, zw: ZonalWeights,
                     zones: list[Zone], stat: str = "mean") -> pd.DataFrame:
    """对 (time, lat, lon) 网格场做分区统计。

    stat:
      mean   - 面积加权平均 (默认, 适用于温度)
      sum    - 面积加权总量 (适用于降水体积, 输出 mm·m²)
      max/min- 区内极值
    """
    nt = da.sizes["time"]
    n_cell = zw.target_lats.size * zw.target_lons.size
    flat = da.values.reshape(nt, n_cell)
    W = zw.W
    if stat == "mean":
        vals = flat @ W.T  # NaN 处理: 用掩膜重算
        vals = _nan_weighted_mean(flat, W)
    elif stat == "sum":
        filled = np.where(np.isnan(flat), 0.0, flat)
        # 每格点 mm * 该格点落入区县的面积 => mm·m²
        vals = filled @ (W.multiply(zw.cell_areas)).T
    elif stat in ("max", "min"):
        vals = _nan_extreme(flat, W, stat)
    else:
        raise ValueError(f"不支持的统计量: {stat}")

    name_by_id = {z.zone_id: z.name for z in zones}
    frames = []
    for k, zid in enumerate(zw.zone_ids):
        frames.append(pd.DataFrame({
            "time": pd.to_datetime(da["time"].values),
            "zone_id": zid, "zone_name": name_by_id.get(zid, zid),
            "variable": da.name, "stat": stat, "value": vals[:, k],
        }))
    return pd.concat(frames, ignore_index=True)


def _nan_weighted_mean(flat: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    """NaN 安全的面积加权平均: 忽略缺测格点并对权重重新归一。"""
    nt, nc = flat.shape
    out = np.full((nt, W.shape[0]), np.nan)
    Wdok = W.todok()
    for k in range(W.shape[0]):
        cells = W.getrow(k).indices
        if cells.size == 0:
            continue
        w = W.getrow(k).data
        sub = flat[:, cells]              # nt × n
        valid = ~np.isnan(sub)
        ww = np.broadcast_to(w, sub.shape)
        wsum = np.where(valid, ww, 0.0).sum(axis=1)
        num = np.where(valid, sub * ww, 0.0).sum(axis=1)
        ok = wsum > 0
        out[ok, k] = num[ok] / wsum[ok]
    return out


def _nan_extreme(flat: np.ndarray, W: sp.csr_matrix, kind: str) -> np.ndarray:
    nt = flat.shape[0]
    out = np.full((nt, W.shape[0]), np.nan)
    for k in range(W.shape[0]):
        cells = W.getrow(k).indices
        if cells.size == 0:
            continue
        sub = flat[:, cells]
        with np.errstate(all="ignore"):
            out[:, k] = np.nanmax(sub, axis=1) if kind == "max" \
                else np.nanmin(sub, axis=1)
    return out


# ---------------------------------------------------------------------------
# 地块 (点/面) 查询与格点级贡献
# ---------------------------------------------------------------------------

def parcel_cells(geometry, target_lats, target_lons) -> list[dict]:
    """地块覆盖/邻近的目标格点及相交面积占比 (供溯源页面)。

    面要素: 按相交面积比例; 点要素: 返回包含该点 (或最近) 的格点, 占比 1。
    """
    from shapely.geometry import Point

    lon0 = float(np.mean(target_lons))
    lat0 = float(np.mean(target_lats))
    area_crs = _equal_area_crs(lon0, lat0)
    cell_polys, _ = _target_cell_polygons(target_lats, target_lons, area_crs)
    tr = Transformer.from_crs(CRS.from_epsg(4326), area_crs, always_xy=True)

    if isinstance(geometry, Point):
        # 找包含该点的格点; 落在边界上时取距离最近的唯一格点
        px, py = tr.transform(geometry.x, geometry.y)
        from shapely.geometry import Point as P
        pt = P(px, py)
        hit = [c for c, pc in enumerate(cell_polys) if pc.contains(pt)]
        if not hit:
            hit = [int(np.argmin([pc.distance(pt) for pc in cell_polys]))]
        c = hit[0]
        j, i = divmod(c, target_lons.size)
        return [{"cell_index": int(c), "ji": [int(j), int(i)],
                 "lat": float(target_lats[j]), "lon": float(target_lons[i]),
                 "area_fraction": 1.0, "intersect_area_m2": 0.0}]

    geom_a = transform(tr.transform, geometry)
    total = geom_a.area
    out = []
    inter_total = 0.0
    for c, pc in enumerate(cell_polys):
        inter = geom_a.intersection(pc).area
        if inter <= 0:
            continue
        j, i = divmod(c, target_lons.size)
        out.append({
            "cell_index": int(c), "ji": [int(j), int(i)],
            "lat": float(target_lats[j]), "lon": float(target_lons[i]),
            # 以落入格网的总面积为分母, 保证占比归一
            "_inter": float(inter),
            "intersect_area_m2": float(inter),
        })
        inter_total += inter
    for d in out:
        d["area_fraction"] = (d.pop("_inter") / inter_total
                              if inter_total > 0 else 0.0)
    out.sort(key=lambda d: d["area_fraction"], reverse=True)
    return out


def point_parcel(lon: float, lat: float, name: str = "point_parcel") -> Zone:
    from shapely.geometry import Point
    return Zone(zone_id=name, name=name, geometry=Point(lon, lat),
                level="parcel")
