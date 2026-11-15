"""异构网格 -> 统一经纬网格的重采样。

设计要点:
  * nearest / linear / cubic 用格点索引空间的卷积模板;
    conservative 用源/目标格网多边形相交面积做一阶守恒权重;
  * 四种方法统一产出行归一化稀疏矩阵 W (target × source),
    同一套权重既被时间分块复用, 也直接支撑格点级溯源;
  * 经纬网源在 (lat, lon) 索引空间插值; 兰伯特/旋转极等投影网
    先经 pyproj 反投影到源投影坐标, 再在 (y, x) 索引空间插值。
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import CRS, Transformer

from .config import TargetGrid
from .io import NormalizedDataset


# ---------------------------------------------------------------------------
# 源网格描述
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SourceGrid:
    kind: str                  # "regular_ll" | "projected_2d"
    lat2d: np.ndarray          # (ny, nx) 纬度
    lon2d: np.ndarray          # (ny, nx) 经度
    crs: CRS | None = None
    # 投影网格的规则 1D 投影坐标 (经纬网为 None)
    proj_x: np.ndarray | None = None
    proj_y: np.ndarray | None = None

    @property
    def shape(self):
        return self.lat2d.shape

    @property
    def n_cells(self):
        return self.lat2d.size


def describe_source_grid(nd: NormalizedDataset) -> SourceGrid:
    ds = nd.ds
    if nd.grid_type == "regular_ll":
        lat1 = ds["lat"].values.astype(float)
        lon1 = ds["lon"].values.astype(float)
        lon2, lat2 = np.meshgrid(lon1, lat1)
        return SourceGrid("regular_ll", lat2, lon2)
    # 二维投影/旋转网格
    lat2 = np.asarray(ds["lat"].values, dtype=float)
    lon2 = np.asarray(ds["lon"].values, dtype=float)
    if nd.crs is None:
        raise ValueError("二维坐标网格缺少 CRS, 无法做投影反算")
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), nd.crs, always_xy=True)
    xx, yy = transformer.transform(lon2, lat2)
    # 校验网格在源 CRS 坐标下规则等间距 (Lambert 米制 / 旋转极角度制均适用)
    dx = np.diff(xx, axis=1)
    dy = np.diff(yy, axis=0)
    if not (np.allclose(dx, dx[:, :1], rtol=1e-4)
            and np.allclose(dy, dy[:1, :], rtol=1e-4)):
        raise ValueError("投影网格在源 CRS 下不是规则网格, 暂不支持")
    return SourceGrid("projected_2d", lat2, lon2, nd.crs,
                      proj_x=xx[0, :], proj_y=yy[:, 0])


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """中心点 -> 边 (首尾外推半个间距)."""
    centers = np.asarray(centers, dtype=float)
    mids = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (centers[1] - centers[0]) / 2.0
    last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return np.concatenate([[first], mids, [last]])


# ---------------------------------------------------------------------------
# 权重容器 (可缓存/可溯源)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RegridWeights:
    method: str
    W: sp.csr_matrix                 # (n_target, n_source) 行归一化
    target_lats: np.ndarray
    target_lons: np.ndarray
    source_shape: tuple[int, int]
    source_lat2d: np.ndarray         # 溯源用
    source_lon2d: np.ndarray
    source_kind: str
    target_valid: np.ndarray         # bool, 落在源范围外的目标格为 False

    def save(self, directory: str | Path, name: str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz = directory / f"{name}.npz"
        meta = directory / f"{name}.json"
        sp.save_npz(npz, self.W)
        meta.write_text(json.dumps({
            "method": self.method,
            "target_lats": self.target_lats.tolist(),
            "target_lons": self.target_lons.tolist(),
            "source_shape": list(self.source_shape),
            "source_kind": self.source_kind,
            "target_valid": self.target_valid.tolist(),
        }, ensure_ascii=False), encoding="utf-8")
        np.savez(directory / f"{name}_src.npz",
                 lat=self.source_lat2d, lon=self.source_lon2d)
        return npz

    @classmethod
    def load(cls, directory: str | Path, name: str) -> "RegridWeights":
        directory = Path(directory)
        meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        W = sp.load_npz(directory / f"{name}.npz")
        src = np.load(directory / f"{name}_src.npz")
        return cls(
            method=meta["method"], W=W,
            target_lats=np.array(meta["target_lats"]),
            target_lons=np.array(meta["target_lons"]),
            source_shape=tuple(meta["source_shape"]),
            source_lat2d=src["lat"], source_lon2d=src["lon"],
            source_kind=meta["source_kind"],
            target_valid=np.array(meta["target_valid"], dtype=bool),
        )


# ---------------------------------------------------------------------------
# 目标点 -> 源分数索引
# ---------------------------------------------------------------------------

def _target_fractional_indices(src: SourceGrid, tgt_lats, tgt_lons):
    """返回 (fi_lat/j, fi_lon/i, valid) 三套 (M,) 数组 (源网格分数索引)."""
    tgt_lats = np.asarray(tgt_lats, dtype=float)
    tgt_lons = np.asarray(tgt_lons, dtype=float)
    if src.kind == "regular_ll":
        lat_c = src.lat2d[:, 0]
        lon_c = src.lon2d[0, :]
        fi_lat = np.interp(tgt_lats, lat_c, np.arange(lat_c.size),
                           left=np.nan, right=np.nan)
        fi_lon = np.interp(tgt_lons, lon_c, np.arange(lon_c.size),
                           left=np.nan, right=np.nan)
        valid = ~(np.isnan(fi_lat) | np.isnan(fi_lon))
        return fi_lat, fi_lon, valid

    # 投影网格: 目标经纬度 -> 源投影坐标 -> 分数索引
    tr = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)
    px, py = tr.transform(tgt_lons, tgt_lats)
    fi_x = np.interp(px, src.proj_x, np.arange(src.proj_x.size),
                     left=np.nan, right=np.nan)
    fi_y = np.interp(py, src.proj_y, np.arange(src.proj_y.size),
                     left=np.nan, right=np.nan)
    valid = ~(np.isnan(fi_x) | np.isnan(fi_y))
    return fi_y, fi_x, valid


def _stencil_weights(fi: float, n: int, kind: str):
    """一维模板: 返回 (索引数组, 权重数组)."""
    if kind == "nearest":
        i = int(round(fi))
        return np.array([i]), np.array([1.0])
    if kind == "linear":
        i0 = int(np.floor(fi))
        frac = fi - i0
        return np.array([i0, i0 + 1]), np.array([1.0 - frac, frac])
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# 权重构建
# ---------------------------------------------------------------------------

def build_weights(src: SourceGrid, target: TargetGrid,
                  method: str = "linear") -> RegridWeights:
    method = method.lower()
    if method not in ("nearest", "linear", "cubic", "conservative"):
        raise ValueError(f"不支持的插值方法: {method}")
    t_lats = target.lats
    t_lons = target.lons
    ny_t, nx_t = t_lats.size, t_lons.size
    n_tgt = ny_t * nx_t
    n_src = src.n_cells
    tg_lon, tg_lat = np.meshgrid(t_lons, t_lats)
    flat_lat = tg_lat.ravel()
    flat_lon = tg_lon.ravel()

    if method == "conservative":
        W, valid = _conservative_matrix(src, flat_lat, flat_lon, n_tgt, n_src)
    else:
        fi_j, fi_i, valid = _target_fractional_indices(
            src, flat_lat, flat_lon)
        rows, cols, vals = [], [], []
        ny_s, nx_s = src.shape
        for t in np.where(valid)[0]:
            if method == "cubic":
                idx, w = _cubic_stencil_2d(fi_j[t], fi_i[t], nx_s, ny_s)
            else:
                jj, wj = _stencil_weights(fi_j[t], ny_s, method)
                ii, wi = _stencil_weights(fi_i[t], nx_s, method)
                idx = [(j * nx_s + i) for j in jj for i in ii]
                w = [a * b for a in wj for b in wi]
            rows.extend([t] * len(idx))
            cols.extend(idx)
            vals.extend(w)
        W = sp.csr_matrix((vals, (rows, cols)), shape=(n_tgt, n_src))
        # 行归一化 (cubic 卷积和不严格为 1)
        sums = np.asarray(W.sum(axis=1)).ravel()
        sums[sums == 0] = 1.0
        W = sp.diags(1.0 / sums) @ W.tocsr()

    return RegridWeights(
        method=method, W=W.tocsr(), target_lats=t_lats, target_lons=t_lons,
        source_shape=src.shape, source_lat2d=src.lat2d,
        source_lon2d=src.lon2d, source_kind=src.kind, target_valid=valid,
    )


def _cubic_stencil_2d(fi_j, fi_i, nx_s, ny_s):
    """双三次 4×4 模板, Keys 卷积 (a=-0.5)."""
    def keys1d(fi, n):
        i0 = int(np.floor(fi))
        x = fi - i0
        xs = np.arange(-1, 3) - x  # 与 4 个锚点的距离
        a = -0.5
        w = np.zeros(4)
        for k, d in enumerate(np.abs(xs)):
            if d <= 1:
                w[k] = (a + 2) * d**3 - (a + 3) * d**2 + 1
            elif d < 2:
                w[k] = a * d**3 - 5 * a * d**2 + 8 * a * d - 4 * a
        idx = np.arange(i0 - 1, i0 + 3)
        keep = (idx >= 0) & (idx < n)
        return idx[keep], w[keep]

    jj, wj = keys1d(fi_j, ny_s)
    ii, wi = keys1d(fi_i, nx_s)
    idx, w = [], []
    for j, a in zip(jj, wj):
        for i, b in zip(ii, wi):
            idx.append(j * nx_s + i)
            w.append(a * b)
    return np.array(idx), np.array(w)


# ---------------------------------------------------------------------------
# 守恒重采样: 多边形相交面积
# ---------------------------------------------------------------------------

def _equal_area_crs(lon0: float, lat0: float) -> CRS:
    return CRS.from_proj4(
        f"+proj=aea +lat_1={lat0 - 5} +lat_2={lat0 + 5} "
        f"+lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m")


def _source_cell_polygons(src: SourceGrid, area_crs: CRS):
    """每个源格点的覆盖多边形 (经纬度), 按规则网格边构造。"""
    ny, nx = src.shape
    if src.kind == "regular_ll":
        lat_edges = _cell_edges(src.lat2d[:, 0])
        lon_edges = _cell_edges(src.lon2d[0, :])
        ll_polys = [
            box(lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
            for j in range(ny) for i in range(nx)
        ]
    else:
        # 投影/旋转规则网格
        is_geographic = src.crs.is_geographic
        dx = float(src.proj_x[1] - src.proj_x[0])
        dy = float(src.proj_y[1] - src.proj_y[0])
        x_edges = np.r_[src.proj_x - dx / 2, src.proj_x[-1] + dx / 2]
        y_edges = np.r_[src.proj_y - dy / 2, src.proj_y[-1] + dy / 2]
        if is_geographic:
            # 旋转极等角度网格: 角点反算成经纬度, 直接构造经纬度矩形
            tr = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)
            ll_polys = []
            for j in range(ny):
                for i in range(nx):
                    clon, clat = tr.transform(
                        [x_edges[i], x_edges[i + 1]],
                        [y_edges[j], y_edges[j + 1]])
                    ll_polys.append(box(min(clon), min(clat),
                                        max(clon), max(clat)))
        else:
            # 米制投影 (如 Lambert): 投影矩形角点反算经纬度
            tr = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)
            ll_polys = []
            for j in range(ny):
                for i in range(nx):
                    corners_px = [
                        (x_edges[i], y_edges[j]), (x_edges[i + 1], y_edges[j]),
                        (x_edges[i + 1], y_edges[j + 1]),
                        (x_edges[i], y_edges[j + 1]),
                    ]
                    clon, clat = tr.transform(
                        [p[0] for p in corners_px], [p[1] for p in corners_px])
                    ll_polys.append(box(min(clon), min(clat),
                                        max(clon), max(clat)))
    tr_ll_aea = Transformer.from_crs(
        CRS.from_epsg(4326), area_crs, always_xy=True)

    def to_aea(poly):
        x, y = tr_ll_aea.transform(*poly.exterior.xy)
        return box(min(x), min(y), max(x), max(y))

    return [to_aea(p) for p in ll_polys]


def _conservative_matrix(src: SourceGrid, tgt_lat, tgt_lon, n_tgt, n_src):
    lon0 = float(np.mean(tgt_lon))
    lat0 = float(np.mean(tgt_lat))
    area_crs = _equal_area_crs(lon0, lat0)
    tr_ll_aea = Transformer.from_crs(
        CRS.from_epsg(4326), area_crs, always_xy=True)

    src_polys = _source_cell_polygons(src, area_crs)
    t_lat_edges = _cell_edges(np.unique(tgt_lat))
    t_lon_edges = _cell_edges(np.unique(tgt_lon))
    tgt_polys_ll = [
        box(t_lon_edges[i], t_lat_edges[j],
            t_lon_edges[i + 1], t_lat_edges[j + 1])
        for j in range(np.unique(tgt_lat).size)
        for i in range(np.unique(tgt_lon).size)
    ]
    rows, cols, vals = [], [], []
    valid = np.zeros(n_tgt, dtype=bool)
    src_union = unary_union(src_polys)
    for t, pll in enumerate(tgt_polys_ll):
        x, y = tr_ll_aea.transform(*pll.exterior.xy)
        pa = box(min(x), min(y), max(x), max(y))
        if not pa.intersects(src_union):
            continue
        valid[t] = True
        total = 0.0
        entries = []
        for s, ps in enumerate(src_polys):
            inter = pa.intersection(ps).area
            if inter > 0:
                entries.append((s, inter))
                total += inter
        if total > 0:
            for s, inter in entries:
                rows.append(t)
                cols.append(s)
                vals.append(inter / total)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n_tgt, n_src))
    return W, valid


# ---------------------------------------------------------------------------
# 应用权重
# ---------------------------------------------------------------------------

def apply_weights(values: np.ndarray, weights: RegridWeights,
                  method: str | None = None) -> np.ndarray:
    """对 (time, n_source) 或 (n_source,) 数组做重采样。

    四种方法全部走同一套行归一化稀疏矩阵 W, 保证溯源时展示的
    "源格点 + 权重" 能严格线性还原输出值。
    """
    method = method or weights.method
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    out = values @ weights.W.T
    out[:, ~weights.target_valid] = np.nan
    return out[0] if squeeze else out


def regrid_dataarray(da: xr.DataArray, weights: RegridWeights,
                     var_name: str) -> xr.DataArray:
    """xarray 入口: 输入 (time, y, x)/(time, lat, lon), 输出目标经纬网格。"""
    import xarray as xr

    ny_t = weights.target_lats.size
    nx_t = weights.target_lons.size
    if da.ndim == 2:
        da = da.expand_dims("time")
    nt = da.sizes["time"]
    flat = da.values.reshape(nt, -1)
    out = apply_weights(flat, weights)
    out = out.reshape(nt, ny_t, nx_t)
    result = xr.DataArray(
        out, dims=("time", "lat", "lon"),
        coords={"time": da["time"].values,
                "lat": weights.target_lats, "lon": weights.target_lons},
        name=var_name, attrs=dict(da.attrs),
    )
    result.attrs["regrid_method"] = weights.method
    return result


# ---------------------------------------------------------------------------
# 溯源
# ---------------------------------------------------------------------------

def target_cell_index(weights: RegridWeights, lat: float, lon: float) -> int:
    """距给定经纬度最近的目标格点序号。"""
    tg_lon, tg_lat = np.meshgrid(weights.target_lons, weights.target_lons)
    d2 = (tg_lat.ravel() - lat) ** 2 + (tg_lon.ravel() - lon) ** 2
    return int(np.argmin(d2))


def source_contributors(weights: RegridWeights, target_idx: int) -> list[dict]:
    """某目标格点用到的全部源格点: 序号/经纬度/权重。"""
    row = weights.W.getrow(target_idx)
    cols = row.indices
    w = row.data
    ny_s, nx_s = weights.source_shape
    out = []
    for c, wv in zip(cols, w):
        j, i = divmod(int(c), nx_s)
        out.append({
            "source_flat_index": int(c),
            "source_ji": [int(j), int(i)],
            "lat": float(weights.source_lat2d[j, i]),
            "lon": float(weights.source_lon2d[j, i]),
            "weight": float(wv),
        })
    if weights.method == "cubic" and not out:
        # cubic 矩阵可能未持久化全部锚点时的兜底 (正常构建时已写入)
        pass
    return out
