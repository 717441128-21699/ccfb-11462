"""需求 3: 面积加权分区统计, 跨格点按占比。"""
import numpy as np
import pytest
import xarray as xr

from agromet.zones import (
    load_zones, build_zonal_weights, zonal_statistics, parcel_cells,
)


def test_zonal_weights_rows_sum_to_one(cfg):
    zones = [z for z in load_zones(cfg.boundary_dir / "counties.geojson")
             if z.level == "county"]
    zw = build_zonal_weights(zones, cfg.target_grid.lats, cfg.target_grid.lons)
    sums = np.asarray(zw.W.sum(axis=1)).ravel()
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    # 每个区县都覆盖到格点
    assert (zw.coverage > 0).all()


def test_constant_field_recovers_constant(cfg):
    """均匀场 (任意位置都=10) 的面积加权均值必须等于 10。"""
    zones = [z for z in load_zones(cfg.boundary_dir / "counties.geojson")
             if z.level == "county"]
    zw = build_zonal_weights(zones, cfg.target_grid.lats, cfg.target_grid.lons)
    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    field = xr.DataArray(
        np.full((1, cfg.target_grid.lats.size, cfg.target_grid.lons.size), 10.0),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": cfg.target_grid.lats, "lon": cfg.target_grid.lons},
        name="tmean")
    df = zonal_statistics(field, zw, zones, stat="mean")
    np.testing.assert_allclose(df["value"].values, 10.0, atol=1e-9)


def test_cross_grid_fraction_split(cfg):
    """跨边界格点: 一个仅覆盖两个区县各一半的条带场, 两区统计对称。"""
    zones = [z for z in load_zones(cfg.boundary_dir / "counties.geojson")
             if z.level == "county"]
    zw = build_zonal_weights(zones, cfg.target_grid.lats, cfg.target_grid.lons)
    times = np.array(["2023-01-01"], dtype="datetime64[ns]")
    data = np.zeros((1, cfg.target_grid.lats.size, cfg.target_grid.lons.size))
    # 取一条横跨东西两县的同一纬度带, 赋常值, 东西两县均值应相等
    mid_j = cfg.target_grid.lats.size // 2
    data[0, mid_j, :] = 5.0
    field = xr.DataArray(data, dims=("time", "lat", "lon"),
                         coords={"time": times, "lat": cfg.target_grid.lats,
                                 "lon": cfg.target_grid.lons}, name="tmean")
    df = zonal_statistics(field, zw, zones, stat="mean").set_index("zone_id")
    north = df.loc[["C01", "C02"], "value"]
    assert north["C01"] == pytest.approx(north["C02"], abs=1e-9)


def test_point_parcel_finds_cell(cfg):
    from shapely.geometry import Point
    cells = parcel_cells(Point(103.2, 30.7), cfg.target_grid.lats, cfg.target_grid.lons)
    assert len(cells) == 1
    assert cells[0]["area_fraction"] == 1.0


def test_polygon_parcel_multiple_cells(cfg):
    from shapely.geometry import box
    geom = box(103.0, 30.6, 103.4, 30.9)
    cells = parcel_cells(geom, cfg.target_grid.lats, cfg.target_grid.lons)
    assert len(cells) > 1
    np.testing.assert_allclose(sum(c["area_fraction"] for c in cells), 1.0, atol=1e-9)
