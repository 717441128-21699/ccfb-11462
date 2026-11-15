"""需求 1: NetCDF / GRIB 读取与标准化 (三套异构合成源)。"""
import numpy as np
import pytest

from agromet.io import open_dataset
from agromet.synth import DROUGHT, FROST


def _source(cfg, key):
    mapping = {"A": "source_A_lambert.grib2",
               "B": "source_B_irregular.nc",
               "C": "source_C_rotated.nc"}
    return open_dataset(cfg.raw_dir / mapping[key], chunks={})


@pytest.mark.parametrize("key", ["A", "B", "C"])
def test_open_three_heterogeneous_sources(cfg, key):
    nd = _source(cfg, key)
    assert set(nd.ds.data_vars) >= {"temp", "precip"}
    assert "time" in nd.ds.coords and "lat" in nd.ds.coords and "lon" in nd.ds.coords
    # 单位标准化
    assert nd.ds["temp"].attrs["units"] == "degC"
    assert nd.ds["precip"].attrs["units"] == "mm"
    nd.ds.close()


def test_grib_lambert_is_2d_projected(cfg):
    nd = _source(cfg, "A")
    assert nd.fmt == "grib"
    assert nd.grid_type == "projected_2d"
    assert nd.crs is not None
    # Lambert 共形圆锥: 投影方法名应包含 lambert
    cf = nd.crs.to_cf()
    assert "lambert" in cf.get("grid_mapping_name", "").lower()
    assert nd.ds["lat"].ndim == 2
    nd.ds.close()


def test_irregular_chinese_nc_unit_conversion(cfg):
    nd = _source(cfg, "B")
    # 开尔文 -> 摄氏度, 合理气温区间
    t = nd.ds["temp"].values
    assert np.nanmin(t) > -60 and np.nanmax(t) < 60
    # 通量 kg m-2 s-1 已按 1 小时积分成 mm, 非负
    p = nd.ds["precip"].values
    assert np.nanmin(p) >= 0
    audits = {a.original_name: a for a in nd.audits}
    assert audits["气温"].canonical_name == "temp"
    assert audits["降雨量"].unit_kind == "precip_rate"
    nd.ds.close()


def test_missing_units_heuristic(cfg):
    nd = _source(cfg, "C")
    audits = {a.original_name: a for a in nd.audits}
    # 无单位属性, 温度按数值范围/默认假设识别, 没有报错且产出标准变量
    assert audits["X_TEMP"].canonical_name == "temp"
    assert audits["X_RAIN"].canonical_name == "precip"
    nd.ds.close()


def test_rotated_pole_crs_recovered(cfg):
    nd = _source(cfg, "C")
    assert nd.grid_type == "projected_2d"
    assert nd.crs is not None
    assert "rotated" in nd.crs.to_cf().get("grid_mapping_name", "")
    nd.ds.close()


def test_frost_values_present_in_source(cfg):
    """埋点在标准化后依然可辨: 霜冻时段 C01 区域出现零下温度。"""
    nd = _source(cfg, "B")
    import pandas as pd
    sub = nd.ds["temp"].sel(time=slice(FROST.start,
                                       pd.Timestamp(FROST.end) + pd.Timedelta(hours=23)))
    la0, la1, lo0, lo1 = FROST.bbox
    region = sub.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
    assert float(region.min()) < -2.0
    nd.ds.close()
