"""需求 4: 逐小时 -> 逐日, 降水累加 / 温度极值, 规则不混。"""
import numpy as np
import pandas as pd
import xarray as xr

from agromet.resample import daily_dataset, validate_daily


def _toy_hourly():
    times = pd.date_range("2023-01-10", periods=48, freq="1h")
    lat = np.array([30.0, 30.1]); lon = np.array([103.0, 103.1])
    # 温度: 第一天 0..23 线性, 第二天固定
    temp = np.zeros((48, 2, 2))
    temp[:24] = np.linspace(-2, 8, 24)[:, None, None]
    temp[24:] = np.linspace(0, 10, 24)[:, None, None]
    # 降水: 每小时 1mm, 第一天应累计 24mm
    precip = np.ones((48, 2, 2))
    ds = xr.Dataset({
        "temp": (("time", "lat", "lon"), temp, {"units": "degC"}),
        "precip": (("time", "lat", "lon"), precip, {"units": "mm"}),
    }, coords={"time": times, "lat": lat, "lon": lon})
    return ds


def test_daily_aggregation_rules():
    daily = daily_dataset(_toy_hourly())
    # 降水累加: 24 个 1mm -> 24mm
    np.testing.assert_allclose(daily["precip"].values, 24.0, atol=1e-9)
    # 温度极值
    assert float(daily["tmin"].isel(time=0).min()) == -2.0
    assert float(daily["tmax"].isel(time=0).max()) == 8.0
    assert float(daily["tmin"].isel(time=1).min()) == 0.0
    assert float(daily["tmax"].isel(time=1).max()) == 10.0
    # 均值介于极值之间
    assert float(daily["tmean"].isel(time=0).values.mean()) == 3.0


def test_daily_n_hours_audit():
    daily = daily_dataset(_toy_hourly())
    audit = validate_daily(daily)
    assert audit["complete"].all()
    assert list(daily["n_hours"].values) == [24, 24]


def test_partial_day_flagged_incomplete():
    ds = _toy_hourly().isel(time=slice(0, 30))  # 第二天只有 6 小时
    daily = daily_dataset(ds)
    audit = validate_daily(daily)
    assert audit["complete"].tolist() == [True, False]
    assert audit["n_hours"].tolist() == [24, 6]


def test_precip_sum_not_average():
    """显式防回归: 日降水不能被错误地取均值。"""
    times = pd.date_range("2023-01-10", periods=24, freq="1h")
    ds = xr.Dataset({
        "precip": (("time",), np.full(24, 2.0), {"units": "mm"}),
    }, coords={"time": times})
    daily = daily_dataset(ds)
    assert float(daily["precip"].values[0]) == 48.0
