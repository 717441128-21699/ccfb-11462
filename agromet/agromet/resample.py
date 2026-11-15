"""时间重采样: 逐小时 (或子日) -> 逐日。

聚合规则按变量物理含义严格区分, 不可混用:
  precip : 日累计 (sum), 输入为通量时上游已积分成步长累计量
  tmin   : 日最低 (min)
  tmax   : 日最高 (max)
  temp   : 同时产出日均/日最低/日最高
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# 标准变量 -> 日尺度聚合算子
DAILY_AGG = {
    "precip": "sum",
    "tmin": "min",
    "tmax": "max",
    "temp": "mean",
}

_TEMP_LIKE = {"tmin", "tmax", "temp"}


def daily_dataset(hourly: xr.Dataset) -> xr.Dataset:
    """把标准化后的小时级 Dataset 聚合成日尺度。

    返回变量: precip(日累计), tmin(日最低), tmax(日最高),
    若源里只有瞬时 temp, 则由 temp 派生 tmin/tmax/tmean。
    同时附带每日实际参与聚合的小时数 n_hours, 供缺测审计。
    """
    if "time" not in hourly.dims:
        raise ValueError("输入数据缺少 time 维")
    out_vars = {}

    # 瞬时温度 -> 极值/均值
    if "temp" in hourly.data_vars and "tmin" not in hourly.data_vars:
        t = hourly["temp"]
        out_vars["tmin"] = t.resample(time="1D").min(skipna=True)
        out_vars["tmax"] = t.resample(time="1D").max(skipna=True)
        out_vars["tmean"] = t.resample(time="1D").mean(skipna=True)
    else:
        for name in ("tmin", "tmax"):
            if name in hourly.data_vars:
                op = DAILY_AGG[name]
                out_vars[name] = getattr(
                    hourly[name].resample(time="1D"), op)(skipna=True)
        if "temp" in hourly.data_vars:
            out_vars["tmean"] = hourly["temp"].resample(time="1D").mean(skipna=True)

    if "precip" in hourly.data_vars:
        out_vars["precip"] = hourly["precip"].resample(time="1D").sum(skipna=True)

    # 每日小时数 (审计完整日)
    ones = xr.DataArray(
        np.ones(hourly.sizes["time"], dtype="int16"),
        dims="time", coords={"time": hourly["time"].values})
    n_hours = ones.resample(time="1D").sum().rename("n_hours")

    result = xr.Dataset(out_vars)
    result["n_hours"] = n_hours
    _annotate(result)
    return result


def _annotate(ds: xr.Dataset) -> None:
    attrs = {
        "precip": ("daily accumulated precipitation", "mm"),
        "tmin": ("daily minimum 2m temperature", "degC"),
        "tmax": ("daily maximum 2m temperature", "degC"),
        "tmean": ("daily mean 2m temperature", "degC"),
    }
    for name, (desc, unit) in attrs.items():
        if name in ds.data_vars:
            ds[name].attrs.update({"long_name": desc, "units": unit,
                                   "aggregation": desc.split(" ")[0]})


def validate_daily(daily: xr.Dataset, expected_hours: int = 24) -> pd.DataFrame:
    """检查每个日值是否由完整 24 小时聚合而来, 返回缺测审计表。"""
    rows = []
    for ts, nh in zip(daily["time"].values, daily["n_hours"].values):
        rows.append({"date": pd.Timestamp(ts).date().isoformat(),
                     "n_hours": int(nh),
                     "complete": bool(nh == expected_hours)})
    return pd.DataFrame(rows)


def daily_dataarray(hourly_da: xr.DataArray, var_name: str) -> xr.DataArray:
    """单变量便捷入口。"""
    ds = hourly_da.to_dataset(name=var_name)
    return daily_dataset(ds)[
        "precip" if var_name == "precip" else
        ("tmin" if var_name == "tmin" else
         ("tmax" if var_name == "tmax" else "tmean"))]
