"""需求 5: 可配置灾害指标 (阈值计数 / 连续条件 / 复合表达式)。"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from agromet.indicators import (
    IndicatorSpec, evaluate_indicator, extract_events,
    _safe_eval_condition, ConditionError,
)


def _daily_ds():
    dates = pd.date_range("2023-01-01", periods=10, freq="1D")
    tmin = np.array([5, -1, -3, -2, 4, 6, -1, -1, 5, 5], dtype=float)
    precip = np.array([0, 0, 0, 5, 0, 0, 0, 0, 0, 10], dtype=float)
    return xr.Dataset({
        "tmin": (("time",), tmin), "precip": (("time",), precip),
    }, coords={"time": dates})


def test_threshold_count():
    ds = _daily_ds()
    spec = IndicatorSpec("f", "霜冻", "threshold_count", "tmin",
                         comparator="<", threshold=0.0)
    hit = evaluate_indicator(ds, spec)
    assert hit.values.tolist() == [0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    events = extract_events(
        pd.Series(hit.values, index=ds["time"].values), spec,
        pd.Series(ds["tmin"].values, index=ds["time"].values))
    assert len(events) == 2
    assert events[0].start_date == "2023-01-02" and events[0].duration_days == 3
    assert events[0].peak_value == -1.0  # 序列最大值


def test_consecutive_dry_spell_min_duration():
    ds = _daily_ds()
    spec = IndicatorSpec("d", "旱", "consecutive", "precip",
                         comparator="<", threshold=0.1, min_duration=5)
    counter = evaluate_indicator(ds, spec)
    # 1/05-1/09 连续 5 个无雨日
    assert counter.values.tolist() == [1, 2, 3, 0, 1, 2, 3, 4, 5, 0]
    events = extract_events(
        pd.Series(counter.values, index=ds["time"].values), spec,
        pd.Series(ds["precip"].values, index=ds["time"].values))
    assert len(events) == 1
    assert events[0].start_date == "2023-01-05"
    assert events[0].end_date == "2023-01-09"
    assert events[0].duration_days == 5


def test_min_duration_filters_short_spells():
    ds = _daily_ds()
    spec = IndicatorSpec("d", "旱", "consecutive", "precip",
                         comparator="<", threshold=0.1, min_duration=6)
    counter = evaluate_indicator(ds, spec)
    events = extract_events(
        pd.Series(counter.values, index=ds["time"].values), spec,
        pd.Series(ds["precip"].values, index=ds["time"].values))
    assert events == []


def test_safe_condition_and_or():
    v = {"tmin": np.array([-1.0, 3.0]), "precip": np.array([0.5, 0.0])}
    out = _safe_eval_condition("tmin < 2 and precip >= 0.1", v)
    assert out.tolist() == [True, False]
    out2 = _safe_eval_condition("tmin < 0 or precip < 0.1", v)
    assert out2.tolist() == [True, True]


def test_unsafe_expression_rejected():
    with pytest.raises(ConditionError):
        _safe_eval_condition("__import__('os').system('echo x')", {"a": np.ones(2)})
    with pytest.raises(ConditionError):
        _safe_eval_condition("a in (1,2)", {"a": np.ones(2)})


def test_aggregate_indicator():
    ds = _daily_ds()
    spec = IndicatorSpec("p", "总降水", "aggregate", "precip", agg="sum")
    val = evaluate_indicator(ds, spec)
    assert float(val) == float(ds["precip"].sum())
