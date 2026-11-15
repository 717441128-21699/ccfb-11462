"""可配置的农业气象灾害指标。

指标定义放在 config/indicators.yaml, 三类内置引擎覆盖常见农险判据,
公式参数全部可配, 不直接 eval 任意代码:

1. threshold_count  阈值计数
     例: 最低温 < 0℃ 的天数 (霜冻日数)
2. consecutive      连续条件 (干旱核心: 连续无雨日)
     例: 连续 precip < 0.1mm 且持续 >= 15 天
3. aggregate        区间聚合
     例: 某时段降水总量 / 平均温度

条件表达式支持: > >= < <= == != 以及 and / or 组合,
变量只能引用数据集中已存在的标准变量。
"""
from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

_ALLOWED_BINOPS = (ast.And, ast.Or)
_ALLOWED_CMP = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)


class ConditionError(ValueError):
    pass


def _safe_eval_condition(expr: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """对布尔条件表达式做白名单 AST 求值。

    variables: 变量名 -> 同形状 ndarray (或标量)。
    """
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, _ALLOWED_BINOPS):
            vals = [ev(v) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (np.logical_and(out, v)
                       if isinstance(node.op, ast.And)
                       else np.logical_or(out, v))
            return out
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ~ev(node.operand)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and isinstance(node.ops[0], _ALLOWED_CMP):
            left = ev(node.left)
            right = ev(node.comparators[0])
            op = node.ops[0]
            table = {
                ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
            }
            return table[type(op)](left, right)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ConditionError(
                    f"条件引用了未知变量 {node.id!r}; 可用: {sorted(variables)}")
            return variables[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):  # 允许简单算术 (如 precip * 1.0)
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
        raise ConditionError(f"条件中存在不允许的语法节点: {ast.dump(node)}")

    return ev(tree)


@dataclass
class IndicatorSpec:
    indicator_id: str
    name: str
    type: str                         # threshold_count | consecutive | aggregate
    variable: str
    condition: str | None = None      # 布尔条件
    threshold: float | None = None
    comparator: str | None = None     # >, >=, <, <=, ==, !=
    min_duration: int = 1             # consecutive 用
    agg: str | None = None            # sum/mean/min/max (aggregate 用)
    unit: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "IndicatorSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_indicators(path: str | Path) -> list[IndicatorSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [IndicatorSpec.from_dict(d) for d in raw["indicators"]]


def _cmp_mask(spec: IndicatorSpec, da: xr.DataArray,
              all_vars: dict | None = None) -> np.ndarray:
    if spec.condition:
        # 复合条件可引用数据集中任意标准变量
        variables = {spec.variable: da.values}
        if all_vars:
            variables.update(all_vars)
        return _safe_eval_condition(spec.condition, variables)
    table = {
        ">": lambda a, t: a > t, ">=": lambda a, t: a >= t,
        "<": lambda a, t: a < t, "<=": lambda a, t: a <= t,
        "==": lambda a, t: a == t, "!=": lambda a, t: a != t,
    }
    if spec.comparator not in table:
        raise ConditionError(f"指标 {spec.indicator_id} 缺少合法比较符")
    return table[spec.comparator](da.values, spec.threshold)


# ---------------------------------------------------------------------------
# 事件提取
# ---------------------------------------------------------------------------

@dataclass
class HazardEvent:
    indicator_id: str
    name: str
    start_date: str
    end_date: str
    duration_days: int
    peak_value: float
    mean_value: float
    total_value: float | None = None


def _runs(mask: np.ndarray):
    """布尔序列 -> (start, end_exclusive) 连续 True 段。"""
    if mask.size == 0:
        return []
    padded = np.r_[False, mask.astype(bool), False]
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def evaluate_indicator(daily: xr.Dataset, spec: IndicatorSpec,
                       dim: str = "time") -> xr.DataArray:
    """计算指标的逐格点/逐区逐日序列。

    threshold_count -> 0/1 命中序列 (可再按时间累计)
    consecutive     -> 连续命中天数计数器 (每天为截至当日的连续长度)
    aggregate       -> 直接返回聚合变量本身
    """
    if spec.variable not in daily.data_vars:
        raise KeyError(f"日数据中没有变量 {spec.variable!r}")
    da = daily[spec.variable]

    if spec.type == "aggregate":
        op = spec.agg or "mean"
        return getattr(da, op)(dim=dim, skipna=True)

    all_vars = {name: daily[name].values for name in daily.data_vars}
    mask = _cmp_mask(spec, da, all_vars)
    if spec.type == "threshold_count":
        return xr.DataArray(mask.astype("int8"), dims=da.dims, coords=da.coords,
                            name=spec.indicator_id)
    if spec.type == "consecutive":
        counter = np.zeros(mask.shape, dtype="int16")
        # 支持 (time, lat, lon) 与 (time,) 两种形状
        for t in range(mask.shape[0]):
            if t == 0:
                counter[t] = mask[t].astype("int16")
            else:
                counter[t] = np.where(mask[t], counter[t - 1] + 1, 0)
        return xr.DataArray(counter, dims=da.dims, coords=da.coords,
                            name=spec.indicator_id)
    raise ValueError(f"未知指标类型: {spec.type}")


def extract_events(series: pd.Series, spec: IndicatorSpec,
                   value_series: pd.Series | None = None) -> list[HazardEvent]:
    """对一维 (按日期索引的) 命中序列提取灾害事件。

    series: threshold_count 时为 0/1; consecutive 时为连续计数器。
    value_series: 对应的原始物理量序列, 用于峰值/均值统计。
    """
    values = series.values
    if spec.type == "consecutive":
        mask = values >= max(spec.min_duration, 1)
        # 把计数器还原成命中布尔, 再取连续段
        hit = values > 0
    else:
        hit = values.astype(bool)
        mask = hit
    events = []
    for s, e in _runs(hit):
        duration = e - s
        if spec.type == "consecutive" and duration < spec.min_duration:
            continue
        idx = series.index[s:e]
        phys = value_series.loc[idx] if value_series is not None else series.loc[idx]
        ev = HazardEvent(
            indicator_id=spec.indicator_id, name=spec.name,
            start_date=pd.Timestamp(idx[0]).date().isoformat(),
            end_date=pd.Timestamp(idx[-1]).date().isoformat(),
            duration_days=int(duration),
            peak_value=float(np.nanmax(phys.values)),
            mean_value=float(np.nanmean(phys.values)),
            total_value=(float(np.nansum(phys.values))
                         if spec.variable == "precip" else None),
        )
        events.append(ev)
    return events


def evaluate_all_grid(daily: xr.Dataset, specs: list[IndicatorSpec]
                      ) -> dict[str, xr.DataArray]:
    """网格尺度逐指标计算 (用于画图/范围判定)。"""
    return {s.indicator_id: evaluate_indicator(daily, s) for s in specs}
