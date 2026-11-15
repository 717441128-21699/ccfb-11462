"""变量名 / 单位的容错识别与换算。

实际业务中 NetCDF/GRIB 的属性写法五花八门:
  * 温度单位 K / kelvin / degC / Celsius / degree_C / "deg C"
  * 降水 mm / kg m-2 s-1 / kg/m2/s / m/s / millimetre
  * 变量名 t2m / TEM_2m / 气温 / tp / RAIN / precipitation_flux
本模块统一识别为内部标准变量: tmin / tmax / precip (摄氏度 / 毫米 / 毫米每小时)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# 1. 变量名识别 (大小写/空格/常见别名/中文名)
# ---------------------------------------------------------------------------

_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "tmin": (
        "tmin", "t2m_min", "tasmin", "t_min", "mintemp", "min_2m_air_temperature",
        "minimum_temperature", "最低温", "最低气温", "日最低温", "2m_temperature_min",
    ),
    "tmax": (
        "tmax", "t2m_max", "tasmax", "t_max", "maxtemp", "max_2m_air_temperature",
        "maximum_temperature", "最高温", "最高气温", "日最高温", "2m_temperature_max",
    ),
    "temp": (  # 瞬时/平均温度, 下游可派生极值
        "t2m", "t", "tas", "temp", "temperature", "air_temperature", "2t",
        "air_temp", "x_temp", "tmp", "tair", "airt",
        "气温", "温度", "2m温度", "2m气温",
    ),
    "precip": (
        "tp", "pr", "precip", "precipitation", "rain", "rainfall", "apcp",
        "total_precipitation", "precipitation_flux", "p", "rr", "prate",
        "tprate", "cprecip", "lssfr", "lsrr", "x_rain", "rain1", "pcp",
        "降水", "降雨量", "降水量", "降雨",
    ),
}

# 反向索引: 规范化后的别名 -> 标准变量
_ALIAS_LOOKUP: dict[str, str] = {}
for std, names in _VAR_ALIASES.items():
    for n in names:
        _ALIAS_LOOKUP[re.sub(r"[\s_\-]+", "", n).lower()] = std


def normalize_name(name: str) -> str:
    """去掉空格/下划线/连字符并小写, 用于别名匹配。"""
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


def recognize_variable(name: str, attrs: dict | None = None) -> str | None:
    """返回标准变量名 tmin/tmax/temp/precip, 认不出来返回 None。

    优先按变量名匹配; 匹配失败时再看 long_name/standard_name/description 属性。
    """
    hit = _ALIAS_LOOKUP.get(normalize_name(name))
    if hit:
        return hit
    if attrs:
        for key in ("long_name", "standard_name", "description", "title", "name"):
            val = attrs.get(key)
            if val and _ALIAS_LOOKUP.get(normalize_name(val)):
                return _ALIAS_LOOKUP[normalize_name(val)]
    return None


# ---------------------------------------------------------------------------
# 2. 单位识别与换算
# ---------------------------------------------------------------------------

def _clean_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    u = str(unit).strip().lower()
    u = u.replace("°", "").replace("℃", "c").replace("degrees", "deg").replace("degree", "deg")
    u = u.replace("metres", "m").replace("meters", "m").replace("millimetres", "mm")
    u = u.replace("millimeters", "mm").replace("second", "sec").replace("hour", "hr")
    u = re.sub(r"[\s]+", "", u)
    return u


@dataclass
class UnitSpec:
    kind: str            # "temperature" | "precip_amount" | "precip_rate" | "unknown"
    scale: float = 1.0   # 线性换算 scale*x + offset
    offset: float = 0.0


def recognize_unit(unit: str | None) -> UnitSpec:
    """把不规范单位字符串归类, 给出换算到内部标准单位的线性系数。

    内部标准: 温度 degC; 降水量 mm; 降水率 mm/h。
    """
    u = _clean_unit(unit)
    if not u:
        return UnitSpec("unknown")

    # ---- 温度 ----
    if u in ("k", "kelvin", "degk", "kdeg"):
        return UnitSpec("temperature", 1.0, -273.15)
    if u in ("c", "degc", "celsius", "deg_c", "cdeg", "centigrade"):
        return UnitSpec("temperature", 1.0, 0.0)
    if u in ("f", "degf", "fahrenheit", "deg_f"):
        return UnitSpec("temperature", 5.0 / 9.0, -32.0 * 5.0 / 9.0)

    # ---- 降水量 (累计量) ----
    if u in ("mm", "millimetre", "millimeter", "mmwater", "kgm-2", "kg/m2",
             "kgm^-2", "kgm**-2"):
        # 1 kg/m^2 的水 = 1 mm 降水
        return UnitSpec("precip_amount", 1.0)
    if u in ("m", "metre", "meter"):
        return UnitSpec("precip_amount", 1000.0)
    if u in ("cm",):
        return UnitSpec("precip_amount", 10.0)
    if u in ("inch", "inches", "in"):
        return UnitSpec("precip_amount", 25.4)

    # ---- 降水率 (通量) ----
    # kg m-2 s-1 ≈ mm/s (水密度 1000 kg/m3)
    rate_patterns = (
        (r"^kgm-?2s-?1$", 3600.0),
        (r"^kg(/|\^)?m2?(\^-?2)?(/|\^)?s2?(\^-?1)?$", 3600.0),
        (r"^mm/s$", 3600.0),
        (r"^m/s$", 3_600_000.0),
        (r"^mm(/|\^)?h(r)?$", 1.0),
        (r"^mmh-?1$", 1.0),
        (r"^m(/|\^)?h(r)?$", 1_000_000.0),
        (r"^cm(/|\^)?h(r)?$", 10.0),
    )
    for pat, scale in rate_patterns:
        if re.match(pat, u):
            return UnitSpec("precip_rate", scale)

    return UnitSpec("unknown")


def convert_values(values: np.ndarray, spec: UnitSpec) -> np.ndarray:
    """按 UnitSpec 做线性换算。"""
    if spec.kind == "unknown":
        return values
    return values * spec.scale + spec.offset


# ---------------------------------------------------------------------------
# 3. 标准变量 -> 标准单位 描述 (供写出属性/审计)
# ---------------------------------------------------------------------------

CANONICAL_UNITS = {
    "tmin": "degC",
    "tmax": "degC",
    "temp": "degC",
    "precip": "mm",        # 若输入是通量, 已按时间步长积分成该步累计量
}

CANONICAL_DESC = {
    "tmin": "minimum 2m air temperature",
    "tmax": "maximum 2m air temperature",
    "temp": "instantaneous 2m air temperature",
    "precip": "accumulated precipitation over the time step",
}
