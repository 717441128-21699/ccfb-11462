"""需求 1: 变量名/单位的容错识别。"""
import numpy as np

from agromet.units import (
    recognize_variable, recognize_unit, convert_values, normalize_name,
)


def test_variable_aliases_multilingual():
    assert recognize_variable("T2M") == "temp"
    assert recognize_variable("气温") == "temp"
    assert recognize_variable("tmin") == "tmin"
    assert recognize_variable("最低气温") == "tmin"
    assert recognize_variable("APCP") == "precip"
    assert recognize_variable("降雨量") == "precip"
    assert recognize_variable("unknown_xyz") is None


def test_variable_from_attrs():
    assert recognize_variable("X_TEMP", {"long_name": "air_temperature"}) == "temp"
    assert recognize_variable("v1", {"description": "降水量"}) == "precip"


def test_temperature_units():
    assert recognize_unit("K").kind == "temperature"
    assert recognize_unit("kelvin").offset == -273.15
    assert recognize_unit("deg C").kind == "temperature"
    assert recognize_unit("℃").kind == "temperature"
    f = recognize_unit("degF")
    np.testing.assert_allclose(convert_values(np.array([32.0]), f), [0.0], atol=1e-9)
    np.testing.assert_allclose(
        convert_values(np.array([273.15]), recognize_unit("K")), [0.0], atol=1e-9)


def test_precip_units_and_flux():
    assert recognize_unit("mm").kind == "precip_amount"
    assert recognize_unit("kg m-2").kind == "precip_amount"
    rate = recognize_unit("kg m-2 s-1")
    assert rate.kind == "precip_rate"
    # 1 kg/m2/s = 3600 mm/h
    np.testing.assert_allclose(
        convert_values(np.array([1.0]), rate), [3600.0], atol=1e-6)
    assert recognize_unit("kg/m2/s").kind == "precip_rate"
    assert recognize_unit("mm/h").kind == "precip_rate"


def test_unknown_unit():
    assert recognize_unit(None).kind == "unknown"
    assert recognize_unit("banana").kind == "unknown"


def test_normalize_name():
    assert normalize_name(" Air Temp-2 ") == "airtemp2"
