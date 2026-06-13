import netCDF4
import numpy as np
import pytest
from numpy import testing

from cloudnetpy.exceptions import ModelDataError
from cloudnetpy.model_evaluation.products.model_products import ModelManager

MODEL = "ecmwf"
PRODUCT = "iwc"


def test_get_cf(model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    obj._get_cf()
    assert "model_cf" in obj.data


def test_get_iwc(model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    obj._get_iwc()
    assert "model_iwc" in obj.data


def test_get_lwc(model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, "lwc")
    obj._get_lwc()
    assert "model_lwc" in obj.data


def test_read_config(model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    var = obj.get_model_var_names(("p",))
    assert "pressure" in var
    var = obj.get_model_var_names(("T",))
    assert "temperature" in var


@pytest.mark.parametrize("key", ["pressure", "temperature"])
def test_set_variables(key, model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    var = obj.getvar(key)
    x = netCDF4.Dataset(model_file).variables[key]
    testing.assert_almost_equal(x, var)


@pytest.mark.parametrize("p, T, q", [(1, 2, 3), (20, 40, 80), (0.3, 0.6, 0.9)])
def test_calc_water_content(p, T, q, model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    x = q * p / (287 * T)
    testing.assert_almost_equal(x, obj._calc_water_content(q, p, T))


@pytest.mark.parametrize(
    "key",
    ["time", "level", "horizontal_resolution", "latitude", "longitude"],
)
def test_add_common_variables(key, model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    assert key in obj.data


@pytest.mark.parametrize("key", ["height", "forecast_time"])
def test_add_model_variables(key, model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    assert f"model_{key}" in obj.data


def test_cut_off_extra_levels(model_file) -> None:
    # The fixture has 2 levels, both below the altitude limit, so nothing is cut.
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    data = np.array([np.arange(100), np.arange(100)])
    x = obj.cut_off_extra_levels(data)
    testing.assert_array_almost_equal(x, data[:, :2])


def test_cut_off_extra_levels_drops_high_levels(model_file_tall) -> None:
    # The tall fixture has 5 levels, of which 3 are below the altitude limit.
    obj = ModelManager(str(model_file_tall), MODEL, PRODUCT)
    assert obj._n_levels == 3
    data = np.arange(5 * 3).reshape(3, 5)
    x = obj.cut_off_extra_levels(data)
    testing.assert_array_almost_equal(x, data[:, :3])


def test_missing_horizontal_resolution_raises(model_file_no_hres) -> None:
    with pytest.raises(ModelDataError, match="horizontal_resolution"):
        ModelManager(str(model_file_no_hres), MODEL, "cf")


def test_zero_horizontal_resolution_raises(model_file_zero_hres) -> None:
    with pytest.raises(ModelDataError, match="invalid horizontal_resolution"):
        ModelManager(str(model_file_zero_hres), MODEL, "cf")


def test_missing_product_variable_raises(model_file_no_clouds) -> None:
    with pytest.raises(ModelDataError, match="cloud_fraction"):
        ModelManager(str(model_file_no_clouds), MODEL, "cf")


def test_calculate_wind_speed(model_file) -> None:
    obj = ModelManager(str(model_file), MODEL, PRODUCT)
    u = obj.getvar("uwind")
    v = obj.getvar("vwind")
    compare = np.sqrt(u.data**2 + v.data**2)  # type: ignore
    x = obj._calculate_wind_speed()
    testing.assert_array_almost_equal(x, compare)
