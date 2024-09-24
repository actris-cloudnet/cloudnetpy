import numpy as np
import pytest
from numpy import ma, testing

from cloudnetpy.model_evaluation.products.grid_methods import ProductGrid
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager

MODEL = "ecmwf"
OUTPUT_FILE = ""
PRODUCT = "iwc"


@pytest.mark.parametrize(
    "product, variables",
    [
        (
            "iwc",
            (
                "ecmwf_iwc",
                "iwc_ecmwf",
                "iwc_att_ecmwf",
                "iwc_rain_ecmwf",
                "iwc_adv_ecmwf",
                "iwc_att_adv_ecmwf",
                "iwc_rain_adv_ecmwf",
            ),
        ),
        (
            "cf",
            (
                "ecmwf_cf",
                "cf_A_ecmwf",
                "cf_V_ecmwf",
                "cf_A_adv_ecmwf",
                "cf_V_adv_ecmwf",
            ),
        ),
        ("lwc", ("lwc_ecmwf", "lwc_ecmwf", "lwc_adv_ecmwf")),
    ],
)
def test_generate_regrid_product(model_file, obs_file, product, variables) -> None:
    obs = ObservationManager(product, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, product)
    ProductGrid(model, obs)
    for var in variables:
        assert var in model.data


@pytest.mark.parametrize("key, value", [("iwc", 3), ("lwc", 1), ("cf", 2)])
def test_get_method_storage(key, value, model_file, obs_file) -> None:
    obs = ObservationManager(key, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, key)
    obj = ProductGrid(model, obs)
    x, y = obj._get_method_storage()
    assert len(x.keys()) == value


@pytest.mark.parametrize("key, value", [("iwc", 3), ("lwc", 1), ("cf", 2)])
def test_get_method_storage_adv(key, value, model_file, obs_file) -> None:
    obs = ObservationManager(key, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, key)
    obj = ProductGrid(model, obs)
    x, y = obj._get_method_storage()
    assert len(y.keys()) == value


@pytest.mark.parametrize("name", ["cf_V", "cf_A"])
def test_cf_method_storage(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    x, y = obj._cf_method_storage()
    assert name in x


@pytest.mark.parametrize("name", ["cf_V_adv", "cf_A_adv"])
def test_cf_method_storage_adv(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    x, y = obj._cf_method_storage()
    assert name in y


@pytest.mark.parametrize("name", ["iwc", "iwc_att", "iwc_rain"])
def test_iwc_method_storage(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    x, y = obj._iwc_method_storage()
    assert name in x


@pytest.mark.parametrize("name", ["iwc_adv", "iwc_att_adv", "iwc_rain_adv"])
def test_iwc_method_storage_adv(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    x, y = obj._iwc_method_storage()
    assert name in y


def test_product_method_storage(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    x, y = obj._product_method_storage()
    assert "lwc" in x


def test_product_method_storage_adv(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    _, y = obj._product_method_storage()
    assert "lwc_adv" in y


def test_regrid_cf_area(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    assert x[0, 0] == 0.75


def test_regrid_cf_area_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[1, :] = ma.masked
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    assert round(x[0, 0], 3) == 0.667


def test_regrid_cf_area_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[:, :] = ma.masked
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    testing.assert_equal(x, ma.masked)


def test_regrid_cf_area_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 99, 1], [0, 1, 1], [99, 0, 1], [0, 0, 0]])
    data = ma.masked_where(data == 99, data)
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    assert x[0, 0] == 0.75


def test_regrid_cf_area_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        mask=True,
    )
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    testing.assert_equal(x, ma.masked)


def test_regrid_cf_volume(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    assert x[0, 0] == 0.5


def test_regrid_cf_volume_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 99, 1], [0, 1, 1], [99, 0, 1], [0, 0, 0]])
    data = ma.masked_where(data == 99, data)
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    assert x[0, 0] == 0.5


def test_regrid_cf_volume_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array(
        [
            [99, 99, 99],
            [99, 99, 99],
            [99, 99, 99],
            [99, 99, 99],
        ],
    )
    data = ma.masked_where(data == 99, data)
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    testing.assert_equal(x, ma.masked)


def test_regrid_cf_volume_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[1, :] = ma.masked
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    assert round(x[0, 0], 3) == 0.444


def test_regrid_cf_volume_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[:, :] = ma.masked
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    testing.assert_equal(x, ma.masked)


def test_reshape_data_to_window(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    xnd = np.array([1, 1, 1, 0, 0, 0])
    ynd = np.array([1, 1, 0, 0])
    ind = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    obj._obs_data = np.array(
        [
            [1, 2, 3, 4],
            [11, 22, 33, 44],
            [111, 222, 333, 444],
            [5, 6, 7, 8],
            [55, 66, 77, 88],
            [555, 666, 777, 888],
        ],
    )
    x = obj._reshape_data_to_window(ind, xnd, ynd)
    compare = np.array([[1, 2], [11, 22], [111, 222]])
    assert x is not None
    testing.assert_array_almost_equal(x, compare)


def test_reshape_data_to_window_middle(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    xnd = np.array([0, 0, 1, 1, 1, 0])
    ynd = np.array([0, 1, 1, 0])
    ind = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    obj._obs_data = np.array(
        [
            [1, 2, 3, 4],
            [11, 22, 33, 44],
            [111, 222, 333, 444],
            [5, 6, 7, 8],
            [55, 66, 77, 88],
            [555, 666, 777, 888],
        ],
    )
    x = obj._reshape_data_to_window(ind, xnd, ynd)
    compare = np.array([[222, 333], [6, 7], [66, 77]])
    assert x is not None
    testing.assert_array_almost_equal(x, compare)


def test_reshape_data_to_window_empty(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    xnd = np.array(
        [
            1,
            1,
            1,
            0,
            0,
            0,
        ],
    )
    ynd = np.array([0, 0, 0, 0])
    ind = np.array([1, 1, 0, 0], dtype=bool)
    x = obj._reshape_data_to_window(ind, xnd, ynd)
    assert x is None


def test_regrid_iwc(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 3]])
    d = {"iwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc"]
    testing.assert_almost_equal(x[0, 0], 1.4)


def test_regrid_iwc_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array(
        [[1, 1, 99, 1], [2, 99, 2, 2], [3, 3, 3, 3], [4, 4, 4, 99]],
    )
    obj._obs_data = ma.masked_where(obj._obs_data == 99, obj._obs_data)
    d = {"iwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc"]
    testing.assert_almost_equal(x[0, 0], 1.5)


def test_regrid_iwc_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array(
        [
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
        ],
    )
    obj._obs_data = ma.masked_where(obj._obs_data == 99, obj._obs_data)
    d = {"iwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    assert d["iwc"][0, 0].mask == True


def test_regrid_iwc_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    obj._obs_data[1, :] = ma.masked
    d = {"iwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc"]
    testing.assert_almost_equal(x[0, 0], 1.0)


def test_regrid_iwc_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array(
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], mask=True
    )
    d = {"iwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    assert d["iwc"][0, 0].mask == True


def test_regrid_iwc_none(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    d = {"iwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    assert d["iwc"][0, 0].mask == True


def test_regrid_iwc_att(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    d = {"iwc_att": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_att"]
    testing.assert_almost_equal(x[0, 0], 0.018)


def test_regrid_iwc_att_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_obj.data["iwc_att"][:].mask = ma.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    )
    d = {"iwc_att": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_att"]
    testing.assert_almost_equal(x[0, 0], 0.018)


def test_regrid_iwc_att_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_obj.data["iwc_att"][:].mask = ma.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    d = {"iwc_att": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_att"]
    # Simo: not sure if this should be masked or not
    assert x[0, 0] == 0.01


def test_regrid_iwc_att_none(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    d = {"iwc_att": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_att"]
    assert x[0, 0].mask == True


def test_regrid_iwc_rain(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 3]])
    d = {"iwc_rain": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_rain"]
    testing.assert_almost_equal(x[0, 0], 2.3)


def test_regrid_iwc_rain_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array(
        [
            [1, 99, 1, 1],
            [2, 2, 2, 99],
            [3, 3, 3, 3],
            [99, 4, 4, 99],
        ],
    )
    obj._obs_data = ma.masked_where(obj._obs_data == 99, obj._obs_data)
    d = {"iwc_rain": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_rain"]
    testing.assert_almost_equal(round(x[0, 0], 3), 2.429)


def test_regrid_iwc_rain_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array(
        [
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
        ],
    )
    obj._obs_data = ma.masked_where(obj._obs_data == 99, obj._obs_data)
    d = {"iwc_rain": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_rain"]
    assert x[0, 0].mask == True


def test_regrid_iwc_rain_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    obj._obs_data[2, :] = ma.masked
    d = {"iwc_rain": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_rain"]
    testing.assert_almost_equal(round(x[0, 0], 3), 2.143)


def test_regrid_iwc_rain_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 3, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    obj._obs_data[:, :] = ma.masked
    d = {"iwc_rain": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=bool)
    no_rain = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]],
        dtype=bool,
    )
    d = obj._regrid_iwc(d, 0, 0, ind, no_rain)
    x = d["iwc_rain"]
    assert x[0, 0].mask == True


def test_regrid_product(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    d = {"lwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    d = obj._regrid_product(d, 0, 0, ind)
    x = d["lwc"]
    testing.assert_almost_equal(x[0, 0], 1.4)


def test_regrid_product_nan(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array(
        [
            [1, 99, 1, 1],
            [99, 1, 2, 2],
            [3, 3, 99, 3],
            [4, 99, 4, 4],
        ],
    )
    obj._obs_data = ma.masked_where(obj._obs_data == 99, obj._obs_data)
    d = {"lwc": ma.zeros((1, 1))}
    ind = ma.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    d = obj._regrid_product(d, 0, 0, ind)
    x = d["lwc"]
    testing.assert_almost_equal(x[0, 0], 1.5)


def test_regrid_product_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    data = ma.array(
        [
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
        ],
    )
    obj._obs_data = ma.masked_where(data == 99, data)
    d = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    d = obj._regrid_product(d, 0, 0, ind)
    assert d["lwc"][0, 0].mask == True


def test_regrid_product_masked(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    obj._obs_data[2, :] = ma.masked
    d = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    d = obj._regrid_product(d, 0, 0, ind)
    x = d["lwc"]
    testing.assert_almost_equal(x[0, 0], 1.4)


def test_regrid_product_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    obj._obs_data[:, :] = ma.masked
    d = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    d = obj._regrid_product(d, 0, 0, ind)
    x = d["lwc"]
    testing.assert_almost_equal(x, ma.masked)


def test_regrid_product_none(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    d = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    d = obj._regrid_product(d, 0, 0, ind)
    x = d["lwc"]
    assert x[0, 0].mask == True


@pytest.mark.parametrize("product", ["cf_A", "cf_V", "cf_A_adv", "cf_V_adv"])
def test_append_data2object_cf(product, model_file, obs_file) -> None:
    obs = ObservationManager("cf", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "cf")
    ProductGrid(model, obs)
    assert product + "_" + MODEL in model.data


@pytest.mark.parametrize(
    "product",
    ["iwc", "iwc_att", "iwc_rain", "iwc_adv", "iwc_att_adv", "iwc_rain_adv"],
)
def test_append_data2object_iwc(product, model_file, obs_file) -> None:
    obs = ObservationManager("iwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "iwc")
    ProductGrid(model, obs)
    assert product + "_" + MODEL in model.data


@pytest.mark.parametrize("product", ["lwc", "lwc_adv"])
def test_append_data2object_lwc(product, model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, "lwc")
    ProductGrid(model, obs)
    assert product + "_" + MODEL in model.data
