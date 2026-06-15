import numpy as np
import pytest
from numpy import ma, testing

from cloudnetpy.model_evaluation.products.grid_methods import ProductGrid
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager

MODEL = "ecmwf"
PRODUCT = "iwc"


@pytest.mark.parametrize(
    "product, variables",
    [
        (
            "iwc",
            (
                "model_iwc",
                "iwc",
                "iwc_adv",
            ),
        ),
        (
            "cf",
            (
                "model_cf",
                "cf_A",
                "cf_V",
                "cf_A_adv",
                "cf_V_adv",
            ),
        ),
        ("lwc", ("model_lwc", "lwc", "lwc_adv")),
    ],
)
def test_generate_regrid_product(model_file, obs_file, product, variables) -> None:
    obs = ObservationManager(product, str(obs_file))
    model = ModelManager(str(model_file), MODEL, product)
    ProductGrid(model, obs)
    for var in variables:
        assert var in model.data


@pytest.mark.parametrize("key, value", [("iwc", 1), ("lwc", 1), ("cf", 2)])
def test_get_method_storage(key, value, model_file, obs_file) -> None:
    obs = ObservationManager(key, str(obs_file))
    model = ModelManager(str(model_file), MODEL, key)
    obj = ProductGrid(model, obs)
    x, y = obj._get_method_storage()
    assert len(x.keys()) == value


@pytest.mark.parametrize("key, value", [("iwc", 1), ("lwc", 1), ("cf", 2)])
def test_get_method_storage_adv(key, value, model_file, obs_file) -> None:
    obs = ObservationManager(key, str(obs_file))
    model = ModelManager(str(model_file), MODEL, key)
    obj = ProductGrid(model, obs)
    x, y = obj._get_method_storage()
    assert len(y.keys()) == value


@pytest.mark.parametrize("name", ["cf_V", "cf_A"])
def test_cf_method_storage(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    x, y = obj._cf_method_storage()
    assert name in x


@pytest.mark.parametrize("name", ["cf_V_adv", "cf_A_adv"])
def test_cf_method_storage_adv(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    x, y = obj._cf_method_storage()
    assert name in y


def test_product_method_storage(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    obj = ProductGrid(model, obs)
    x, y = obj._product_method_storage()
    assert "lwc" in x


def test_product_method_storage_adv(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    obj = ProductGrid(model, obs)
    _, y = obj._product_method_storage()
    assert "lwc_adv" in y


def test_regrid_cf_area(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    assert x[0, 0] == 0.75


def test_regrid_cf_area_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[1, :] = ma.masked
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    assert round(x[0, 0], 3) == 0.667


def test_regrid_cf_area_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[:, :] = ma.masked
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    testing.assert_equal(x, ma.masked)


def test_regrid_cf_area_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 99, 1], [0, 1, 1], [99, 0, 1], [0, 0, 0]])
    data = ma.masked_where(data == 99, data)
    d = {"cf_A": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_A"]
    assert x[0, 0] == 0.75


def test_regrid_cf_area_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
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
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    assert x[0, 0] == 0.5


def test_regrid_cf_volume_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 99, 1], [0, 1, 1], [99, 0, 1], [0, 0, 0]])
    data = ma.masked_where(data == 99, data)
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    assert x[0, 0] == 0.5


def test_regrid_cf_volume_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
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
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[1, :] = ma.masked
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    assert round(x[0, 0], 3) == 0.444


def test_regrid_cf_volume_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    obj = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[:, :] = ma.masked
    d = {"cf_V": ma.zeros((1, 1))}
    d = obj._regrid_cf(d, 0, 0, data)
    x = d["cf_V"]
    testing.assert_equal(x, ma.masked)


def test_reshape_data_to_window(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
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
    model = ModelManager(str(model_file), MODEL, PRODUCT)
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
    model = ModelManager(str(model_file), MODEL, PRODUCT)
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


def test_regrid_product(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    obj = ProductGrid(model, obs)
    obj._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    d = {"lwc": ma.zeros((1, 1))}
    ind: ma.MaskedArray = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool
    )
    d = obj._regrid_product(d, 0, 0, ind)
    x = d["lwc"]
    testing.assert_almost_equal(x[0, 0], 1.4)


def test_regrid_product_nan(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
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
    ind: ma.MaskedArray = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool
    )
    d = obj._regrid_product(d, 0, 0, ind)
    x = d["lwc"]
    testing.assert_almost_equal(x[0, 0], 1.5)


def test_regrid_product_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
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
    model = ModelManager(str(model_file), MODEL, "lwc")
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
    model = ModelManager(str(model_file), MODEL, "lwc")
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
    model = ModelManager(str(model_file), MODEL, "lwc")
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
    model = ModelManager(str(model_file), MODEL, "cf")
    ProductGrid(model, obs)
    assert product in model.data


@pytest.mark.parametrize("product", ["iwc", "iwc_adv"])
def test_append_data2object_iwc(product, model_file, obs_file) -> None:
    obs = ObservationManager("iwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "iwc")
    ProductGrid(model, obs)
    assert product in model.data


@pytest.mark.parametrize("product", ["lwc", "lwc_adv"])
def test_append_data2object_lwc(product, model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    ProductGrid(model, obs)
    assert product in model.data
