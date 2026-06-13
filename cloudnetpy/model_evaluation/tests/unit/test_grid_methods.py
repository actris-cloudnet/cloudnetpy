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
    grid = ProductGrid(model, obs)
    storage, adv_storage = grid._get_method_storage()
    assert len(storage.keys()) == value


@pytest.mark.parametrize("key, value", [("iwc", 1), ("lwc", 1), ("cf", 2)])
def test_get_method_storage_adv(key, value, model_file, obs_file) -> None:
    obs = ObservationManager(key, str(obs_file))
    model = ModelManager(str(model_file), MODEL, key)
    grid = ProductGrid(model, obs)
    storage, adv_storage = grid._get_method_storage()
    assert len(adv_storage.keys()) == value


@pytest.mark.parametrize("name", ["cf_V", "cf_A"])
def test_cf_method_storage(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    storage, adv_storage = grid._cf_method_storage()
    assert name in storage


@pytest.mark.parametrize("name", ["cf_V_adv", "cf_A_adv"])
def test_cf_method_storage_adv(name, model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    storage, adv_storage = grid._cf_method_storage()
    assert name in adv_storage


def test_product_method_storage(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    storage, adv_storage = grid._product_method_storage()
    assert "lwc" in storage


def test_product_method_storage_adv(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    _, adv_storage = grid._product_method_storage()
    assert "lwc_adv" in adv_storage


def test_regrid_cf_area(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    storage = {"cf_A": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_A"]
    assert result[0, 0] == 0.75


def test_regrid_cf_area_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[1, :] = ma.masked
    storage = {"cf_A": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_A"]
    assert round(result[0, 0], 3) == 0.667


def test_regrid_cf_area_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[:, :] = ma.masked
    storage = {"cf_A": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_A"]
    testing.assert_equal(result, ma.masked)


def test_regrid_cf_area_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 99, 1], [0, 1, 1], [99, 0, 1], [0, 0, 0]])
    data = ma.masked_where(data == 99, data)
    storage = {"cf_A": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_A"]
    assert result[0, 0] == 0.75


def test_regrid_cf_area_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        mask=True,
    )
    storage = {"cf_A": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_A"]
    testing.assert_equal(result, ma.masked)


def test_regrid_cf_volume(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    storage = {"cf_V": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_V"]
    assert result[0, 0] == 0.5


def test_regrid_cf_volume_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 99, 1], [0, 1, 1], [99, 0, 1], [0, 0, 0]])
    data = ma.masked_where(data == 99, data)
    storage = {"cf_V": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_V"]
    assert result[0, 0] == 0.5


def test_regrid_cf_volume_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array(
        [
            [99, 99, 99],
            [99, 99, 99],
            [99, 99, 99],
            [99, 99, 99],
        ],
    )
    data = ma.masked_where(data == 99, data)
    storage = {"cf_V": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_V"]
    testing.assert_equal(result, ma.masked)


def test_regrid_cf_volume_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[1, :] = ma.masked
    storage = {"cf_V": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_V"]
    assert round(result[0, 0], 3) == 0.444


def test_regrid_cf_volume_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    data = ma.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    data[:, :] = ma.masked
    storage = {"cf_V": ma.zeros((1, 1))}
    storage = grid._regrid_cf(storage, 0, 0, data)
    result = storage["cf_V"]
    testing.assert_equal(result, ma.masked)


def test_reshape_data_to_window(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    x_ind = np.array([1, 1, 1, 0, 0, 0])
    y_ind = np.array([1, 1, 0, 0])
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
    grid._obs_data = np.array(
        [
            [1, 2, 3, 4],
            [11, 22, 33, 44],
            [111, 222, 333, 444],
            [5, 6, 7, 8],
            [55, 66, 77, 88],
            [555, 666, 777, 888],
        ],
    )
    result = grid._reshape_data_to_window(ind, x_ind, y_ind)
    expected = np.array([[1, 2], [11, 22], [111, 222]])
    assert result is not None
    testing.assert_array_almost_equal(result, expected)


def test_reshape_data_to_window_middle(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    x_ind = np.array([0, 0, 1, 1, 1, 0])
    y_ind = np.array([0, 1, 1, 0])
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
    grid._obs_data = np.array(
        [
            [1, 2, 3, 4],
            [11, 22, 33, 44],
            [111, 222, 333, 444],
            [5, 6, 7, 8],
            [55, 66, 77, 88],
            [555, 666, 777, 888],
        ],
    )
    result = grid._reshape_data_to_window(ind, x_ind, y_ind)
    expected = np.array([[222, 333], [6, 7], [66, 77]])
    assert result is not None
    testing.assert_array_almost_equal(result, expected)


def test_reshape_data_to_window_empty(model_file, obs_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    grid = ProductGrid(model, obs)
    x_ind = np.array(
        [
            1,
            1,
            1,
            0,
            0,
            0,
        ],
    )
    y_ind = np.array([0, 0, 0, 0])
    ind = np.array([1, 1, 0, 0], dtype=bool)
    result = grid._reshape_data_to_window(ind, x_ind, y_ind)
    assert result is None


def test_regrid_product(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    grid._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    storage = {"lwc": ma.zeros((1, 1))}
    ind: ma.MaskedArray = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool
    )
    storage = grid._regrid_product(storage, 0, 0, ind)
    result = storage["lwc"]
    testing.assert_almost_equal(result[0, 0], 1.4)


def test_regrid_product_nan(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    grid._obs_data = ma.array(
        [
            [1, 99, 1, 1],
            [99, 1, 2, 2],
            [3, 3, 99, 3],
            [4, 99, 4, 4],
        ],
    )
    grid._obs_data = ma.masked_where(grid._obs_data == 99, grid._obs_data)
    storage = {"lwc": ma.zeros((1, 1))}
    ind: ma.MaskedArray = ma.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool
    )
    storage = grid._regrid_product(storage, 0, 0, ind)
    result = storage["lwc"]
    testing.assert_almost_equal(result[0, 0], 1.5)


def test_regrid_product_all_nan(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    data = ma.array(
        [
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
            [99, 99, 99, 99],
        ],
    )
    grid._obs_data = ma.masked_where(data == 99, data)
    storage = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    storage = grid._regrid_product(storage, 0, 0, ind)
    assert storage["lwc"][0, 0].mask == True


def test_regrid_product_masked(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    grid._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    grid._obs_data[2, :] = ma.masked
    storage = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    storage = grid._regrid_product(storage, 0, 0, ind)
    result = storage["lwc"]
    testing.assert_almost_equal(result[0, 0], 1.4)


def test_regrid_product_all_masked(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    grid._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    grid._obs_data[:, :] = ma.masked
    storage = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    storage = grid._regrid_product(storage, 0, 0, ind)
    result = storage["lwc"]
    testing.assert_almost_equal(result, ma.masked)


def test_regrid_product_none(model_file, obs_file) -> None:
    obs = ObservationManager("lwc", str(obs_file))
    model = ModelManager(str(model_file), MODEL, "lwc")
    grid = ProductGrid(model, obs)
    grid._obs_data = ma.array([[1, 1, 1, 1], [2, 1, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    storage = {"lwc": ma.zeros((1, 1))}
    ind = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    storage = grid._regrid_product(storage, 0, 0, ind)
    result = storage["lwc"]
    assert result[0, 0].mask == True


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
