import numpy as np
import pytest
from numpy import ma, testing
from scipy.special import gamma

from cloudnetpy.model_evaluation.products.advance_methods import AdvanceProductMethods
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager

MODEL = "ecmwf"
PRODUCT = "cf"


@pytest.mark.parametrize("name", ("model_cf_cirrus",))
def test_cf_cirrus_filter(obs_file, model_file, name) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    AdvanceProductMethods(model, obs)
    assert name in model.data


@pytest.mark.parametrize(
    "name, data",
    [
        (
            "cf",
            ma.array(
                [[-99, 2], [3, 6], [5, 8]],
                mask=[[True, False], [False, False], [False, False]],
            ),
        ),
        ("h", np.array([[10, 14], [8, 14], [9, 15]])),
    ],
)
def test_getvar_from_object(obs_file, model_file, name, data) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    result = advance.getvar_from_object(name)
    testing.assert_array_almost_equal(result, data)


@pytest.mark.parametrize("name", ("T",))
def test_getvar_from_object_None(obs_file, model_file, name) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    with pytest.raises(KeyError):
        advance.getvar_from_object(name)


@pytest.mark.parametrize(
    "radar_f, values",
    [
        (35, (0.000242, -0.0186, 0.0699, -1.63)),
        (95, (0.00058, -0.00706, 0.0923, -0.992)),
    ],
)
def test_set_frequency_parameters(obs_file, model_file, radar_f, values) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    obs.radar_freq = radar_f
    result = advance.set_frequency_parameters()
    assert result == values


def test_fit_z_sensitivity(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    height = np.array(
        [[5000, 9000, 13000], [10000, 15000, 20000], [8000, 12000, 16000]]
    )
    expected = ma.masked_invalid([[np.nan, 0.15, 0.5], [0.1, 1, np.nan], [0.15, 0, 1]])
    result = advance.fit_z_sensitivity(height)
    testing.assert_array_almost_equal(result, expected)


def test_filter_high_iwc_low_cf(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    cf = ma.array([0.0001, 0.0002, 0, 0.0001, 1, 0.0006])
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    expected = ma.array(
        [0.0001, 0.0002, 0, -99, 1, 0.0006],
        mask=[False, False, False, True, False, False],
    )
    result = advance.filter_high_iwc_low_cf(cf, iwc, lwc)
    testing.assert_array_almost_equal(result, expected)


def test_filter_high_iwc_low_cf_no_ice(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    cf = ma.array([0.0001, 0.0002, 0, 0, 0, 0.0006])
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    with pytest.raises(ValueError):
        advance.filter_high_iwc_low_cf(cf, iwc, lwc)


def test_mask_weird_indices(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    cf = ma.array([0.0001, 0.0002, 0, 0.0001, 1, 0.0006])
    expected = ma.copy(cf)
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    ind = (iwc / cf > 0.5e-3) & (cf < 0.001)
    ind = ind | (iwc == 0) & (lwc == 0) & (cf == 0)
    expected[ind] = ma.masked
    result = advance.mask_weird_indices(cf, iwc, lwc)
    testing.assert_array_almost_equal(result, expected)


def test_mask_weird_indices_values(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    cf = ma.array([0.0001, 0.0002, 0, 0.0001, 1, 0.0006])
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    expected = ma.array(
        [0.0001, 0.0002, 0, -99, 1, 0.0006],
        mask=[False, False, False, True, False, False],
    )
    result = advance.mask_weird_indices(cf, iwc, lwc)
    testing.assert_array_almost_equal(result, expected)


def test_find_ice_in_clouds(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    cf_f = np.array(
        [
            [-1, 0, 2, 3],
            [-1, 1, 2, 3],
        ],
    )
    iwc = np.array(
        [
            [-1, 0, 2, 31],
            [-1, 1, 200, 3],
        ],
    )
    lwc = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    expected = np.array([31 / 3 * 1000, 200 / 2 * 1000])
    result, _ = advance.find_ice_in_clouds(cf_f, iwc, lwc)
    testing.assert_array_almost_equal(result, expected)


def test_get_ice_indices(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    cf_f = np.array(
        [
            [-1, 0, 2, 3],
            [-1, 1, 2, 3],
        ],
    )
    iwc = np.array(
        [
            [-1, 0, 2, 31],
            [-1, 1, 200, 3],
        ],
    )
    lwc = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    rows = np.array([0, 1])
    cols = np.array([3, 2])
    expected = (rows, cols)
    result = advance.get_ice_indices(cf_f, iwc, lwc)
    testing.assert_array_almost_equal(result, expected)


def test_iwc_variance(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    rows = np.array([0, 1, 2])
    cols = np.array([0, 0, 1])
    ind = (rows, cols)
    height = np.array([[1, 5], [2, 6], [3, 7]])
    result = advance.iwc_variance(height, ind)
    assert len(result) == 3


def test_calculate_variance_iwc(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    shear = np.array([[1, 1, 2, 1], [2, 2, 1, 0], [0, 0, 1, 0]])
    ind_arr = np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
    ind = np.where(ind_arr > -1)
    expected = 10 ** (0.3 * np.log10(model.resolution_h) - 0.04 * shear[ind] - 1.03)
    result = advance.calculate_variance_iwc(shear, tuple(ind))
    testing.assert_array_almost_equal(result, expected)


def test_calculate_wind_shear(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    u = np.array([[1, 2, 0, 1], [-1, 0, 1, -1], [1, 0, 1, -1]])
    v = np.array([[1, 0, 1, -1], [1, 2, -1, 0], [1, 2, 0, 1]])
    wind = np.sqrt(np.power(u, 2) + np.power(v, 2))
    height = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    expected = np.array([[2, 2.83, 2.24, -2.24], [0, 1.41, 0.71, 1.41], [2, 0, -1, 1]])
    result = advance.calculate_wind_shear(wind, u, v, height)
    testing.assert_array_almost_equal(np.round(result, 2), expected)


def test_calculate_iwc_distribution(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    n_std = 5
    n_dist = 250
    f_variance_iwc = 0.1
    cloud_iwc = 0.2
    finish = cloud_iwc + n_std * (np.sqrt(f_variance_iwc) * cloud_iwc)
    expected = np.arange(0, finish, finish / (n_dist - 1))
    result = advance.calculate_iwc_distribution(cloud_iwc, f_variance_iwc)
    testing.assert_array_almost_equal(result, expected)


def test_gamma_distribution(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    iwc_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    expected = np.zeros(iwc_dist.shape)
    f_variance_iwc = 0.1
    cloud_iwc = 0.2
    alpha = 1 / f_variance_iwc
    for i in range(len(iwc_dist)):
        expected[i] = (
            1
            / gamma(alpha)
            * (alpha / cloud_iwc) ** alpha
            * iwc_dist[i] ** (alpha - 1)
            * ma.exp(-(alpha * iwc_dist[i] / cloud_iwc))
        )
    result = advance.gamma_distribution(iwc_dist, f_variance_iwc, cloud_iwc)
    testing.assert_array_almost_equal(result, expected)


def test_filter_cirrus(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    advance = AdvanceProductMethods(model, obs)
    cf_f = 0.7
    p_iwc = np.array([1, 2, 3, 4, 5, 6])
    ind = np.array([0, 0, 1, 1, 0, 1], dtype=bool)
    expected = (np.sum(p_iwc * ind) / np.sum(p_iwc)) * cf_f
    result = advance.filter_cirrus(p_iwc, ind, np.array([cf_f]))
    testing.assert_almost_equal(result, expected)
