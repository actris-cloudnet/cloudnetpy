import numpy as np
import pytest
from numpy import ma, testing
from scipy.special import gamma

from cloudnetpy.model_evaluation.products.advance_methods import AdvanceProductMethods
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager

MODEL = "ecmwf"
OUTPUT_FILE = ""
PRODUCT = "cf"


@pytest.mark.parametrize("name", ("ecmwf_cf_cirrus",))
def test_cf_cirrus_filter(obs_file, model_file, name) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    AdvanceProductMethods(model, str(model_file), obs)
    assert name in model.data


@pytest.mark.parametrize(
    "name, data",
    [
        ("cf", ma.array([[-99, 2], [3, 6], [5, 8]], mask=[[1, 0], [0, 0], [0, 0]])),
        ("h", np.array([[10, 14], [8, 14], [9, 15]])),
    ],
)
def test_getvar_from_object(obs_file, model_file, name, data) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    x = adv_pro.getvar_from_object(name)
    testing.assert_array_almost_equal(x, data)


@pytest.mark.parametrize("name", ("T",))
def test_getvar_from_object_None(obs_file, model_file, name) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    with pytest.raises(KeyError):
        adv_pro.getvar_from_object(name)


@pytest.mark.parametrize(
    "radar_f, values",
    [
        (35, (0.000242, -0.0186, 0.0699, -1.63)),
        (95, (0.00058, -0.00706, 0.0923, -0.992)),
    ],
)
def test_set_frequency_parameters(obs_file, model_file, radar_f, values) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    obs.radar_freq = radar_f
    x = adv_pro.set_frequency_parameters()
    assert x == values


def test_fit_z_sensitivity(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    h = np.array([[5000, 9000, 13000], [10000, 15000, 20000], [8000, 12000, 16000]])
    compare = np.array([[0, 0.15, 0.5], [0.1, 1, 0], [0.15, 0, 1]])
    x = adv_pro.fit_z_sensitivity(h)
    testing.assert_array_almost_equal(x, compare)


def test_filter_high_iwc_low_cf(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    cf = ma.array([0.0001, 0.0002, 0, 0.0001, 1, 0.0006])
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    compare = ma.array([0.0001, 0.0002, 0, -99, 1, 0.0006], mask=[0, 0, 0, 1, 0, 0])
    x = adv_pro.filter_high_iwc_low_cf(cf, iwc, lwc)
    testing.assert_array_almost_equal(x, compare)


def test_filter_high_iwc_low_cf_no_ice(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    cf = ma.array([0.0001, 0.0002, 0, 0, 0, 0.0006])
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    with pytest.raises(ValueError):
        adv_pro.filter_high_iwc_low_cf(cf, iwc, lwc)


def test_mask_weird_indices(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    cf = ma.array([0.0001, 0.0002, 0, 0.0001, 1, 0.0006])
    compare = ma.copy(cf)
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    ind = (iwc / cf > 0.5e-3) & (cf < 0.001)
    ind = ind | (iwc == 0) & (lwc == 0) & (cf == 0)
    compare[ind] = ma.masked
    x = adv_pro.mask_weird_indices(cf, iwc, lwc)
    testing.assert_array_almost_equal(x, compare)


def test_mask_weird_indices_values(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    cf = ma.array([0.0001, 0.0002, 0, 0.0001, 1, 0.0006])
    iwc = np.array([0.0, 0, 0, 0.2, 0.4, 0])
    lwc = np.array([0.0, 0.02, 0.01, 0, 0.01, 0.01])
    compare = ma.array([0.0001, 0.0002, 0, -99, 1, 0.0006], mask=[0, 0, 0, 1, 0, 0])
    x = adv_pro.mask_weird_indices(cf, iwc, lwc)
    testing.assert_array_almost_equal(x, compare)


def test_find_ice_in_clouds(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
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
    x, _ = adv_pro.find_ice_in_clouds(cf_f, iwc, lwc)
    testing.assert_array_almost_equal(x, expected)


def test_get_ice_indices(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
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
    x = np.array([0, 1])
    y = np.array([3, 2])
    expected = (x, y)
    result = adv_pro.get_ice_indices(cf_f, iwc, lwc)
    testing.assert_array_almost_equal(result, expected)


def test_iwc_variance(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    x = np.array([0, 1, 2])
    y = np.array([0, 0, 1])
    ind = (x, y)
    height = np.array([[1, 5], [2, 6], [3, 7]])
    x = adv_pro.iwc_variance(height, ind)
    assert len(x) == 3


def test_calculate_variance_iwc(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    shear = np.array([[1, 1, 2, 1], [2, 2, 1, 0], [0, 0, 1, 0]])
    ind_arr = np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
    ind = np.where(ind_arr > -1)
    compare = 10 ** (0.3 * np.log10(model.resolution_h) - 0.04 * shear[ind] - 1.03)
    x = adv_pro.calculate_variance_iwc(shear, tuple(ind))
    testing.assert_array_almost_equal(x, compare)


def test_calculate_wind_shear(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    u = np.array([[1, 2, 0, 1], [-1, 0, 1, -1], [1, 0, 1, -1]])
    v = np.array([[1, 0, 1, -1], [1, 2, -1, 0], [1, 2, 0, 1]])
    wind = np.sqrt(np.power(u, 2) + np.power(v, 2))
    h = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    compare = np.array([[2, 2.83, 2.24, -2.24], [0, 0, 0, 0], [2, 0, -1, 1]])
    x = adv_pro.calculate_wind_shear(wind, u, v, h)
    testing.assert_array_almost_equal(np.round(x, 2), compare)


def test_calculate_iwc_distribution(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    n_std = 5
    n_dist = 250
    f_variance_iwc = 0.1
    cloud_iwc = 0.2
    finish = cloud_iwc + n_std * (np.sqrt(f_variance_iwc) * cloud_iwc)
    compare = np.arange(0, finish, finish / (n_dist - 1))
    x = adv_pro.calculate_iwc_distribution(cloud_iwc, f_variance_iwc)
    testing.assert_array_almost_equal(x, compare)


def test_gamma_distribution(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    iwc_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    compare = np.zeros(iwc_dist.shape)
    f_variance_iwc = 0.1
    cloud_iwc = 0.2
    alpha = 1 / f_variance_iwc
    for i in range(len(iwc_dist)):
        compare[i] = (
            1
            / gamma(alpha)
            * (alpha / cloud_iwc) ** alpha
            * iwc_dist[i] ** (alpha - 1)
            * ma.exp(-(alpha * iwc_dist[i] / cloud_iwc))
        )
    x = adv_pro.gamma_distribution(iwc_dist, f_variance_iwc, cloud_iwc)
    testing.assert_array_almost_equal(x, compare)


def test_get_observation_index(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    tZT = 0.01
    z_sen = 0.02
    temperature = -13
    tT = 0.04
    tZ = 0.05
    t = 0.06
    min_iwc = 10 ** (tZT * z_sen * temperature + tT * temperature + tZ * z_sen + t)
    iwc_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    compare = iwc_dist > min_iwc
    x = adv_pro.get_observation_index(
        iwc_dist,
        tZT,
        tT,
        tZ,
        np.array([t]),
        temperature,
        z_sen,
    )
    testing.assert_array_almost_equal(x, compare)


def test_filter_cirrus(obs_file, model_file) -> None:
    obs = ObservationManager(PRODUCT, str(obs_file))
    model = ModelManager(str(model_file), MODEL, OUTPUT_FILE, PRODUCT)
    adv_pro = AdvanceProductMethods(model, str(model_file), obs)
    cf_f = 0.7
    p = np.array([1, 2, 3, 4, 5, 6])
    ind = np.array([0, 0, 1, 1, 0, 1], dtype=bool)
    compare = (np.sum(p * ind) / np.sum(p)) * cf_f
    x = adv_pro.filter_cirrus(p, ind, np.array([cf_f]))
    testing.assert_almost_equal(x, compare)
