from bisect import bisect_left

import netCDF4
import numpy as np
import pytest
from numpy import testing

from cloudnetpy import utils
from cloudnetpy.products import drizzle
from cloudnetpy.products.drizzle_error import get_drizzle_error

DIMENSIONS_X = ("time", "model_time")
TEST_ARRAY_X = np.arange(2)
DIMENSIONS_Y = ("height", "model_height")
TEST_ARRAY_Y = np.arange(3)


@pytest.fixture(scope="session")
def drizzle_source_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        _create_dimensions(root_grp, TEST_ARRAY_X, DIMENSIONS_X)
        _create_dimension_variables(root_grp, TEST_ARRAY_X, DIMENSIONS_X)
        _create_dimensions(root_grp, TEST_ARRAY_Y, DIMENSIONS_Y)
        _create_dimension_variables(root_grp, TEST_ARRAY_Y, DIMENSIONS_Y)
        var = root_grp.createVariable("altitude", "f8")
        var[:] = 1
        var.units = "km"
        var = root_grp.createVariable("beta", "f8", ("time", "height"))
        var[:] = [[0.1, 0.1, 0.1], [1, 0.2, 3]]
        var = root_grp.createVariable("beta_error", "f8")
        var[:] = 0.1
        var = root_grp.createVariable("beta_bias", "f8")
        var[:] = 0.1
        var = root_grp.createVariable("v", "f8", ("time", "height"))
        var[:] = [[1, 2, 3], [1, 2, 3]]
        var = root_grp.createVariable("Z", "f8", ("time", "height"))
        var[:] = [[1, 0.1, 0.2], [0.3, 2, 0.1]]
        var = root_grp.createVariable("Z_error", "f8", ("time", "height"))
        var[:] = [[0.01, 0.1, 0.2], [0.3, 0.2, 0.1]]
        var = root_grp.createVariable("Z_bias", "f8")
        var[:] = 0.1
        var = root_grp.createVariable("category_bits", "i4", ("time", "height"))
        var[:] = [[0, 1, 2], [4, 8, 16]]
        var = root_grp.createVariable("quality_bits", "i4", ("time", "height"))
        var[:] = [[0, 1, 2], [4, 8, 16]]
        var = root_grp.createVariable("radar_frequency", "f8")
        var[:] = 35.5
    return file_name


def _create_dimensions(root_grp, test_array, dimension):
    n_dim = len(test_array)
    for dim_name in dimension:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp, test_array, dimension):
    for dim_name in dimension:
        x = root_grp.createVariable(dim_name, "f8", (dim_name,))
        x[:] = test_array
        if dim_name == "height":
            x.units = "m"


def test_convert_z_units(drizzle_source_file):
    obj = drizzle.DrizzleSource(drizzle_source_file)
    z = obj.getvar("Z") - 180
    expected = utils.db2lin(z)
    testing.assert_array_almost_equal(obj._convert_z_units(), expected)


@pytest.mark.parametrize("key", ["Do", "mu", "S", "lwf", "termv", "width", "ray", "v"])
def test_read_mie_lut(drizzle_source_file, key):
    obj = drizzle.DrizzleSource(drizzle_source_file)
    assert key in obj.mie.keys()


def test_get_wl_band(drizzle_source_file):
    obj = drizzle.DrizzleSource(drizzle_source_file)
    expected = "35"
    testing.assert_equal(obj._get_wl_band(), expected)


@pytest.fixture(scope="session")
def drizzle_cat_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        _create_dimensions(root_grp, TEST_ARRAY_X, DIMENSIONS_X)
        _create_dimension_variables(root_grp, TEST_ARRAY_X, DIMENSIONS_X)
        _create_dimensions(root_grp, TEST_ARRAY_Y, DIMENSIONS_Y)
        _create_dimension_variables(root_grp, TEST_ARRAY_Y, DIMENSIONS_Y)
        var = root_grp.createVariable("altitude", "f8", ("time",))
        var[:] = 100
        var = root_grp.createVariable("uwind", "f8", ("model_time", "model_height"))
        var[:] = [[2, 2, 1], [1, 3, 5]]
        var = root_grp.createVariable("vwind", "f8", ("model_time", "model_height"))
        var[:] = [[-2, -2, 1], [1, -3, 0]]
        var = root_grp.createVariable("category_bits", "i4", ("time", "height"))
        var[:] = [[0, 1, 2], [4, 8, 16]]
        var = root_grp.createVariable("quality_bits", "i4", ("time", "height"))
        var[:] = [[0, 1, 2], [4, 8, 16]]
        var = root_grp.createVariable("rainfall_rate", "i4", "time")
        var[:] = [0, 0]
        var = root_grp.createVariable("v_sigma", "f8", ("time", "height"))
        var[:] = [[-2, np.nan, 2], [1, -1, 0]]  # type: ignore
        var = root_grp.createVariable("width", "f8", ("time", "height"))
        var[:] = [[2, 0, 1], [1, 3, 0]]
    return file_name


def test_find_v_sigma(drizzle_cat_file):
    obj = drizzle.DrizzleClassification(drizzle_cat_file)
    expected = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
    testing.assert_array_almost_equal(obj._find_v_sigma(drizzle_cat_file), expected)


def test_find_warm_liquid(drizzle_cat_file):
    obj = drizzle.DrizzleClassification(drizzle_cat_file)
    obj.category_bits.droplet = np.array([0, 0, 0, 1, 1, 1, 0], dtype=bool)
    obj.category_bits.freezing = np.array([1, 1, 0, 0, 1, 0, 1], dtype=bool)
    expected = np.array([0, 0, 0, 1, 0, 1, 0], dtype=bool)
    testing.assert_array_almost_equal(obj._find_warm_liquid(), expected)


@pytest.mark.parametrize(
    "is_rain, falling, droplet, cold, melting, insect, "
    "radar, lidar, clutter, molecular, attenuated, v_sigma",
    [
        (
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([1, 1, 0, 1]),
        ),
    ],
)
def test_find_drizzle(
    drizzle_cat_file,
    is_rain,
    falling,
    droplet,
    cold,
    melting,
    insect,
    radar,
    lidar,
    clutter,
    molecular,
    attenuated,
    v_sigma,
):
    obj = drizzle.DrizzleClassification(drizzle_cat_file)
    obj.is_rain = is_rain
    obj.category_bits.falling = falling
    obj.category_bits.droplet = droplet
    obj.category_bits.freezing = cold
    obj.category_bits.melting = melting
    obj.category_bits.insect = insect
    obj.quality_bits.radar = radar
    obj.quality_bits.lidar = lidar
    obj.quality_bits.clutter = clutter
    obj.quality_bits.molecular = molecular
    obj.quality_bits.attenuated_liquid = attenuated
    obj.is_v_sigma = v_sigma
    expected = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
    # testing.assert_array_almost_equal(obj._find_drizzle(), expected)


@pytest.mark.parametrize(
    "is_rain, warm, falling, melting, insect, radar, clutter, molecular",
    [
        (
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
        ),
    ],
)
def test_find_would_be_drizzle(
    drizzle_cat_file,
    is_rain,
    warm,
    falling,
    melting,
    insect,
    radar,
    clutter,
    molecular,
):
    obj = drizzle.DrizzleClassification(drizzle_cat_file)
    obj.is_rain = is_rain
    obj.warm_liquid = warm
    obj.category_bits.falling = falling
    obj.category_bits.melting = melting
    obj.category_bits.insect = insect
    obj.quality_bits.radar = radar
    obj.quality_bits.clutter = clutter
    obj.quality_bits.molecular = molecular
    expected = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]])
    testing.assert_array_almost_equal(obj._find_would_be_drizzle(), expected)


def test_find_cold_rain(drizzle_cat_file):
    obj = drizzle.DrizzleClassification(drizzle_cat_file)
    expected = np.array([0, 1])
    testing.assert_array_almost_equal(obj._find_cold_rain(), expected)


def test_calculate_spectral_width(drizzle_cat_file):
    obj = drizzle.SpectralWidth(drizzle_cat_file)
    width = netCDF4.Dataset(drizzle_cat_file).variables["width"][:]
    v_sigma = netCDF4.Dataset(drizzle_cat_file).variables["v_sigma"][:]
    factor = obj._calc_v_sigma_factor()
    expected = width - factor * v_sigma
    testing.assert_almost_equal(obj._calculate_spectral_width(), expected)


def test_calc_beam_divergence(drizzle_cat_file):
    obj = drizzle.SpectralWidth(drizzle_cat_file)
    height = netCDF4.Dataset(drizzle_cat_file).variables["height"][:]
    height_agl = height - 100
    expected = height_agl * np.deg2rad(0.5)
    testing.assert_almost_equal(obj._calc_beam_divergence(), expected)


def test_calc_v_sigma_factor(drizzle_cat_file):
    obj = drizzle.SpectralWidth(drizzle_cat_file)
    height = netCDF4.Dataset(drizzle_cat_file).variables["height"][:]
    height_agl = height - 100
    uwind = netCDF4.Dataset(drizzle_cat_file).variables["uwind"][:]
    vwind = netCDF4.Dataset(drizzle_cat_file).variables["vwind"][:]
    beam = height_agl * np.deg2rad(0.5)
    wind = utils.l2norm(uwind, vwind)
    a_wind = (wind + beam) ** (2 / 3)
    s_wind = (30 * wind + beam) ** (2 / 3)
    expected = a_wind / (s_wind - a_wind)
    testing.assert_array_almost_equal(obj._calc_v_sigma_factor(), expected)


def test_calc_horizontal_wind(drizzle_cat_file):
    obj = drizzle.SpectralWidth(drizzle_cat_file)
    uwind = netCDF4.Dataset(drizzle_cat_file).variables["uwind"][:]
    vwind = netCDF4.Dataset(drizzle_cat_file).variables["vwind"][:]
    expected = utils.l2norm(uwind, vwind)
    testing.assert_array_almost_equal(obj._calc_horizontal_wind(), expected)


@pytest.fixture(scope="session")
def class_objects(drizzle_source_file, drizzle_cat_file):
    drizzle_source = drizzle.DrizzleSource(drizzle_source_file)
    drizzle_class = drizzle.DrizzleClassification(drizzle_cat_file)
    spectral_w = drizzle.SpectralWidth(drizzle_cat_file)
    return [drizzle_source, drizzle_class, spectral_w]


@pytest.mark.parametrize("key", ["Do", "mu", "S", "beta_corr"])
def test_init_variables(class_objects, key):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleSolver(d_source, d_class, s_width)
    result, x = obj._init_variables()
    assert key in result.keys()


def test_calc_beta_z_ratio(class_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleSolver(d_source, d_class, s_width)
    obj._data.beta = np.array([[1, 1, 2], [1, 1, 3]])
    obj._data.z = np.array([[2, 2, 1], [1, 1, 1]])
    expected = 2 / np.pi * obj._data.beta / obj._data.z
    testing.assert_array_almost_equal(obj._calc_beta_z_ratio(), expected)


def test_find_lut_indices(class_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleSolver(d_source, d_class, s_width)
    ind = (1, 2)
    dia_init = np.array([[1, 3, 2], [3, 1, 2]])
    n_dia = 1
    n_width = 2
    ind_d = bisect_left(obj._data.mie["Do"], dia_init[ind], hi=n_dia - 1)
    ind_w = bisect_left(obj._width_lut[:, ind_d], -obj._width_ht[ind], hi=n_width - 1)
    expected = (ind_w, ind_d)
    testing.assert_almost_equal(
        obj._find_lut_indices(ind, dia_init, n_dia, n_width),
        expected,
    )


@pytest.mark.parametrize("key, value", [("Do", 10), ("mu", -1), ("S", 93.7247943)])
def test_update_result_tables(class_objects, key, value):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleSolver(d_source, d_class, s_width)
    ind = (0, 1)
    dia = 10
    lut = (0, 1)
    obj._update_result_tables(ind, dia, lut)
    testing.assert_almost_equal(obj.params[key][ind], value)


def test_is_converged(class_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleSolver(d_source, d_class, s_width)
    ind = (1, 2)
    dia_init = np.array([[1, 3, 2], [3, 1, 2]])
    dia = 1
    expected = False
    assert obj._is_converged(ind, dia, dia_init) == expected


def test_calc_dia(class_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleSolver(d_source, d_class, s_width)
    beta_z = np.array([1, 2, 3])
    expected = (drizzle.gamma(3) / drizzle.gamma(7) * 3.67**4 / beta_z) ** (1 / 4)
    testing.assert_array_almost_equal(obj._calc_dia(beta_z), expected)


@pytest.fixture(scope="session")
def params_objects(class_objects):
    d_source, d_class, s_width = class_objects
    return drizzle.DrizzleSolver(d_source, d_class, s_width)


def test_find_indices(class_objects, params_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleProducts(d_source, params_objects)
    obj._params["Do"] = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    x, y = obj._find_indices()
    expected = (np.array([0, 0, 1, 1]), np.array([1, 2, 0, 1]))
    testing.assert_array_almost_equal(x, expected)


@pytest.mark.parametrize(
    "key",
    ["drizzle_N", "drizzle_lwc", "drizzle_lwf", "v_drizzle", "v_air"],
)
def test_calc_derived_products(class_objects, params_objects, key):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleProducts(d_source, params_objects)
    dictio = obj._calc_derived_products()
    assert key in dictio.keys()


def test_calc_density(class_objects, params_objects):
    d_source, _, _ = class_objects
    obj = drizzle.DrizzleProducts(d_source, params_objects)
    obj._data.z = np.array([[1, 1, 1], [1, 1, 1]])
    a = 3.67**6 / 1**6
    expected = np.array([[0.0, a, a], [a, a, 0.0]])
    testing.assert_array_almost_equal(obj._calc_density(), expected)


def test_calc_lwc(class_objects, params_objects):
    d_source, _, _ = class_objects
    obj = drizzle.DrizzleProducts(d_source, params_objects)
    dia, mu, s = (obj._params.get(key) for key in ("Do", "mu", "S"))
    assert mu is not None
    assert obj._data.beta is not None
    assert s is not None
    assert dia is not None
    gamma_ratio = drizzle.gamma(4 + mu) / drizzle.gamma(3 + mu) / (3.67 + mu)
    assert gamma_ratio is not None
    expected = 1000 / 3 * obj._data.beta * s * dia * gamma_ratio
    testing.assert_array_almost_equal(obj._calc_lwc(), expected)


def test_calc_lwf(class_objects, params_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleProducts(d_source, params_objects)
    lwc_in = np.array([[0.001, 0.001, 0.002], [0.003, 0.002, 0.001]])
    expected = np.array([[0.001, 0.005508, 0.011016], [0.016524, 0.011016, 0.001]])
    testing.assert_array_almost_equal(obj._calc_lwf(lwc_in), expected)


def test_calc_fall_velocity(class_objects, params_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleProducts(d_source, params_objects)
    expected = np.array([[0, -7.11002091, -7.11002091], [-7.11002091, -7.11002091, 0]])
    testing.assert_array_almost_equal(obj._calc_fall_velocity(), expected)


def test_calc_v_air(class_objects, params_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.DrizzleProducts(d_source, params_objects)
    d_v = np.array([[2.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
    obj._ind_drizzle = (np.array([0, 1]), np.array([1, 2]))
    expected = np.array([[-2.0, 0.0, -4.0], [-1.0, -3.0, -2.0]])
    testing.assert_array_almost_equal(obj._calc_v_air(d_v), expected)


@pytest.fixture(scope="session")
def ret_status(class_objects):
    d_source, d_class, s_width = class_objects
    obj = drizzle.RetrievalStatus(d_class)
    return obj


@pytest.mark.parametrize("value", [0, 1, 2])
def test_find_retrieval_below_melting(ret_status, value):
    obj = ret_status
    obj.retrieval_status = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
    obj.drizzle_class.cold_rain = np.array([0, 1, 1])
    obj.drizzle_class.drizzle = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
    obj._find_retrieval_below_melting()
    assert value in obj.retrieval_status


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4])
def test_find_retrieval_in_warm_liquid(ret_status, value):
    obj = ret_status
    obj.retrieval_status = np.array([[0, 0, 1], [1, 2, 3], [2, 1, 1]])
    obj.drizzle_class.warm_liquid = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]])
    obj._find_retrieval_in_warm_liquid()
    assert value in obj.retrieval_status


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4, 5])
def test_get_retrieval_status(ret_status, value):
    obj = ret_status
    obj.retrieval_status = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
    obj.drizzle_class.cold_rain = np.array([0, 1, 0])
    obj.drizzle_class.drizzle = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
    obj.drizzle_class.warm_liquid = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1]])
    obj.drizzle_class.would_be_drizzle = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    obj.drizzle_class.is_rain = np.array([1, 0, 0])
    obj._get_retrieval_status()
    assert value in obj.retrieval_status


@pytest.fixture(scope="session")
def result(class_objects, params_objects):
    d_source, d_class, s_width = class_objects
    errors = get_drizzle_error(d_source, params_objects)
    cal_products = drizzle.DrizzleProducts(d_source, params_objects)
    return {**params_objects.params, **cal_products.derived_products, **errors}


def test_screen_rain(class_objects, result):
    from cloudnetpy.products.drizzle import _screen_rain

    d_source, d_class, s_width = class_objects
    result = _screen_rain(result, d_class)
    expected = True
    for key in result.keys():
        if not utils.isscalar(result[key]):
            if np.any(result[key][-1]) != np.any(np.array([0, 0, 0])):
                expected = False
    assert expected is True


@pytest.mark.parametrize(
    "key",
    [
        "Do",
        "mu",
        "S",
        "beta_corr",
        "drizzle_N",
        "drizzle_lwc",
        "drizzle_lwf",
        "v_drizzle",
        "v_air",
        "Do_error",
        "drizzle_lwc_error",
        "drizzle_lwf_error",
        "S_error",
        "Do_bias",
        "drizzle_lwc_bias",
        "drizzle_lwf_bias",
        "drizzle_N_error",
        "v_drizzle_error",
        "mu_error",
        "drizzle_N_bias",
        "v_drizzle_bias",
    ],
)
def test_append_data(class_objects, result, key):
    from cloudnetpy.products.drizzle import _append_data

    d_source, d_class, s_width = class_objects
    _append_data(d_source, result)
    assert key in d_source.data.keys()
