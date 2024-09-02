from collections import namedtuple

import netCDF4
import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cloudnetpy.categorize import atmos_utils
from cloudnetpy.products.lwc import CloudAdjustor, Lwc, LwcError, LwcSource
from cloudnetpy.products.product_tools import QualityBits, CategoryBits

DIMENSIONS = ("time", "height", "model_time", "model_height")
TEST_ARRAY = np.arange(3)
CategorizeBits = namedtuple("CategorizeBits", ["category_bits", "quality_bits"])


@pytest.fixture(scope="session")
def lwc_source_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        _create_dimensions(root_grp)
        _create_dimension_variables(root_grp)
        var = root_grp.createVariable("altitude", "f8")
        var[:] = 1
        var.units = "km"
        var = root_grp.createVariable("lwp", "f8", "time")
        var[:] = [1, 1, 0.5]
        var = root_grp.createVariable("lwp_error", "f8", "time")
        var[:] = [0.2, 0.2, 0.1]
        var = root_grp.createVariable("rainfall_rate", "i4", "time")
        var[:] = [0, 1, 1]
        var = root_grp.createVariable("category_bits", "i4", "time")
        var[:] = [0, 1, 2]
        var = root_grp.createVariable("quality_bits", "i4", "time")
        var[:] = [8, 16, 32]
        var = root_grp.createVariable("temperature", "f8", ("time", "height"))
        var[:] = np.array([[282, 280, 278], [286, 284, 282], [284, 282, 280]])
        var = root_grp.createVariable("pressure", "f8", ("time", "height"))
        var[:] = np.array([[1010, 1000, 990], [1020, 1010, 1000], [1030, 1020, 1010]])
    return file_name


def _create_dimensions(root_grp):
    n_dim = len(TEST_ARRAY)
    for dim_name in DIMENSIONS:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp):
    for dim_name in DIMENSIONS:
        x = root_grp.createVariable(dim_name, "f8", (dim_name,))
        x[:] = TEST_ARRAY
        if dim_name == "height":
            x.units = "m"


def test_get_atmosphere_t(lwc_source_file):
    obj = LwcSource(lwc_source_file)
    expected = np.array([[282, 280, 278], [286, 284, 282], [284, 282, 280]])
    assert_array_equal(obj.atmosphere[0], expected)


def test_get_atmosphere_p(lwc_source_file):
    obj = LwcSource(lwc_source_file)
    expected = np.array([[1010, 1000, 990], [1020, 1010, 1000], [1030, 1020, 1010]])
    assert_array_equal(obj.atmosphere[-1], expected)


class DataSet:
    variables = {"height": np.array([10, 20, 30])}


class LwcSourceObj(LwcSource):
    def __init__(self):
        self.dataset = DataSet()
        self.path_lengths = np.array([10, 10, 10])
        self.categorize_bits = CategorizeBits(
            category_bits=CategoryBits(
                droplet=np.array([[1, 0, 1], [0, 1, 1]], dtype=bool),
                falling=np.array([]),
                aerosol=np.array([]),
                freezing=np.array([]),
                melting=np.array([]),
                insect=np.array([]),
            ),
            quality_bits=QualityBits(radar=np.array([[1, 0, 1], [0, 1, 1]], dtype=bool),
                                     lidar=np.array([[1, 0, 1], [0, 1, 1]], dtype=bool),
                                     clutter=np.array([]),
                                     molecular=np.array([]),
                                     attenuated_liquid=np.array([]),
                                     corrected_liquid=np.array([]),
                                     attenuated_rain=np.array([]),
                                     corrected_rain=np.array([]),
                                     attenuated_melting=np.array([]),
                                     corrected_melting=np.array([]),
            ),

        )  # type: ignore
        self.atmosphere = (
            np.array([[282, 281, 280], [280, 279, 278]]),
            np.array([[101000, 100500, 100000], [100000, 99500, 99000]]),
        )
        self.lwp = np.array([2.0, 0.0])
        self.lwp_error = np.array([0.1, 0.2])
        self.is_rain = np.array([0, 1])


LWC_OBJ = Lwc(LwcSourceObj())
STATUS_OBJ = CloudAdjustor(LwcSourceObj(), LWC_OBJ)
ERROR_OBJ = LwcError(LwcSourceObj(), LWC_OBJ)


@pytest.mark.parametrize("value", [0, 1])
def test_get_liquid(value):
    assert value in LWC_OBJ.is_liquid


# def test_init_lwc_adiabatic():
#     lwc_source = LwcSourceObj()
#     expected = atmos_utils.fill_clouds_with_lwc_dz(*lwc_source.atmosphere, LWC_OBJ.is_liquid)
#     expected[0, 0] *= 10
#     expected[0, 2] *= 10
#     expected[1, 1] *= 10
#     expected[1, 2] *= 20
#     assert_array_almost_equal(LWC_OBJ._init_lwc_adiabatic(), expected)


# def test_screen_rain_lwc():
#     expected = ma.array([[5, 1, 2], [3, 6, 0]], mask=[[0, 0, 0], [1, 1, 1]])
#     assert isinstance(LWC_OBJ.lwc, ma.MaskedArray)
#     assert_array_equal(expected.mask, LWC_OBJ.lwc.mask)


@pytest.mark.parametrize("value", [0, 1])
def test_init_status(value):
    assert value in STATUS_OBJ._init_status()


@pytest.mark.parametrize("key", ["radar", "lidar"])
def test_get_echo(key):
    assert key in STATUS_OBJ.echo.keys()


@pytest.mark.parametrize("value", [0, 1, 2])
def test_update_status(value):
    time = np.array([0])
    STATUS_OBJ._update_status(time)
    assert value in STATUS_OBJ.status


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_adjust_lwc(value):
    time = 0
    base = 0
    STATUS_OBJ.status = ma.array([[1, 0, 2], [0, 0, 2]])
    STATUS_OBJ._adjust_lwc(time, base)
    assert value in STATUS_OBJ.status


def test_has_converged():
    ind = 1
    assert STATUS_OBJ._has_converged(ind)


def test_out_of_bound():
    ind = 2
    assert STATUS_OBJ._out_of_bound(ind)


def test_find_adjustable_clouds():
    assert 1 not in STATUS_OBJ._find_adjustable_clouds()


def test_find_topmost_clouds():
    expected = np.asarray([[0, 0, 1], [0, 1, 1]], dtype=bool)
    assert_array_equal(STATUS_OBJ._find_topmost_clouds(), expected)


def test_find_echo_combinations_in_liquid():
    STATUS_OBJ.echo["lidar"] = np.array([[0, 1, 0], [1, 1, 0]])
    STATUS_OBJ.echo["radar"] = np.array([[0, 0, 0], [0, 1, 1]])
    STATUS_OBJ.is_liquid = np.array([[1, 1, 1], [0, 1, 1]])
    expected = np.array([[0, 1, 0], [0, 3, 2]])
    assert_array_equal(STATUS_OBJ._find_echo_combinations_in_liquid(), expected)


def test_find_lidar_only_clouds():
    inds = np.array([[1, 0, 0], [0, 1, 3]])
    expected = np.array([True, False])
    assert_array_equal(STATUS_OBJ._find_lidar_only_clouds(inds), expected)


def test_remove_good_profiles():
    top_c = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=bool)
    expected = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=bool)
    assert_array_equal(STATUS_OBJ._remove_good_profiles(top_c), expected)


def test_find_lwp_difference():
    STATUS_OBJ.lwc_adiabatic = np.array([[1, 8, 2], [2, 3, 7]])
    STATUS_OBJ.lwc_source.lwp = np.array([50, 30])
    expected = np.array([60, 90])
    assert_array_equal(STATUS_OBJ._find_lwp_difference(), expected)


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4])
def test_screen_rain_status(value):
    STATUS_OBJ.lwc_source.is_rain = np.array([0, 1])
    STATUS_OBJ.status = ma.array([[0, 2, 2, 3, 1], [1, 3, 0, 2, 2]])
    STATUS_OBJ._mask_rain()
    assert value in STATUS_OBJ.status


def test_limit_error():
    error = np.array([[0, 0, 1], [0.2, 0.4, 0.3]])
    max_v = 0.5
    expected = np.array([[0, 0, 0.5], [0.2, 0.4, 0.3]])
    assert_array_equal(ERROR_OBJ._limit_error(error, max_v), expected)


def test_calc_lwc_gradient():
    from cloudnetpy.utils import l2norm

    ERROR_OBJ.lwc = ma.array([[0.1, 0.2, 0.3], [0.1, 0.3, 0.6]])
    expected = l2norm(*np.gradient(ERROR_OBJ.lwc))
    assert_array_almost_equal(ERROR_OBJ._calc_lwc_gradient(), expected)


def test_calc_lwc_relative_error():
    from cloudnetpy.utils import l2norm

    ERROR_OBJ.lwc = ma.array([[0.1, 0.2, 0.3], [0.1, 0.3, 0.6]])
    x = l2norm(*np.gradient(ERROR_OBJ.lwc))
    expected = x / ERROR_OBJ.lwc / 2
    expected[expected > 5] = 5
    assert_array_almost_equal(ERROR_OBJ._calc_lwc_relative_error(), expected)


def test_calc_lwp_relative_error():
    ERROR_OBJ.lwc_source.lwp = np.array([0.1, 0.5])
    ERROR_OBJ.lwc_source.lwp_error = np.array([0.2, 5.5])
    expected = ERROR_OBJ.lwc_source.lwp_error / ERROR_OBJ.lwc_source.lwp
    expected[expected > 10] = 10
    assert_array_equal(ERROR_OBJ._calc_lwp_relative_error(), expected)


def test_calc_combined_error():
    from cloudnetpy.utils import l2norm, transpose

    err_2d = np.array([[0, 0.1, 0.1], [0.2, 0.4, 0.15]])
    err_1d = np.array([0.3, 0.2])
    expected = l2norm(err_2d, transpose(err_1d))
    assert_array_equal(ERROR_OBJ._calc_combined_error(err_2d, err_1d), expected)


def test_fill_error_array():
    error_in = np.array([[0, 0.1, 0.1], [0.2, 0.4, 0.15]])
    ERROR_OBJ.lwc = ma.array(
        [[0.1, 0.2, 0.1], [0.1, 0.2, 0.2]],
        mask=[[0, 1, 0], [1, 0, 0]],
    )
    expected = ma.array([[0, 0, 0], [0, 0, 0]], mask=[[0, 1, 0], [1, 0, 0]])
    ERROR_OBJ._fill_error_array(error_in)
    error = ERROR_OBJ._fill_error_array(error_in)
    assert_array_almost_equal(error.mask, expected.mask)


# def test_screen_rain_error():
#     expected = ma.array([[0.709, 0, 0.709], [0, 0, 0]], mask=[[0, 1, 0], [1, 1, 1]])
#     assert isinstance(ERROR_OBJ.error, ma.MaskedArray)
#     assert_array_equal(ERROR_OBJ.error.mask, expected.mask)


@pytest.mark.parametrize("key", ["lwc", "lwc_retrieval_status", "lwc_error"])
def test_append_data(lwc_source_file, key):
    lwc_source = LwcSource(lwc_source_file)
    lwc_source.append_results(LWC_OBJ.lwc, STATUS_OBJ.status, ERROR_OBJ.error)
    assert key in lwc_source.data.keys()
