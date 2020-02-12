import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from collections import namedtuple
import pytest
import netCDF4
from cloudnetpy.products.lwc import LwcSource, Lwc, LwcStatus, LwcError
from cloudnetpy.categorize import atmos


DIMENSIONS = ('time', 'height', 'model_time', 'model_height')
TEST_ARRAY = np.arange(3)
CategorizeBits = namedtuple('CategorizeBits', ['category_bits', 'quality_bits'])


@pytest.fixture(scope='session')
def lwc_source_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    _create_dimensions(root_grp)
    _create_dimension_variables(root_grp)
    var = root_grp.createVariable('altitude', 'f8')
    var[:] = 1
    var.units = 'km'
    var = root_grp.createVariable('lwp', 'f8', 'time')
    var[:] = [1, 1, 0.5]
    var = root_grp.createVariable('lwp_error', 'f8', 'time')
    var[:] = [0.2, 0.2, 0.1]
    var = root_grp.createVariable('is_rain', 'i4', 'time')
    var[:] = [0, 1, 1]
    var = root_grp.createVariable('category_bits', 'i4', 'time')
    var[:] = [0, 1, 2]
    var = root_grp.createVariable('quality_bits', 'i4', 'time')
    var[:] = [8, 16, 32]
    var = root_grp.createVariable('temperature', 'f8', ('time', 'height'))
    var[:] = np.array([[282, 280, 278],
                       [286, 284, 282],
                       [284, 282, 280]])
    var = root_grp.createVariable('pressure', 'f8', ('time', 'height'))
    var[:] = np.array([[1010, 1000, 990],
                       [1020, 1010, 1000],
                       [1030, 1020, 1010]])
    root_grp.close()
    return file_name


def _create_dimensions(root_grp):
    n_dim = len(TEST_ARRAY)
    for dim_name in DIMENSIONS:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp):
    for dim_name in DIMENSIONS:
        x = root_grp.createVariable(dim_name, 'f8', (dim_name,))
        x[:] = TEST_ARRAY
        if dim_name == 'height':
            x.units = 'm'


def test_get_atmosphere_t(lwc_source_file):
    obj = LwcSource(lwc_source_file)
    compare = np.array([[282, 280, 278],
                        [286, 284, 282],
                        [284, 282, 280]])
    assert_array_equal(compare, obj.atmosphere[0])


def test_get_atmosphere_p(lwc_source_file):
    obj = LwcSource(lwc_source_file)
    compare = np.array([[1010, 1000, 990],
                        [1020, 1010, 1000],
                        [1030, 1020, 1010]])
    assert_array_equal(compare, obj.atmosphere[-1])


class LwcSourceObj:
    def __init__(self):
        self.dheight = 10
        self.categorize_bits = \
            CategorizeBits(category_bits={'droplet': np.asarray([[1, 0, 1],
                                                                 [0, 1, 1]], dtype=bool)},
                           quality_bits={'radar': np.asarray([[1, 0, 1],
                                                              [0, 1, 1]], dtype=bool),
                                         'lidar': np.asarray([[1, 0, 1],
                                                              [0, 1, 1]], dtype=bool)})
        self.atmosphere = (np.array([[282, 281, 280],
                                     [280, 279, 278]]),
                           np.array([[101000, 100500, 100000],
                                     [100000, 99500, 99000]]))
        self.lwp = np.array([2, 0])
        self.lwp_error = np.array([0.1, 0.2])
        self.is_rain = np.array([0, 1])


LWC_OBJ = Lwc(LwcSourceObj())
STATUS_OBJ = LwcStatus(LwcSourceObj(), LWC_OBJ)
ERROR_OBJ = LwcError(LwcSourceObj(), LWC_OBJ)


@pytest.mark.parametrize("value", [0, 1])
def test_get_liquid(value):
    assert value in LWC_OBJ.is_liquid


def test_init_lwc_adiabatic(lwc_source_file):
    lwc_source = LwcSourceObj()
    compare = atmos.fill_clouds_with_lwc_dz(lwc_source.atmosphere,
                                            LWC_OBJ.is_liquid)
    compare[0, 0] *= 10
    compare[0, 2] *= 10
    compare[1, 1] *= 10
    compare[1, 2] *= 20
    assert_array_almost_equal(LWC_OBJ._init_lwc_adiabatic(), compare)


def test_screen_rain_lwc():
    compare = np.ma.array([[5, 1, 2],
                           [3, 6, 0]],
                          mask=[[0, 0, 0],
                                [1, 1, 1]])
    assert_array_equal(compare.mask, LWC_OBJ.lwc.mask)


@pytest.mark.parametrize("value", [0, 1])
def test_init_status(value):
    assert value in STATUS_OBJ._init_status()


@pytest.mark.parametrize("key", ['radar', 'lidar'])
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
    STATUS_OBJ.status = np.array([[1, 0, 2], [0, 0, 2]])
    STATUS_OBJ._adjust_lwc(time, base)
    assert value in STATUS_OBJ.status


def test_has_converged():
    ind = 1
    assert STATUS_OBJ._has_converged(ind) is True


def test_out_of_bound():
    ind = 2
    assert STATUS_OBJ._out_of_bound(ind) is True


def test_find_adjustable_clouds():
    assert 1 not in STATUS_OBJ._find_adjustable_clouds()


def test_find_topmost_clouds():
    compare = np.asarray([[0, 0, 1], [0, 1, 1]], dtype=bool)
    assert_array_equal(STATUS_OBJ._find_topmost_clouds(), compare)


def test_find_echo_combinations_in_liquid():
    STATUS_OBJ.echo['lidar'] = np.array([[0, 1, 0], [1, 1, 0]])
    STATUS_OBJ.echo['radar'] = np.array([[0, 0, 0], [0, 1, 1]])
    STATUS_OBJ.is_liquid = np.array([[1, 1, 1], [0, 1, 1]])
    compare = np.array([[0, 1, 0], [0, 3, 2]])
    assert_array_equal(STATUS_OBJ._find_echo_combinations_in_liquid(), compare)


def test_find_lidar_only_clouds():
    inds = np.array([[1, 0, 0], [0, 1, 3]])
    compare = np.array([True, False])
    assert_array_equal(STATUS_OBJ._find_lidar_only_clouds(inds), compare)


def test_remove_good_profiles():
    top_c = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=bool)
    compare = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=bool)
    assert_array_equal(STATUS_OBJ._remove_good_profiles(top_c), compare)


def test_find_lwp_difference():
    STATUS_OBJ.lwc_adiabatic = np.array([[1, 8, 2], [2, 3, 7]])
    STATUS_OBJ.lwc_source.lwp = np.array([50, 30])
    compare = np.array([60, 90])
    assert_array_equal(STATUS_OBJ._find_lwp_difference(), compare)


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4])
def test_screen_rain_status(value):
    STATUS_OBJ.lwc_source.is_rain = np.array([0, 1])
    STATUS_OBJ.status = np.array([[0, 2, 2, 3, 1], [1, 3, 0, 2, 2]])
    STATUS_OBJ.screen_rain()
    assert value in STATUS_OBJ.status


def test_limit_error():
    error = np.array([[0, 0, 1], [0.2, 0.4, 0.3]])
    max_v = 0.5
    compare = np.array([[0, 0, 0.5], [0.2, 0.4, 0.3]])
    assert_array_equal(ERROR_OBJ._limit_error(error, max_v), compare)


def test_calc_lwc_gradient():
    from cloudnetpy.utils import l2norm
    ERROR_OBJ.lwc = np.ma.array([[0.1, 0.2, 0.3],
                                 [0.1, 0.3, 0.6]])
    compare = l2norm(*np.gradient(ERROR_OBJ.lwc))
    assert_array_almost_equal(ERROR_OBJ._calc_lwc_gradient(), compare)


def test_calc_lwc_relative_error():
    from cloudnetpy.utils import l2norm
    ERROR_OBJ.lwc = np.ma.array([[0.1, 0.2, 0.3],
                                 [0.1, 0.3, 0.6]])
    x = l2norm(*np.gradient(ERROR_OBJ.lwc))
    compare = x / ERROR_OBJ.lwc / 2
    compare[compare > 5] = 5
    assert_array_almost_equal(ERROR_OBJ._calc_lwc_relative_error(), compare)


def test_calc_lwp_relative_error():
    ERROR_OBJ.lwc_source.lwp = np.array([0.1, 0.5])
    ERROR_OBJ.lwc_source.lwp_error = np.array([0.2, 5.5])
    compare = ERROR_OBJ.lwc_source.lwp_error / ERROR_OBJ.lwc_source.lwp
    compare[compare > 10] = 10
    assert_array_equal(ERROR_OBJ._calc_lwp_relative_error(), compare)


def test_calc_combined_error():
    from cloudnetpy.utils import transpose, l2norm
    err_2d = np.array([[0, 0.1, 0.1], [0.2, 0.4, 0.15]])
    err_1d = np.array([0.3, 0.2])
    compare = l2norm(err_2d, transpose(err_1d))
    assert_array_equal(ERROR_OBJ._calc_combined_error(err_2d, err_1d), compare)


def test_fill_error_array():
    error_in = np.array([[0, 0.1, 0.1], [0.2, 0.4, 0.15]])
    ERROR_OBJ.lwc = np.ma.array([[0.1, 0.2, 0.1],
                               [0.1, 0.2, 0.2]],
                              mask=[[0, 1, 0],
                                    [1, 0, 0]])
    compare = np.ma.array([[0, 0, 0],
                           [0, 0, 0]],
                          mask=[[0, 1, 0],
                                [1, 0, 0]])
    ERROR_OBJ._fill_error_array(error_in)
    error = ERROR_OBJ._fill_error_array(error_in)
    assert_array_almost_equal(error.mask, compare.mask)


def test_screen_rain_error():
    compare = np.ma.array([[0.709, 0, 0.709],
                           [0, 0, 0]],
                          mask=[[0, 1, 0],
                                [1, 1, 1]])
    assert_array_equal(ERROR_OBJ.lwc_error.mask, compare.mask)


@pytest.mark.parametrize("key", ["lwc", "lwc_retrieval_status", "lwc_error"])
def test_append_data(lwc_source_file, key):
    from cloudnetpy.products.lwc import _append_data
    lwc_source = LwcSource(lwc_source_file)
    _append_data(lwc_source, LWC_OBJ, STATUS_OBJ, ERROR_OBJ)
    assert key in lwc_source.data.keys()
