import numpy as np
from numpy import testing
from collections import namedtuple
import pytest
import netCDF4
from cloudnetpy.products.lwc import LwcSource, Lwc, LwcStatus, CalculateError


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


def test_get_atmosphere_T(lwc_source_file):
    obj = LwcSource(lwc_source_file)
    compare = np.array([[282, 280, 278],
                        [286, 284, 282],
                        [284, 282, 280]])
    testing.assert_array_equal(compare, obj.atmosphere[0])


def test_get_atmosphere_p(lwc_source_file):
    obj = LwcSource(lwc_source_file)
    compare = np.array([[1010, 1000, 990],
               [1020, 1010, 1000],
               [1030, 1020, 1010]])
    testing.assert_array_equal(compare, obj.atmosphere[-1])


class LwcSourceObj:
    def __init__(self):
        self.is_liquid = np.asarray([[1, 0, 1], [0, 1, 0], [1, 1, 1],
                                     [0, 0, 1], [0, 1, 1]], dtype=bool)
        self.dheight = 10
        self.categorize_bits = \
            CategorizeBits(category_bits={'droplet': np.asarray([[1, 0, 1], [0, 1, 0],
                                                                 [1, 1, 1], [0, 0, 1],
                                                                 [0, 1, 1]], dtype=bool)},
                           quality_bits={'radar': np.asarray([[1, 0, 1], [0, 1, 0],
                                                              [1, 1, 1], [0, 0, 1],
                                                              [0, 1, 1]], dtype=bool),
                                         'lidar': np.asarray([[1, 0, 1], [0, 1, 0],
                                                              [1, 1, 1], [0, 0, 1],
                                                              [0, 1, 1]], dtype=bool)})
        self.atmosphere = [np.array([[282, 281, 280], [280, 279, 278],
                                     [286, 285, 284], [284, 283, 282],
                                     [284, 283, 282]]),
                           np.array([[1010, 1005, 1000], [1000, 995, 990],
                                     [1020, 1015, 1010], [1100, 1005, 1000],
                                     [1030, 1025, 1020]])]
        self.lwp = np.array([1, 0, 2, 2, 1])
        self.lwp_error = np.array([0.1, 0.2, 0.1, 0.3, 0.0])
        self.is_rain = np.array([1, 0, 0, 1, 1])


LWC_OBJ = Lwc(LwcSourceObj())
STATUS_OBJ = LwcStatus(LwcSourceObj(), LWC_OBJ)
ERROR_OBJ = CalculateError(LwcSourceObj(), LWC_OBJ)


@pytest.mark.parametrize("value", [0, 1])
def test_get_liquid(value):
    assert value in LWC_OBJ.is_liquid


def test_init_lwc_adiabatic():
    compare = np.array([[-0.0002, 0.0, 0.0], [0.0, 0.0001, 0.0],
                        [-0.0022, -0.0044, -0.0065], [0.0, 0.0, -0.0002],
                        [0.0, -0.0003, -0.0006]])
    testing.assert_equal(np.around(LWC_OBJ._init_lwc_adiabatic(),
                                   decimals=4), compare)


def test_adiabatic_lwc_to_lwc():
    compare = np.array([[0.106, 0.0, -0.006], [0.0, 0.0, 0.0],
                        [0.0333, 0.0667, 0.1], [0.0, 0.0, 0.2],
                        [0.0, 0.0333, 0.0667]])
    testing.assert_equal(np.around(LWC_OBJ._adiabatic_lwc_to_lwc(),
                                   decimals=4), compare)


def test_screen_rain_lwc():
    compare = np.ma.array([[-1, -1, -1], [0, 0, 0], [0.033, 0.067, 0.1],
                           [-1, -1, -1], [-1, -1, -1]], mask=-1)
    testing.assert_equal(np.around(LWC_OBJ.lwc, decimals=3), compare)


@pytest.mark.parametrize("value", [0, 1])
def test_init_status(value):
    assert value in STATUS_OBJ._init_status()


@pytest.mark.parametrize("key", ['radar', 'lidar'])
def test_get_echo(key):
    assert key in STATUS_OBJ.echo.keys()


@pytest.mark.parametrize("value", [0, 1, 2])
def test_update_status(value):
    time = np.array([1, 3])
    STATUS_OBJ._update_status(time)
    assert value in STATUS_OBJ.status


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_adjust_lwc(value):
    time = 0
    base = 0
    STATUS_OBJ.status = np.array([[1, 0, 0, 0, 2], [0, 0, 1, 0, 2]])
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
    compare = np.asarray([[0, 0, 1], [0, 1, 0], [0, 0, 1],
                          [0, 0, 1], [0, 1, 1]], dtype=bool)
    testing.assert_array_equal(STATUS_OBJ._find_topmost_clouds(), compare)


def test_find_echo_combinations_in_liquid():
    STATUS_OBJ.echo['lidar'] = np.array([[0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    STATUS_OBJ.echo['radar'] = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]])
    STATUS_OBJ.is_liquid = np.array([[1, 0, 1, 1, 1], [0, 0, 1, 1, 1]])
    compare = np.array([[0, 0, 1, 1, 0], [0, 0, 3, 2, 2]])
    testing.assert_equal(STATUS_OBJ._find_echo_combinations_in_liquid(), compare)


def test_find_lidar_only_clouds():
    inds = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 1, 3]])
    compare = np.array([True, False])
    testing.assert_equal(STATUS_OBJ._find_lidar_only_clouds(inds), compare)


def test_remove_good_profiles():
    top_c = np.asarray([[0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool)
    compare = np.asarray([[0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)
    testing.assert_equal(STATUS_OBJ._remove_good_profiles(top_c), compare)


def test_find_lwp_difference():
    STATUS_OBJ.lwc_adiabatic = np.array([[1, 8, 2], [2, 3, 7], [3, 0, 1], [2, 2, 8], [9, 0, 1]])
    STATUS_OBJ.lwc_source.lwp = np.array([50, 30, 70, 10, 40])
    compare = np.array([60, 90, -30, 110, 60])
    testing.assert_equal(STATUS_OBJ._find_lwp_difference(), compare)


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4])
def test_screen_rain_status(value):
    STATUS_OBJ.lwc_source.is_rain = np.array([0, 1])
    STATUS_OBJ.status = np.array([[0, 2, 2, 3, 1], [1, 3, 0, 2, 2]])
    STATUS_OBJ.screen_rain()
    assert value in STATUS_OBJ.status


def test_calculate_lwc_error():
    compare = np.ma.array([[0, 0, 0], [0, 0, 0], [0.502, 0.255, 0.174],
                           [0, 0, 0], [0, 0, 0]], mask=0)
    testing.assert_equal(np.around(ERROR_OBJ.calculate_lwc_error(),
                                   decimals=3), compare)


def test_limit_error():
    error = np.array([[0, 0, 1], [0.2, 0.4, 0.3], [1, 0.9, 0.3],
                      [1, 1, 1], [0, 0.1, 0.2]])
    max_v = 0.5
    compare = np.array([[0, 0, 0.5], [0.2, 0.4, 0.3], [0.5, 0.5, 0.3],
                        [0.5, 0.5, 0.5], [0, 0.1, 0.2]])
    testing.assert_equal(ERROR_OBJ._limit_error(error, max_v), compare)


def test_calc_lwc_gradient():
    compare = np.array([[0.0, 0.0, 0.0], [0.02, 0.03, 0.05],
                        [0.03, 0.03, 0.03], [0.02, 0.03, 0.05],[0.0, 0.0, 0.0]])
    testing.assert_equal(np.around(ERROR_OBJ._calc_lwc_gradient(), decimals=2),
                         compare)


def test_calc_lwc_relative_error():
    compare = np.ma.array([[0, 0, 0], [0, 0, 0], [0.5, 0.25, 0.17],
                           [0, 0, 0], [0, 0, 0]], mask=0)
    testing.assert_equal(np.around(ERROR_OBJ._calc_lwc_relative_error(), decimals=2),
                         compare)


def test_calc_lwp_relative_error():
    compare = np.array([0.1, 10.0, 0.05, 0.15, 0.0])
    testing.assert_equal(ERROR_OBJ._calc_lwp_relative_error(), compare)


def test_calc_combined_error():
    err_2d = np.array([[0, 0.1, 0.1], [0.2, 0.4, 0.15], [0, 0.3, 0.1],
                       [0.1, 0.5, 0.5], [0, 0.2, 0.4]])
    err_1d = np.array([0.3, 0.2, 0.6, 0, 0.1])
    compare = np.array([[0.3, 0.316, 0.316], [0.283, 0.447, 0.25],
                        [0.6, 0.671, 0.608], [0.1, 0.5, 0.5], [0.1, 0.224, 0.412]])
    testing.assert_equal(np.around(ERROR_OBJ._calc_combined_error(err_2d, err_1d),
                                   decimals=3), compare)


def test_fill_error_array():
    error_in = np.array([[0, 0.1, 0.1], [0.2, 0.4, 0.15], [0, 0.3, 0.1],
                       [0.1, 0.5, 0.5], [0, 0.2, 0.4]])
    compare = np.ma.array([[0, 0, 0], [0, 0, 0], [0.0, 0.3, 0.1],
                           [0, 0, 0], [0, 0, 0]], mask=0)
    testing.assert_equal(ERROR_OBJ._fill_error_array(error_in), compare)


def test_screen_rain_error():
    compare = np.ma.array([[0, 0, 0], [0, 0, 0], [0.502, 0.255, 0.174],
                           [0, 0, 0], [0, 0, 0]], mask=0)
    testing.assert_equal(np.around(ERROR_OBJ.lwc_error, decimals=3), compare)


@pytest.mark.parametrize("key", ["lwc", "lwc_retrieval_status", "lwc_error"])
def test_append_data(lwc_source_file, key):
    from cloudnetpy.products.lwc import _append_data
    lwc_source = LwcSource(lwc_source_file)
    _append_data(lwc_source, LWC_OBJ, STATUS_OBJ, ERROR_OBJ)
    assert key in lwc_source.data.keys()
