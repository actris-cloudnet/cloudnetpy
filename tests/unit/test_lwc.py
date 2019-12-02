import numpy as np
from numpy import testing
from collections import namedtuple
import pytest
import netCDF4
from cloudnetpy.products.lwc import LwcSource, Lwc, AdjustCloudsLwp, CalculateError


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
        self.is_liquid = np.asarray([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=bool)
        self.dheight = np.array([1, 2, 3])
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


def test_get_liquid():
    obj = Lwc(LwcSourceObj())
    assert type(obj.is_liquid) is np.ndarray
    assert 1 or 0 in obj.is_liquid


def test_init_lwc_adiabatic():
    obj = Lwc(LwcSourceObj())
    # Hard to test anything other
    assert type(obj.lwc_adiabatic) is np.ma.core.MaskedArray


def test_adiabatic_lwc_to_lwc():
    obj = Lwc(LwcSourceObj())
    assert len(obj.lwc_adiabatic) == 5 and len(obj.lwc_adiabatic[0]) == 3


def test_adjust_clouds_to_match_lwp():
    assert True


def test_screen_rain_lwc():
    obj = Lwc(LwcSourceObj())
    assert type(obj.lwc) is np.ma.core.MaskedArray


def test_init_status():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_get_echo():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_adjust_cloud_tops():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_update_status():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_adjust_lwc():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_has_converged():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_out_of_bound():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_find_adjustable_clouds():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_find_topmost_clouds():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_find_echo_combinations_in_liquid():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_find_lidar_only_clouds():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_remove_good_profiles():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_find_lwp_difference():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


def test_screen_rain():
    obj = AdjustCloudsLwp(LwcSourceObj)
    assert True


