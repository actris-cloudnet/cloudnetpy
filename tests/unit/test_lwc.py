import numpy as np
import numpy.ma as ma
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
        self.is_liquid = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]])
        self.dheight = np.array([1, 2, 3, 4, 5, 6, 7])
        self.categorize_bits = \
            CategorizeBits(category_bits={'droplet': np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]])},
                           quality_bits={'radar': np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]]),
                                         'lidar': np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1]])})
        self.atmosphere = [np.array([[282, 280, 278], [286, 284, 282], [284, 282, 280]]),
                           np.array([[1010, 1000, 990], [1020, 1010, 1000], [1030, 1020, 1010]])]
        self.lwp = np.array([1, 0, 1, 1, 2, 1, 1])
        self.lwp_error = np.array([0.2, 0.1, 0.1, 0.2, 0.1, 0.3, 0.1])
        self.is_rain = np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]])


def test_get_liquid():
    obj = Lwc(LwcSourceObj())
    assert 'droplet' in obj.is_liquid.keys()


def test_init_lwc_adiabatic():
    assert True


def test_adiabatic_lwc_to_lwc():
    assert True


def test_adjust_clouds_to_match_lwp():
    assert True


def test_screen_rain_lwc():
    assert True

