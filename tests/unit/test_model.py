import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
import netCDF4
from cloudnetpy.categorize import model


@pytest.mark.parametrize("input, result", [
    ('this_is_a_ecmwf_model_file.nc', 'ecmwf'),
    ('a_gdasXYZ_model_file.nc', 'gdas'),
    ('an_unknown.nc', '')

])
def test_find_model_type(input, result):
    assert model._find_model_type(input) == result


def test_calc_mean_height():
    height = np.array([[0, 1, 2, 3, 4],
                       [0.2, 1.2, 2.2, 3.2, 4.2],
                       [-0.2, 0.8, 1.8, 2.8, 3.8]])
    result = np.array([0, 1, 2, 3, 4])
    assert_array_equal(model._calc_mean_height(height), result)


BIG_ARRAY = np.ones((2, 3, 4))

EMPTY_ARRAY = ma.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],
                       mask=[[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=float)
MASKED_ONE = ma.copy(EMPTY_ARRAY)
MASKED_ONE[0, 0] = ma.masked

ALT_SITE = 123


@pytest.fixture(scope='session')
def fake_model_file(tmpdir_factory):
    """Creates a simple radar file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("ecmwf_file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_time, n_height = 3, 4
    root_grp.createDimension('time', n_time)
    root_grp.createDimension('height', n_height)
    root_grp.createDimension('wl', 2)
    root_grp.createVariable('time', 'f8', 'time')[:] = np.arange(n_time)
    var = root_grp.createVariable('height', 'f8', ('time', 'height'))
    var[:] = np.array([[1000, 2000, 3000, 4000],
                       [1000, 2000, 3000, 4000],
                       [1000, 2000, 3000, 4000]])
    var.units = 'm'
    _create_var(root_grp, 'temperature')
    _create_var(root_grp, 'pressure')
    _create_var(root_grp, 'uwind')
    _create_var(root_grp, 'vwind')
    _create_var(root_grp, 'q')
    _create_var(root_grp, 'rh')
    _create_var(root_grp, 'gas_atten', data=BIG_ARRAY, dim=('wl', 'time', 'height'))
    _create_var(root_grp, 'specific_gas_atten', data=BIG_ARRAY, dim=('wl', 'time', 'height'))
    _create_var(root_grp, 'specific_saturated_gas_atten', data=BIG_ARRAY, dim=('wl', 'time', 'height'))
    _create_var(root_grp, 'liquid_atten', data=BIG_ARRAY, dim=('wl', 'time', 'height'))
    _create_var(root_grp, 'specific_liquid_atten', data=BIG_ARRAY, dim=('wl', 'time', 'height'))
    root_grp.close()
    return file_name


def _create_var(root_grp, name, dim=('time', 'height'), data=EMPTY_ARRAY):
    var = root_grp.createVariable(name, 'f8', dim)
    var[:] = data
    var.units = 'g'


def test_get_model_heights(fake_model_file):
    obj = model.Model(str(fake_model_file), ALT_SITE)
    assert_array_equal(obj.model_heights, ALT_SITE + np.array([[1000, 2000, 3000, 4000],
                                                               [1000, 2000, 3000, 4000],
                                                               [1000, 2000, 3000, 4000]]))


def test_mean_height(fake_model_file):
    obj = model.Model(str(fake_model_file), ALT_SITE)
    assert_array_equal(obj.mean_height, ALT_SITE + np.array([1000, 2000, 3000, 4000]))


def test_interpolate_to_common_height(fake_model_file):
    obj = model.Model(str(fake_model_file), ALT_SITE)
    radar_wl_band = 0
    obj.interpolate_to_common_height(radar_wl_band)
    for key in ('uwind', 'vwind', 'q', 'temperature', 'pressure', 'rh',
                'gas_atten', 'specific_gas_atten', 'specific_saturated_gas_atten',
                'specific_liquid_atten'):
        assert key in obj.data_sparse


def test_interpolate_to_grid(fake_model_file):
    obj = model.Model(str(fake_model_file), ALT_SITE)
    radar_wl_band = 0
    obj.interpolate_to_common_height(radar_wl_band)
    time_grid = np.array([1,3])
    height_grid = np.array([1, 3])
    obj.interpolate_to_grid(time_grid, height_grid)
    assert_array_equal(obj.height, height_grid)
    assert hasattr(obj, 'data_dense')
