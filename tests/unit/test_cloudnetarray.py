import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import netCDF4
import pytest
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy import utils
from cloudnetpy.metadata import MetaData


@pytest.fixture(scope='session')
def fake_nc_file(tmpdir_factory):
    """Creates a simple categorize for testing."""
    file_name = tmpdir_factory.mktemp("data").join("nc_file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_time, n_height = 5, 4
    root_grp.createDimension('time', n_time)
    root_grp.createDimension('height', n_height)
    var = root_grp.createVariable('time', 'f8', 'time')
    var[:] = np.arange(n_time)
    var = root_grp.createVariable('height', 'f8', 'height')
    var[:] = np.arange(n_height)
    var = root_grp.createVariable('var_float_scalar', 'f8')
    var[:] = 1.0
    var = root_grp.createVariable('2d_array', 'f8', ('time', 'height'))
    var[:] = np.array([[1, 1, 1, 1],
                       [2, 2, 2, 1],
                       [3, 3, 3, 1],
                       [4, 4, 4, 1],
                       [5, 5, 5, 1]])
    root_grp.close()
    return file_name


def test_get_data(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['time']
    name, units = 'test_name', 'm s-1'
    obj = CloudnetArray(test_var, name, units)
    assert_array_equal(obj.data, test_var[:])
    assert obj.name == name
    assert obj.units == units
    assert obj.data_type == 'f4'


def test_lin2db(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['time']
    name, units = 'test_name', 'm s-1'
    obj = CloudnetArray(test_var, name, units)
    obj.lin2db()
    assert obj.units == 'dB'
    assert_array_equal(obj.data, utils.lin2db(test_var[:]))


def test_db2lin(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['time']
    name, units = 'test_name', 'dB'
    obj = CloudnetArray(test_var, name, units)
    obj.db2lin()
    assert obj.units == ''
    assert_array_equal(obj.data, utils.db2lin(test_var[:]))


def test_mask_indices(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['time']
    obj = CloudnetArray(test_var, 'test_name')
    obj.mask_indices([0, 1])
    result = ma.array([0, 1, 2, 3, 4], mask=[1, 1, 0, 0, 0])
    assert_array_equal(obj.data.data, result.data)
    assert_array_equal(obj.data.mask, result.mask)


def test_fetch_attributes(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['time']
    obj = CloudnetArray(test_var, 'test_name')
    obj.attr1 = 'a'
    obj.attr2 = 'b'
    obj.attr3 = 'c'
    result = ['units', 'attr1', 'attr2', 'attr3']
    for attr in obj.fetch_attributes():
        assert attr in result


def test_set_attributes(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['time']
    obj = CloudnetArray(test_var, 'test_name')
    meta = MetaData(long_name='the long name', units='g m-3')
    obj.set_attributes(meta)
    for key, value in zip(['long_name', 'units'], ['the long name', 'g m-3']):
        assert hasattr(obj, key)
        assert getattr(obj, key) == value


def test_rebin_data(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['2d_array']
    obj = CloudnetArray(test_var, 'test_name')
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([2.1, 4.1])
    obj.rebin_data(time, time_new)
    result = np.array([[2.5, 2.5, 2.5, 1],
                       [4.5, 4.5, 4.5, 1]])
    assert_array_equal(obj.data, result)


def test_rebin_data_2d(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['2d_array']
    obj = CloudnetArray(test_var, 'test_name')
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([2.1, 4.1])
    height = np.array([1, 2, 3, 4])
    height_new = np.array([2.1, 4.1])
    obj.rebin_data(time, time_new, height, height_new)
    result = np.array([[2.5, 1],
                       [4.5, 1]])
    assert_array_equal(obj.data, result)
