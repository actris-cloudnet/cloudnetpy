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
    n_time = 3
    root_grp.createDimension('time', n_time)
    var = root_grp.createVariable('var_float_array', 'f8', 'time')
    var[:] = np.arange(n_time)
    var = root_grp.createVariable('var_float_scalar', 'f8')
    var[:] = 1.0
    root_grp.close()
    return file_name


def test_get_data(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['var_float_array']
    name, units = 'test_name', 'm s-1'
    obj = CloudnetArray(test_var, name, units)
    assert_array_equal(obj.data, test_var[:])
    assert obj.name == name
    assert obj.units == units
    assert obj.data_type == 'f4'


def test_lin2db(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['var_float_array']
    name, units = 'test_name', 'm s-1'
    obj = CloudnetArray(test_var, name, units)
    obj.lin2db()
    assert obj.units == 'dB'
    assert_array_equal(obj.data, utils.lin2db(test_var[:]))


def test_db2lin(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['var_float_array']
    name, units = 'test_name', 'dB'
    obj = CloudnetArray(test_var, name, units)
    obj.db2lin()
    assert obj.units == ''
    assert_array_equal(obj.data, utils.db2lin(test_var[:]))


def test_mask_indices(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['var_float_array']
    obj = CloudnetArray(test_var, 'test_name')
    obj.mask_indices([0, 1])
    result = ma.array([0, 1, 2], mask=[1, 1, 0])
    assert_array_equal(obj.data.data, result.data)
    assert_array_equal(obj.data.mask, result.mask)


def test_fetch_attributes(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['var_float_array']
    obj = CloudnetArray(test_var, 'test_name')
    obj.attr1 = 'a'
    obj.attr2 = 'b'
    obj.attr3 = 'c'
    result = ['units', 'attr1', 'attr2', 'attr3']
    for attr in obj.fetch_attributes():
        assert attr in result


def test_set_attributes(fake_nc_file):
    nc = netCDF4.Dataset(fake_nc_file)
    test_var = nc.variables['var_float_array']
    obj = CloudnetArray(test_var, 'test_name')
    meta = MetaData(long_name='the long name', units='g m-3')
    obj.set_attributes(meta)
    for key, value in zip(['long_name', 'units'], ['the long name', 'g m-3']):
        assert hasattr(obj, key)
        assert getattr(obj, key) == value

