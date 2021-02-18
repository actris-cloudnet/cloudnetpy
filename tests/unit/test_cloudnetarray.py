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


@pytest.mark.parametrize("array, expected_data, expected_dtype", [
    (1, np.array([1]), 'i4'),
    (1.0, np.array([1.0]), 'f4'),
    ("1", np.array([1]), 'i4'),
    ("1.0", np.array([1.0]), 'f4'),
    (np.array([1]), np.array([1]), 'i4'),
    (np.array([1.0]), np.array([1.0]), 'f4'),
    (np.median([1.0, 1.0, 1.0]), np.array([1.0]), 'f4'),
])
def test_different_inputs(array, expected_data, expected_dtype):
    obj = CloudnetArray(array, 'test')
    assert obj.data_type == expected_dtype
    assert obj.data == expected_data
    assert isinstance(obj.data, np.ndarray)


def test_bad_input():
    with pytest.raises(ValueError):
        CloudnetArray((1, 2, 3), 'bad_input')
    with pytest.raises(ValueError):
        CloudnetArray([1, 2, 3], 'bad_input')


def test_masked_array():
    data = ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    obj = CloudnetArray(data, 'test')
    assert_array_equal(obj.data.data, data.data)
    assert_array_equal(obj.data.mask, data.mask)
    assert obj.data_type == 'f4'


class TestCloudnetArrayWithNc:

    @pytest.fixture(autouse=True)
    def init_tests(self, fake_nc_file):
        self.nc = netCDF4.Dataset(fake_nc_file)
        self.time = self.nc.variables['time']
        yield
        self.nc.close()

    def test_get_data(self):
        name, units = 'test_name', 'm s-1'
        obj = CloudnetArray(self.time, name, units)
        assert_array_equal(obj.data, self.time[:])
        assert obj.name == name
        assert obj.units == units
        assert obj.data_type == 'f4'

    def test_lin2db(self):
        name, units = 'test_name', 'm s-1'
        obj = CloudnetArray(self.time, name, units)
        obj.lin2db()
        assert obj.units == 'dB'
        assert_array_equal(obj.data, utils.lin2db(self.time[:]))

    def test_db2lin(self):
        name, units = 'test_name', 'dB'
        obj = CloudnetArray(self.time, name, units)
        obj.db2lin()
        assert obj.units == ''
        assert_array_equal(obj.data, utils.db2lin(self.time[:]))

    def test_mask_indices(self):
        obj = CloudnetArray(self.time, 'test_name')
        obj.mask_indices([0, 1])
        result = ma.array([0, 1, 2, 3, 4], mask=[1, 1, 0, 0, 0])
        assert_array_equal(obj.data.data, result.data)
        assert_array_equal(obj.data.mask, result.mask)

    def test_fetch_attributes(self):
        obj = CloudnetArray(self.time, 'test_name')
        obj.attr1 = 'a'
        obj.attr2 = 'b'
        obj.attr3 = 'c'
        result = ['units', 'attr1', 'attr2', 'attr3']
        assert obj.fetch_attributes() == result

    def test_set_attributes(self):
        obj = CloudnetArray(self.time, 'test_name')
        meta = MetaData(long_name='the long name', units='g m-3')
        obj.set_attributes(meta)
        for key, value in zip(['long_name', 'units'], ['the long name', 'g m-3']):
            assert hasattr(obj, key)
            assert getattr(obj, key) == value

    def test_rebin_data(self):
        test_var = self.nc.variables['2d_array']
        obj = CloudnetArray(test_var, 'test_name')
        time = np.array([1, 2, 3, 4, 5])
        time_new = np.array([2.1, 4.1])
        obj.rebin_data(time, time_new)
        result = np.array([[2.5, 2.5, 2.5, 1],
                           [4.5, 4.5, 4.5, 1]])
        assert_array_equal(obj.data, result)

    def test_rebin_data_2d(self):
        test_var = self.nc.variables['2d_array']
        obj = CloudnetArray(test_var, 'test_name')
        time = np.array([1, 2, 3, 4, 5])
        time_new = np.array([2.1, 4.1])
        height = np.array([1, 2, 3, 4])
        height_new = np.array([2.1, 4.1])
        obj.rebin_data(time, time_new, height, height_new)
        result = np.array([[2.5, 1],
                           [4.5, 1]])
        assert_array_equal(obj.data, result)
