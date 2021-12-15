import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
import netCDF4
from cloudnetpy.categorize.lidar import Lidar

WAVELENGTH = 900.0


@pytest.fixture(scope='session')
def fake_lidar_file(tmpdir_factory):
    """Creates a simple lidar file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("radar_file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_time, n_height = 4, 4
    root_grp.createDimension('time', n_time)
    root_grp.createDimension('height', n_height)
    root_grp.createVariable('time', 'f8', 'time')[:] = np.arange(n_time)
    var = root_grp.createVariable('height', 'f8', 'height')
    var[:] = np.arange(n_height)
    var.units = 'km'
    root_grp.createVariable('wavelength', 'f8')[:] = WAVELENGTH
    root_grp.createVariable('latitude', 'f8')[:] = 60.43
    root_grp.createVariable('longitude', 'f8')[:] = 25.4
    var = root_grp.createVariable('altitude', 'f8')
    var[:] = 120.3
    var.units = 'm'
    var = root_grp.createVariable('beta', 'f8', ('time', 'height'))
    var[:] = ma.array([[1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [1, 2, 3, 4]], dtype=float,
                      mask=[[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
    root_grp.close()
    return file_name


def test_init(fake_lidar_file):
    obj = Lidar(fake_lidar_file)
    assert obj.data['lidar_wavelength'].data == WAVELENGTH
    assert obj.data['beta_bias'].data == 3
    assert obj.data['beta_error'].data == 0.5


def test_rebin(fake_lidar_file):
    obj = Lidar(fake_lidar_file)
    time_new = np.array([1.1, 2.1])
    height_new = np.array([505, 1501])
    ind = obj.interpolate_to_grid(time_new, height_new)
    result = np.array([[2, 3],
                       [2, 3]])
    assert_array_equal(obj.data['beta'].data, result)
    assert ind == [0, 1]
