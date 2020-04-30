""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import jenoptik
import pytest
import numpy as np
from numpy.testing import assert_array_equal
import netCDF4


@pytest.fixture
def fake_jenoptik_file(tmpdir):
    """Creates a simple categorize for testing."""
    file_name = tmpdir.join('jenoptik.nc')
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_time, n_height = 3, 4
    root_grp.createDimension('time', n_time)
    root_grp.createDimension('range', n_height)
    var = root_grp.createVariable('time', 'f8', 'time')
    var[:] = 3600*np.arange(1, 4)
    var = root_grp.createVariable('range', 'f8', 'range')
    var[:] = np.array([2000, 3000, 4000, 5000])
    var.units = 'm'
    var = root_grp.createVariable('nn1', 'f8', 'time')
    nn1 = 142
    var[:] = np.ones(3)*nn1
    root_grp.createVariable('zenith', 'f8')[:] = 2
    root_grp.year = '2019'
    root_grp.month = '5'
    root_grp.day = '23'
    root_grp.close()
    return file_name


site_name = 'Mace Head'


def test_calc_range(fake_jenoptik_file):
    obj = jenoptik.JenoptikCeilo(fake_jenoptik_file, site_name)
    assert_array_equal(obj._calc_range(), [1500, 2500, 3500, 4500])


def test_convert_time(fake_jenoptik_file):
    obj = jenoptik.JenoptikCeilo(fake_jenoptik_file, site_name)
    assert_array_equal(obj._convert_time(), [1, 2, 3])


def test_convert_time_error(fake_jenoptik_file):
    root_grp = netCDF4.Dataset(fake_jenoptik_file, "a")
    root_grp.variables['time'][:] = np.array([1, 0, 3])
    root_grp.close()
    obj = jenoptik.JenoptikCeilo(fake_jenoptik_file, site_name)
    with pytest.raises(RuntimeError):
        obj._convert_time()


def test_read_date(fake_jenoptik_file):
    obj = jenoptik.JenoptikCeilo(fake_jenoptik_file, site_name)
    assert_array_equal(obj._read_date(), ('2019', '5', '23'))


def test_read_metadata(fake_jenoptik_file):
    obj = jenoptik.JenoptikCeilo(fake_jenoptik_file, site_name)
    assert obj._read_metadata()['tilt_angle'] == 2
