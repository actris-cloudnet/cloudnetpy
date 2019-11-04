import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
import netCDF4
from cloudnetpy.categorize.radar import Radar

FOLDING_VELOCITY = 8.7


@pytest.fixture(scope='session')
def fake_radar_file(tmpdir_factory):
    """Creates a simple radar file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("radar_file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_time, n_height = 3, 4
    root_grp.createDimension('time', n_time)
    root_grp.createDimension('height', n_height)
    root_grp.createVariable('time', 'f8', 'time')[:] = np.arange(n_time)
    var = root_grp.createVariable('height', 'f8', 'height')
    var[:] = np.arange(n_height)
    var.units = 'km'
    root_grp.createVariable('radar_frequency', 'f8')[:] = 35.5
    root_grp.createVariable('nyquist_velocity', 'f8')[:] = FOLDING_VELOCITY
    var = root_grp.createVariable('v', 'f8', ('time', 'height'))
    var[:] = ma.zeros((n_time, n_height))
    var = root_grp.createVariable('width', 'f8', ('time', 'height'))
    var[:] = ma.zeros((n_time, n_height))
    var = root_grp.createVariable('ldr', 'f8', ('time', 'height'))
    var[:] = ma.zeros((n_time, n_height))
    var = root_grp.createVariable('Zh', 'f8', ('time', 'height'))
    var[:] = ma.zeros((n_time, n_height))
    root_grp.close()
    return file_name


def test_sequence_indices(fake_radar_file):
    obj = Radar(fake_radar_file)
    assert_array_equal(obj.sequence_indices, [[0, 1, 2, 3]])


def test_wl_band(fake_radar_file):
    obj = Radar(fake_radar_file)
    assert obj.wl_band == 0


def test_folding_velocity(fake_radar_file):
    obj = Radar(fake_radar_file)
    assert obj.folding_velocity == FOLDING_VELOCITY


def test_source(fake_radar_file):
    obj = Radar(fake_radar_file)
    assert obj.source == ''


def test_correct_atten(fake_radar_file):
    obj = Radar(fake_radar_file)
    atten = {'radar_gas_atten': np.ones((3, 4)),
             'radar_liquid_atten': np.array([[0, 1, 1, 1],
                                             [0, 1, 1, 1],
                                             [0, 0, 0, 0]])}
    obj.correct_atten(atten)
    z = obj.data['Z'][:]
    result = np.array([[1, 2, 2, 2],
                       [1, 2, 2, 2],
                       [1, 1, 1, 1]])
    assert_array_equal(z.data, result)
