""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import lufft
import pytest
import numpy as np
from numpy.testing import assert_array_equal
import netCDF4


@pytest.fixture
def fake_jenoptik_file(tmpdir):
    file_name = tmpdir.join('jenoptik.nc')
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_time, n_height = 5, 4
    root_grp.createDimension('time', n_time)
    root_grp.createDimension('range', n_height)
    var = root_grp.createVariable('time', 'f8', 'time')
    var[:] = [3696730788, 3696728448, 3696728447, 3696728450, 3696896790]
    var.units = 'seconds since 1904-01-01 00:00:00.000 00:00'
    var = root_grp.createVariable('range', 'f8', 'range')
    var[:] = np.array([2000, 3000, 4000, 5000])
    var.units = 'm'
    var = root_grp.createVariable('beta_raw', 'f8', ('time', 'range'))
    var[:] = np.random.rand(5, 4)
    var.units = 'sr-1 m-1'
    root_grp.createVariable('zenith', 'f8')[:] = 2
    root_grp.year = '2021'
    root_grp.month = '2'
    root_grp.day = '21'
    root_grp.close()
    return file_name


class TestCHM15k:

    date = '2021-02-21'

    @pytest.fixture(autouse=True)
    def init_tests(self, fake_jenoptik_file):
        self.file = fake_jenoptik_file
        self.obj = lufft.LufftCeilo(fake_jenoptik_file, self.date)
        self.obj.read_ceilometer_file()

    def test_calc_range(self):
        assert_array_equal(self.obj.data['range'], [1500, 2500, 3500, 4500])

    def test_convert_time(self):
        assert len(self.obj.data['time']) == 4
        assert all(np.diff(self.obj.data['time']) > 0)

    def test_read_date(self):
        assert_array_equal(self.obj.metadata['date'], self.date.split('-'))

    def test_read_metadata(self):
        assert self.obj.data['tilt_angle'] == 2

    def test_convert_time_error(self):
        obj = lufft.LufftCeilo(self.file, '2122-01-01')
        with pytest.raises(ValueError):
            obj.read_ceilometer_file()
