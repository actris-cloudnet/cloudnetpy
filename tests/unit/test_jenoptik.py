""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import jenoptik
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
    root_grp.createVariable('zenith', 'f8')[:] = 2
    root_grp.year = '2021'
    root_grp.month = '2'
    root_grp.day = '21'
    root_grp.close()
    return file_name


class TestCHM15k:

    site_name = 'Mace Head'
    date = '2021-02-21'

    @pytest.fixture(autouse=True)
    def init_tests(self, fake_jenoptik_file):
        self.file = fake_jenoptik_file
        self.obj = jenoptik.JenoptikCeilo(fake_jenoptik_file, self.site_name, self.date)
        self.obj.backscatter = np.random.rand(5, 4)

    def test_calc_range(self):
        assert_array_equal(self.obj._calc_range(), [1500, 2500, 3500, 4500])

    def test_convert_time(self):
        self.obj.backscatter = np.random.rand(5, 4)
        time = self.obj._fetch_time()
        assert len(time) == 4
        assert all(np.diff(time) > 0)

    def test_read_date(self):
        assert_array_equal(self.obj._read_date(), self.date.split('-'))

    def test_read_metadata(self):
        assert self.obj._read_metadata()['tilt_angle'] == 2

    def test_convert_time_error(self):
        obj = jenoptik.JenoptikCeilo(self.file, self.site_name, '2022-01-01')
        obj.backscatter = np.random.rand(5, 4)
        with pytest.raises(ValueError):
            obj._fetch_time()
