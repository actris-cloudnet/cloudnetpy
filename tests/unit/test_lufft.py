""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import lufft, ceilo2nc
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import netCDF4
import os
import glob
from cloudnetpy import concat_lib

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


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
        assert self.obj.data['zenith_angle'] == 2

    def test_convert_time_error(self):
        obj = lufft.LufftCeilo(self.file, '2122-01-01')
        with pytest.raises(ValueError):
            obj.read_ceilometer_file()


class TestWithRealData:

    site_meta = {
        'name': 'Bucharest',
        'altitude': 123,
        'latitude': 45.0,
        'longitude': 22.0
    }
    files = glob.glob(f'{SCRIPT_PATH}/data/chm15k/*.nc')

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.output = 'dummy_output_file.nc'
        self.filename = 'dummy_daily_file.nc'
        concat_lib.concatenate_files(self.files, self.filename)
        yield
        os.remove(self.filename)
        os.remove(self.output)

    def test_variables(self):
        ceilo2nc(self.filename, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        for key in ('beta', 'beta_smooth', 'calibration_factor', 'range', 'height', 'zenith_angle',
                    'time', 'beta_raw'):
            assert key in nc.variables
        for key in ('depolarisation_raw', 'depolarisation'):
            assert key not in nc.variables
        for key in ('altitude', 'latitude', 'longitude'):
            assert nc.variables[key][:] == self.site_meta[key]
        assert nc.variables['wavelength'][:] == 1064
        assert nc.variables['zenith_angle'][:] == 0
        assert nc.variables['zenith_angle'].units == 'degree'
        assert_array_almost_equal(nc.variables['height'][:] - self.site_meta['altitude'],
                                  nc.variables['range'][:], decimal=3)
        assert np.all(np.diff(nc.variables['time'][:]) > 0)
        assert nc.variables['beta'].units == 'sr-1 m-1'
        assert nc.variables['beta_smooth'].units == 'sr-1 m-1'
        assert nc.variables['zenith_angle'].dtype == 'float32'
        assert nc.variables['latitude'].units == 'degree_north'
        assert nc.variables['longitude'].units == 'degree_east'
        assert nc.variables['altitude'].units == 'm'
        nc.close()

    def test_global_attributes(self):
        uuid = ceilo2nc(self.filename, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        assert nc.source == 'Lufft CHM15k ceilometer'
        assert nc.location == self.site_meta['name']
        assert nc.title == f'Lidar file from {self.site_meta["name"]}'
        assert nc.file_uuid == uuid
        assert nc.cloudnet_file_type == 'lidar'
        assert nc.year == '2020'
        assert nc.month == '10'
        assert nc.day == '22'
        nc.close()

    def test_date_argument(self):
        ceilo2nc(self.filename, self.output, self.site_meta, date='2020-10-22')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 20
        assert nc.year == '2020'
        assert nc.month == '10'
        assert nc.day == '22'
        nc.close()
        with pytest.raises(ValueError):
            ceilo2nc(self.filename, self.output, self.site_meta, date='2020-10-23')

