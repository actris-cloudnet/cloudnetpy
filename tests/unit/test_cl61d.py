import os
import glob
from cloudnetpy import concat_lib
from cloudnetpy.instruments import ceilo2nc
import pytest
import netCDF4
import numpy as np
import numpy.ma as ma

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class TestCl61d:

    site_meta = {
        'name': 'Hyytiälä',
        'altitude': 123,
        'calibration_factor': 2.0,
        'latitude': 45.0,
        'longitude': 22.0
    }
    files = glob.glob(f'{SCRIPT_PATH}/data/cl61d/*.nc')
    files.sort()

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.output = 'dummy_output_file.nc'
        self.filename = 'dummy_daily_file.nc'
        concat_lib.concatenate_files(self.files, self.filename, concat_dimension='profile')
        yield
        os.remove(self.filename)
        os.remove(self.output)

    def test_variables(self):
        ceilo2nc(self.filename, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        for key in ('beta', 'depolarisation', 'beta_smooth', 'calibration_factor', 'range',
                    'height', 'zenith_angle', 'time'):
            assert key in nc.variables
        for key in ('beta_raw', 'depolarisation_raw', 'x_pol', 'x_pol'):
            assert key not in nc.variables
        for key in ('altitude', 'latitude', 'longitude'):
            assert nc.variables[key][:] == self.site_meta[key]
        assert abs(nc.variables['wavelength'][:] - 910.55) < 0.001
        assert nc.variables['zenith_angle'][:] == 3
        assert nc.variables['zenith_angle'].units == 'degree'
        assert np.all((nc.variables['height'][:] - self.site_meta['altitude']
                       - nc.variables['range'][:]) < 0)
        assert nc.variables['beta'].units == 'sr-1 m-1'
        assert nc.variables['beta_smooth'].units == 'sr-1 m-1'
        assert nc.variables['depolarisation'].units == '1'
        depol = nc.variables['depolarisation'][:]
        assert nc.variables['zenith_angle'].dtype == 'float32'
        assert nc.variables['latitude'].units == 'degree_north'
        assert nc.variables['longitude'].units == 'degree_east'
        assert nc.variables['altitude'].units == 'm'
        assert ma.max(depol) < 1
        assert ma.min(depol) > 0
        nc.close()

    def test_global_attributes(self):
        uuid = ceilo2nc(self.filename, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        assert nc.source == 'Vaisala CL61d ceilometer'
        assert nc.location == self.site_meta['name']
        assert nc.title == f'Lidar file from {self.site_meta["name"]}'
        assert nc.file_uuid == uuid
        assert nc.cloudnet_file_type == 'lidar'
        nc.close()

    def test_date_argument(self):
        ceilo2nc(self.filename, self.output, self.site_meta, date='2021-08-30')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 12
        assert np.all(np.diff(nc.variables['time'][:]) > 0)
        assert nc.year == '2021'
        assert nc.month == '08'
        assert nc.day == '30'
        nc.close()
