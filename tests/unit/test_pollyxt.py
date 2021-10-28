import os
from cloudnetpy.instruments import pollyxt2nc
import pytest
import netCDF4
import numpy as np
import numpy.ma as ma

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class TestPolly:

    site_meta = {
        'name': 'Mindelo',
        'altitude': 123,
        'latitude': 45.0,
        'longitude': 22.0
    }
    filepath = f'{SCRIPT_PATH}/data/pollyxt/'

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.output = 'dummy_output_file.nc'
        yield
        if os.path.isfile(self.output):
            os.remove(self.output)

    def test_variables(self):
        pollyxt2nc(self.filepath, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        for key in ('beta', 'beta_raw', 'calibration_factor', 'range', 'height', 'zenith_angle',
                    'time', 'depolarisation', 'depolarisation_raw'):
            assert key in nc.variables
            assert bool(np.isnan(nc.variables[key]).all()) is False
        for key in ('altitude', 'latitude', 'longitude'):
            assert nc.variables[key][:] == self.site_meta[key]
        for key in ('snr', 'beta_smooth', 'depolarisation_smooth'):
            assert key not in nc.variables
        assert nc.variables['wavelength'][:] == 1064.0
        assert nc.variables['wavelength'].dtype == 'float32'
        assert nc.variables['zenith_angle'][:] == 5.0
        assert nc.variables['zenith_angle'].units == 'degree'
        assert nc.variables['zenith_angle'].dtype == 'float32'
        assert nc.variables['latitude'].units == 'degree_north'
        assert nc.variables['longitude'].units == 'degree_east'
        assert nc.variables['altitude'].units == 'm'
        assert np.all((nc.variables['height'][:] - self.site_meta['altitude']
                       - nc.variables['range'][:]) < 0)
        assert np.all(np.diff(nc.variables['time'][:]) > 0)
        assert nc.variables['beta'].units == 'sr-1 m-1'
        assert ma.min(nc.variables['beta'][:]) > 0
        assert nc.variables['depolarisation'].units == '1'
        assert nc.variables['depolarisation_raw'].units == '1'
        depol = nc.variables['depolarisation'][:]
        assert ma.max(depol) < 1
        assert ma.min(depol) > -0.1
        nc.close()

    def test_global_attributes(self):
        uuid = pollyxt2nc(self.filepath, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        assert nc.source == 'PollyXT Raman lidar'
        assert nc.location == self.site_meta['name']
        assert nc.title == f'Lidar file from {self.site_meta["name"]}'
        assert nc.file_uuid == uuid
        assert nc.cloudnet_file_type == 'lidar'
        assert nc.year == '2021'
        assert nc.month == '09'
        assert nc.day == '17'
        nc.close()

    def test_date_argument(self):
        pollyxt2nc(self.filepath, self.output, self.site_meta, date='2021-09-17')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 80
        assert nc.year == '2021'
        assert nc.month == '09'
        assert nc.day == '17'
        nc.close()
        with pytest.raises(ValueError):
            pollyxt2nc(self.filepath, self.output, self.site_meta, date='2021-09-15')
