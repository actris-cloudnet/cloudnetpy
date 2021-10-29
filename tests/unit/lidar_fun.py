import netCDF4
import numpy as np
import numpy.ma as ma


class LidarFun:
    """Common tests for all lidars."""

    def __init__(self, nc: netCDF4.Dataset, site_meta: dict, date: str, uuid):
        self.nc = nc
        self.site_meta = site_meta
        self.date = date
        self.uuid = uuid

    def test_nan_values(self):
        for key in self.nc.variables.keys():
            assert bool(np.isnan(self.nc.variables[key]).all()) is False

    def test_data_types(self):
        for key in self.nc.variables.keys():
            assert self.nc.variables[key].dtype == 'float32'

    def test_axis(self):
        assert self.nc.variables['time'].axis == 'T'
        assert self.nc.variables['range'].axis == 'Z'
        for key in self.nc.variables.keys():
            if key not in ('time', 'range'):
                assert hasattr(self.nc.variables[key], 'axis') is False

    def test_empty_units(self):
        for key in self.nc.variables.keys():
            if hasattr(self.nc.variables[key], 'units'):
                assert self.nc.variables[key].units != ''

    def test_variable_values(self):
        for key in ('altitude', 'latitude', 'longitude'):
            assert self.nc.variables[key][:] == self.site_meta[key]
        assert 900 < self.nc.variables['wavelength'][:] < 1065
        assert 0 < self.nc.variables['zenith_angle'][:] < 90
        assert np.all((self.nc.variables['height'][:] - self.site_meta['altitude']
                       - self.nc.variables['range'][:]) < 0)
        assert ma.min(self.nc.variables['beta'][:]) > 0

    def test_standard_names(self):
        data = [
            ('time', 'time'),
            ('latitude', 'latitude'),
            ('longitude', 'longitude'),
            ('height', 'height_above_mean_sea_level'),
        ]
        for key, expected in data:
            value = self.nc.variables[key].standard_name
            assert value == expected, f'{value} != {expected}'

    def test_units(self):
        data = [
            ('zenith_angle', 'degree'),
            ('latitude', 'degree_north'),
            ('longitude', 'degree_east'),
            ('altitude', 'm'),
            ('range', 'm'),
            ('time', f'hours since {self.date} 00:00:00 +0:00'),
            ('calibration_factor', '1'),
            ('beta', 'sr-1 m-1'),
            ('beta_raw', 'sr-1 m-1'),
            ('beta_smooth', 'sr-1 m-1'),
            ('depolarisation', '1'),
            ('depolarisation_raw', '1'),
            ('wavelength', 'nm'),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].units
                assert value == expected, f'{value} != {expected}'

    def test_long_names(self):
        data = [
            ('zenith_angle', 'Zenith angle'),
            ('latitude', 'Latitude of site'),
            ('longitude', 'Longitude of site'),
            ('altitude', 'Altitude of site'),
            ('range', 'Range from instrument'),
            ('time', 'Time UTC'),
            ('calibration_factor', 'Attenuated backscatter calibration factor'),
            ('beta', 'Attenuated backscatter coefficient'),
            ('beta_raw', 'Attenuated backscatter coefficient'),
            ('depolarisation', 'Lidar volume linear depolarisation ratio'),
            ('depolarisation_raw', 'Lidar volume linear depolarisation ratio'),
            ('wavelength', 'Laser wavelength')
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f'{value} != {expected}'

    def test_global_attributes(self):
        assert self.nc.location == self.site_meta['name']
        assert self.nc.title == f'Lidar file from {self.site_meta["name"]}'
        assert self.nc.file_uuid == self.uuid
        assert self.nc.cloudnet_file_type == 'lidar'
        assert self.nc.Conventions == 'CF-1.8'
        y, m, d = self.date.split('-')
        assert self.nc.year == y
        assert self.nc.month == m
        assert self.nc.day == d
        for key in ('cloudnetpy_version', 'references', 'history'):
            assert hasattr(self.nc, key)
