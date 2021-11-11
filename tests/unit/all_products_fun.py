import netCDF4
import numpy as np


class AllProductsFun:
    """Common tests for all Cloudnet products."""

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
            value = self.nc.variables[key].dtype
            assert value == 'float32', f'{value} - {key}'

    def test_time_axis(self):
        assert self.nc.variables['time'].axis == 'T'

    def test_empty_units(self):
        for key in self.nc.variables.keys():
            if hasattr(self.nc.variables[key], 'units'):
                assert self.nc.variables[key].units != ''

    def test_variable_values(self):
        for key in ('altitude', 'latitude', 'longitude'):
            assert self.nc.variables[key][:] == self.site_meta[key]

    def test_standard_names(self):
        data = [
            ('time', 'time'),
            ('latitude', 'latitude'),
            ('longitude', 'longitude'),
        ]
        for key, expected in data:
            value = self.nc.variables[key].standard_name
            assert value == expected, f'{value} != {expected}'

    def test_units(self):
        data = [
            ('latitude', 'degree_north'),
            ('longitude', 'degree_east'),
            ('altitude', 'm'),
            ('time', f'hours since {self.date} 00:00:00 +00:00'),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].units
                assert value == expected, f'{value} != {expected}'

    def test_long_names(self):
        data = [
            ('latitude', 'Latitude of site'),
            ('longitude', 'Longitude of site'),
            ('altitude', 'Altitude of site'),
            ('time', 'Time UTC'),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f'{value} != {expected}'

    def test_global_attributes(self):
        assert self.nc.location == self.site_meta['name']
        assert self.nc.file_uuid == self.uuid
        assert self.nc.Conventions == 'CF-1.8'
        y, m, d = self.date.split('-')
        assert self.nc.year == y
        assert self.nc.month == m
        assert self.nc.day == d
        for key in ('cloudnetpy_version', 'references', 'history'):
            assert hasattr(self.nc, key)
