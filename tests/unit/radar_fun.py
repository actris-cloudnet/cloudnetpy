import netCDF4
import numpy as np


class RadarFun:
    """Common tests for all radars."""

    def __init__(self, nc: netCDF4.Dataset, site_meta: dict, date: str, uuid):
        self.nc = nc
        self.site_meta = site_meta
        self.date = date
        self.uuid = uuid

    def test_variable_names(self):
        keys = {'Zh', 'v', 'radar_frequency', 'range', 'zenith_angle'}
        for key in keys:
            assert key in self.nc.variables

    def test_axis(self):
        assert self.nc.variables['range'].axis == 'Z'
        for key in self.nc.variables.keys():
            if key not in ('time', 'range'):
                assert hasattr(self.nc.variables[key], 'axis') is False

    def test_variable_values(self):
        assert 0 <= np.all(self.nc.variables['zenith_angle'][:]) < 10
        assert np.all((self.nc.variables['height'][:] - self.site_meta['altitude']
                       - self.nc.variables['range'][:]) <= 1e-3)

    def test_standard_names(self):
        data = [
            ('height', 'height_above_mean_sea_level'),
        ]
        for key, expected in data:
            value = self.nc.variables[key].standard_name
            assert value == expected, f'{value} != {expected}'

    def test_units(self):
        data = [
            ('zenith_angle', 'degree'),
            ('range', 'm'),
            ('frequency', 'GHz'),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].units
                assert value == expected, f'{value} != {expected}'

    def test_long_names(self):
        data = [
            ('zenith_angle', 'Zenith angle'),
            ('range', 'Range from instrument'),
            ('radar_frequency', 'Radar transmit frequency'),
            ('Zh', 'Radar reflectivity factor')
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f'{value} != {expected}'

    def test_global_attributes(self):
        assert self.nc.cloudnet_file_type == 'radar'
