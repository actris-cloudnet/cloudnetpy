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

    def test_data_types(self):
        for key in self.nc.variables.keys():
            value = self.nc.variables[key].dtype
            assert value == 'float32', f'{value} - {key}'

    def test_axis(self):
        assert self.nc.variables['range'].axis == 'Z'
        for key in self.nc.variables.keys():
            if key not in ('time', 'range'):
                assert hasattr(self.nc.variables[key], 'axis') is False

    def test_variable_values(self):
        assert 900 < self.nc.variables['wavelength'][:] < 1065
        assert 0 <= self.nc.variables['zenith_angle'][:] < 90
        assert np.all((self.nc.variables['height'][:] - self.site_meta['altitude']
                       - self.nc.variables['range'][:]) <= 1e-3)
        assert ma.min(self.nc.variables['beta'][:]) > 0

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
            ('range', 'Range from instrument'),
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
        assert self.nc.cloudnet_file_type == 'lidar'
