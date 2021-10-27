""" This module contains unit tests for ceilo-module. """
import os
from cloudnetpy.instruments import vaisala, ceilo2nc
import pytest
import numpy as np
from numpy.testing import assert_equal
import netCDF4


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("input, result", [
    ('01:30:00', 1.5),
    ('02:00:00', 2),
    ('13:15:00', 13.25),
])
def test_time_to_fraction_hour(input, result):
    assert vaisala.time_to_fraction_hour(input) == result


@pytest.mark.parametrize("keys, values, result", [
    (('a', 'b'), [[1, 2], [1, 2], [1, 2]],
     {'a': np.array([1, 1, 1]), 'b': np.array([2, 2, 2])}),
])
def test_values_to_dict(keys, values, result):
    assert_equal(vaisala.values_to_dict(keys, values), result)


@pytest.mark.parametrize("string, indices, result", [
    ('abcd', [3, 4], ['d']),
    ('abcd', [0, 4], ['abcd']),
    ('abcdedfg', [1, 2, 4, 5], ['b', 'cd', 'e']),
])
def test_split_string(string, indices, result):
    assert_equal(vaisala.split_string(string, indices), result)


class TestCL51:

    site_meta = {
        'name': 'Kumpula',
        'altitude': 123,
        'latitude': 45.0,
        'longitude': 22.0
    }

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.input = f'{SCRIPT_PATH}/data/vaisala/cl51.DAT'
        self.output = 'dummy_output_file.nc'
        yield
        if os.path.isfile(self.output):
            os.remove(self.output)

    def test_variables(self):
        ceilo2nc(self.input, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        for key in ('beta', 'beta_raw', 'beta_smooth', 'calibration_factor', 'range', 'height',
                    'zenith_angle', 'time'):
            assert key in nc.variables
        for key in ('depolarisation', 'depolarisation_raw', ):
            assert key not in nc.variables
        for key in ('altitude', 'latitude', 'longitude'):
            assert nc.variables[key][:] == self.site_meta[key]
        assert nc.variables['wavelength'][:] == 910
        assert nc.variables['zenith_angle'][:] == 4.5
        assert nc.variables['zenith_angle'].units == 'degree'
        assert nc.variables['zenith_angle'].dtype == 'float32'
        assert nc.variables['latitude'].units == 'degree_north'
        assert nc.variables['longitude'].units == 'degree_east'
        assert nc.variables['altitude'].units == 'm'
        assert np.all((nc.variables['height'][:] - nc.variables['range'][:]) > 0)
        assert np.all((nc.variables['height'][:] - self.site_meta['altitude']
                       - nc.variables['range'][:]) < 0)
        assert np.all(np.diff(nc.variables['time'][:]) > 0)
        assert nc.variables['beta'].units == 'sr-1 m-1'
        assert nc.variables['beta_smooth'].units == 'sr-1 m-1'
        nc.close()

    def test_global_attributes(self):
        uuid = ceilo2nc(self.input, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        assert nc.source == 'Vaisala CL51 ceilometer'
        assert nc.location == self.site_meta['name']
        assert nc.title == f'Lidar file from {self.site_meta["name"]}'
        assert nc.file_uuid == uuid
        assert nc.cloudnet_file_type == 'lidar'
        assert nc.year == '2020'
        assert nc.month == '11'
        assert nc.day == '15'
        nc.close()

    def test_date_argument(self):
        ceilo2nc(self.input, self.output, self.site_meta, date='2020-11-15')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 2
        assert nc.year == '2020'
        assert nc.month == '11'
        assert nc.day == '15'
        nc.close()
        with pytest.raises(ValueError):
            ceilo2nc(self.input, self.output, self.site_meta, date='2021-09-15')


class TestCL31:

    site_meta = {
        'name': 'Kumpula',
        'altitude': 123,
        'latitude': 45.0,
        'longitude': 22.0
    }

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.input = f'{SCRIPT_PATH}/data/vaisala/cl31.DAT'
        self.output = 'dummy_output_file.nc'
        yield
        if os.path.isfile(self.output):
            os.remove(self.output)

    def test_variables(self):
        ceilo2nc(self.input, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        for key in ('beta', 'beta_raw', 'beta_smooth', 'calibration_factor', 'range', 'height',
                    'zenith_angle', 'time'):
            assert key in nc.variables
        for key in ('depolarisation', 'depolarisation_raw', ):
            assert key not in nc.variables
        for key in ('altitude', 'latitude', 'longitude'):
            assert nc.variables[key][:] == self.site_meta[key]
        assert nc.variables['wavelength'][:] == 910
        assert nc.variables['zenith_angle'][:] == 12
        assert nc.variables['zenith_angle'].units == 'degree'
        assert nc.variables['zenith_angle'].dtype == 'float32'
        assert nc.variables['latitude'].units == 'degree_north'
        assert nc.variables['longitude'].units == 'degree_east'
        assert nc.variables['altitude'].units == 'm'
        vertical_range = nc.variables['height'][:] - nc.variables['altitude'][:]
        assert np.all((nc.variables['range'][:] - vertical_range) > 0)
        assert nc.variables['beta'].units == 'sr-1 m-1'
        assert nc.variables['beta_smooth'].units == 'sr-1 m-1'
        nc.close()

    def test_global_attributes(self):
        uuid = ceilo2nc(self.input, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        assert nc.source == 'Vaisala CL31 ceilometer'
        assert nc.location == self.site_meta['name']
        assert nc.title == f'Lidar file from {self.site_meta["name"]}'
        assert nc.file_uuid == uuid
        assert nc.cloudnet_file_type == 'lidar'
        assert nc.year == '2020'
        assert nc.month == '04'
        assert nc.day == '10'
        nc.close()

    def test_date_argument(self):
        self.input = f'{SCRIPT_PATH}/data/vaisala/cl31_badtime.DAT'
        ceilo2nc(self.input, self.output, self.site_meta, date='2020-04-10')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 3
        assert nc.year == '2020'
        assert nc.month == '04'
        assert nc.day == '10'
        nc.close()
        ceilo2nc(self.input, self.output, self.site_meta, date='2020-04-11')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 2
        assert nc.year == '2020'
        assert nc.month == '04'
        assert nc.day == '11'
        nc.close()
        with pytest.raises(ValueError):
            ceilo2nc(self.input, self.output, self.site_meta, date='2020-04-12')


class TestCT25k:

    site_meta = {
        'name': 'Kumpula',
        'altitude': 123,
        'latitude': 45.0,
        'longitude': 22.0
    }

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.input = f'{SCRIPT_PATH}/data/vaisala/ct25k.dat'
        self.output = 'dummy_output_file.nc'
        yield
        if os.path.isfile(self.output):
            os.remove(self.output)

    def test_variables(self):
        ceilo2nc(self.input, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        for key in ('beta', 'beta_raw', 'beta_smooth', 'calibration_factor', 'range', 'height',
                    'zenith_angle', 'time'):
            assert key in nc.variables
        for key in ('depolarisation', 'depolarisation_raw', ):
            assert key not in nc.variables
        for key in ('altitude', 'latitude', 'longitude'):
            assert nc.variables[key][:] == self.site_meta[key]
        assert nc.variables['wavelength'][:] == 905
        assert nc.variables['zenith_angle'][:] == 15
        assert nc.variables['zenith_angle'].units == 'degree'
        assert nc.variables['zenith_angle'].dtype == 'float32'
        assert nc.variables['latitude'].units == 'degree_north'
        assert nc.variables['longitude'].units == 'degree_east'
        assert nc.variables['altitude'].units == 'm'
        assert np.all((nc.variables['height'][:] - self.site_meta['altitude']
                       - nc.variables['range'][:]) < 0)
        assert np.all(np.diff(nc.variables['time'][:]) > 0)
        assert nc.variables['beta'].units == 'sr-1 m-1'
        assert nc.variables['beta_smooth'].units == 'sr-1 m-1'
        nc.close()

    def test_global_attributes(self):
        uuid = ceilo2nc(self.input, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        assert nc.source == 'Vaisala CT25k ceilometer'
        assert nc.location == self.site_meta['name']
        assert nc.title == f'Lidar file from {self.site_meta["name"]}'
        assert nc.file_uuid == uuid
        assert nc.cloudnet_file_type == 'lidar'
        assert nc.year == '2020'
        assert nc.month == '10'
        assert nc.day == '29'
        nc.close()

    def test_date_argument(self):
        ceilo2nc(self.input, self.output, self.site_meta, date='2020-10-29')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 3
        assert nc.year == '2020'
        assert nc.month == '10'
        assert nc.day == '29'
        nc.close()
        with pytest.raises(ValueError):
            ceilo2nc(self.input, self.output, self.site_meta, date='2021-09-15')
