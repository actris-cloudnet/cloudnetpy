import os
import sys
from os import path
import math
from tempfile import TemporaryDirectory
import pytest
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import netCDF4
from cloudnetpy.instruments import rpg2nc
from cloudnetpy.instruments import rpg
from distutils.dir_util import copy_tree
from cloudnetpy_qc import Quality
from cloudnetpy.exceptions import ValidTimeStampError, InconsistentDataError


SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from radar_fun import RadarFun
from all_products_fun import AllProductsFun

site_meta = {
    'name': 'the_station',
    'altitude': 50,
    'latitude': 23,
    'longitude': 34.0
}
filepath = f'{SCRIPT_PATH}/data/rpg-fmcw-94'


class TestReduceHeader:
    n_points = 100
    header = {'a': n_points * [1], 'b': n_points * [2], 'c': n_points * [3]}

    def test_1(self):
        assert_array_equal(rpg._reduce_header(self.header), {'a': 1, 'b': 2, 'c': 3})

    def test_2(self):
        self.header['a'][50] = 10
        with pytest.raises(InconsistentDataError):
            assert_array_equal(rpg._reduce_header(self.header), {'a': 1, 'b': 2, 'c': 3})


class TestRPG2nc94GHz:

    date = '2020-10-22'
    output = 'dummy_rpg_output_file.nc'
    output2 = 'dummy_temp_rpg_file.nc'
    uuid, valid_files = rpg2nc(filepath, output, site_meta, date=date)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()
    nc = netCDF4.Dataset(output)
    radar_fun = RadarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)

    def test_variable_names(self):
        mandatory_variables = ('Zh', 'v', 'width', 'ldr', 'time', 'range', 'altitude', 'latitude',
                               'longitude', 'radar_frequency', 'nyquist_velocity', 'zenith_angle',
                               'skewness', 'kurtosis', 'rain_rate', 'relative_humidity',
                               'temperature', 'pressure', 'wind_speed', 'wind_direction',
                               'voltage', 'brightness_temperature', 'lwp', 'if_power',
                               'azimuth_angle', 'status_flag', 'transmitted_power',
                               'transmitter_temperature', 'receiver_temperature',
                               'pc_temperature', 'rho_cx', 'phi_cx')
        for key in mandatory_variables:
            assert key in self.nc.variables

    def test_long_names(self):
        data = [
            ('rho_cx', 'Co-cross-channel correlation coefficient'),
            ('phi_cx', 'Co-cross-channel differential phase'),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f'{value} != {expected}'

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if 'test_' in name:
                getattr(self.all_fun, name)()

    def test_variables(self):
        assert math.isclose(self.nc.variables['radar_frequency'][:].data, 94.0, abs_tol=0.1)
        assert np.all(self.nc.variables['zenith_angle'][:].data) == 0

    def test_fill_values(self):
        bad_values = (-999, 1e-10)
        for key in self.nc.variables.keys():
            for value in bad_values:
                array = self.nc.variables[key][:]
                if array.ndim > 1:
                    assert not np.any(np.isclose(array, value)), f'{key} - {value}: {array}'

    def test_global_attributes(self):
        assert self.nc.source == 'RPG-Radiometer Physics RPG-FMCW-94'
        assert self.nc.title == f'RPG-FMCW-94 cloud radar from {site_meta["name"]}'

    def test_common_radar(self):
        for name, method in RadarFun.__dict__.items():
            if 'test_' in name:
                getattr(self.radar_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_default_processing(self):
        uuid, files = rpg2nc(filepath, self.output2, site_meta)
        print('')
        for file in files:
            print(file)
        assert len(files) == 3
        assert len(uuid) == 36

    def test_date_validation(self):
        uuid, files = rpg2nc(filepath, self.output2, site_meta, date=self.date)
        assert len(files) == 2

    def test_processing_of_one_file(self):
        uuid, files = rpg2nc(filepath, self.output2, site_meta, date='2020-10-23')
        assert len(files) == 1

    def test_incorrect_date_processing(self):
        with pytest.raises(ValidTimeStampError):
            rpg2nc(filepath, self.output2, site_meta, date='2010-10-24')

    def test_uuid_from_user(self):
        test_uuid = 'abc'
        uuid, _ = rpg.rpg2nc(filepath, self.output2, site_meta, date='2020-10-23', uuid=test_uuid)
        assert uuid == test_uuid

    def test_handling_of_corrupted_files(self):
        temp_dir = TemporaryDirectory()
        copy_tree(filepath, temp_dir.name)
        with open(f'{temp_dir.name}/foo.LV1', 'w') as f:
            f.write('kissa')
        _, files = rpg.rpg2nc(temp_dir.name, self.output2, site_meta, date='2020-10-22')
        assert len(files) == 2

    def test_geolocation_from_source_file(self):
        meta_without_geolocation = {'name': 'Kumpula', 'altitude': 34}
        rpg.rpg2nc(filepath, self.output2, meta_without_geolocation)
        nc = netCDF4.Dataset(self.output2)
        for key in ('latitude', 'longitude'):
            assert key in nc.variables
            assert nc.variables[key][:] > 0
        nc.close()

    def test_cleanup(self):
        os.remove(self.output)
        os.remove(self.output2)
        self.nc.close()


class TestRPG2ncSTSR35GHz:

    date = '2021-09-13'
    output = 'dummy_rpg_stsr_output_file.nc'
    output2 = 'dummy_temp_stsr_rpg_file.nc'
    uuid, valid_files = rpg2nc(filepath, output, site_meta, date=date)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()
    nc = netCDF4.Dataset(output)
    radar_fun = RadarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)

    def test_variable_names(self):
        mandatory_variables = ('Zh', 'v', 'width', 'sldr', 'time', 'range', 'altitude', 'latitude',
                               'longitude', 'radar_frequency', 'nyquist_velocity', 'zenith_angle',
                               'skewness', 'kurtosis', 'rain_rate', 'relative_humidity',
                               'temperature', 'pressure', 'wind_speed', 'wind_direction',
                               'voltage', 'brightness_temperature', 'lwp', 'if_power',
                               'azimuth_angle', 'status_flag', 'transmitted_power',
                               'transmitter_temperature', 'receiver_temperature',
                               'pc_temperature', 'zdr', 'rho_hv', 'phi_dp', 'sldr', 'srho_hv',
                               'kdp', 'differential_attenuation')
        for key in mandatory_variables:
            assert key in self.nc.variables

    def test_long_names(self):
        data = [
            ('rho_cx', 'Co-cross-channel correlation coefficient'),
            ('phi_cx', 'Co-cross-channel differential phase'),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f'{value} != {expected}'

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if 'test_' in name:
                getattr(self.all_fun, name)()

    def test_variables(self):
        assert math.isclose(self.nc.variables['radar_frequency'][:].data, 35.0, rel_tol=0.1)
        assert math.isclose(ma.median(self.nc.variables['zenith_angle'][:].data), 15, abs_tol=1)

    def test_fill_values(self):
        bad_values = (-999, 1e-10)
        for key in self.nc.variables.keys():
            for value in bad_values:
                array = self.nc.variables[key][:]
                if array.ndim > 1:
                    assert not np.any(np.isclose(array, value)), f'{key} - {value}: {array}'

    def test_global_attributes(self):
        assert self.nc.source == 'RPG-Radiometer Physics RPG-FMCW-35'
        assert self.nc.title == f'RPG-FMCW-35 cloud radar from {site_meta["name"]}'

    def test_common_radar(self):
        for name, method in RadarFun.__dict__.items():
            if 'test_' in name:
                getattr(self.radar_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_cleanup(self):
        os.remove(self.output)
        self.nc.close()
