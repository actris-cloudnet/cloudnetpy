import os
from cloudnetpy.instruments import pollyxt2nc
import pytest
import netCDF4
import numpy as np
import numpy.ma as ma
import sys
from cloudnetpy.exceptions import ValidTimeStampError

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from lidar_fun import LidarFun
from all_products_fun import AllProductsFun

site_meta = {
    'name': 'Mindelo',
    'altitude': 123,
    'latitude': 45.0,
    'longitude': 22.0
}
filepath = f'{SCRIPT_PATH}/data/pollyxt/'
date = '2021-09-17'


class TestPolly:

    output = 'dummy_output_file.nc'
    uuid = pollyxt2nc(filepath, output, site_meta)
    nc = netCDF4.Dataset(output)
    lidar_fun = LidarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)

    def test_variable_names(self):
        keys = {'beta', 'beta_raw', 'calibration_factor', 'range', 'height', 'zenith_angle', 'time',
                'depolarisation', 'depolarisation_raw', 'altitude', 'latitude', 'longitude',
                'wavelength'}
        assert set(self.nc.variables.keys()) == keys

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if 'test_' in name:
                getattr(self.all_fun, name)()

    def test_common_lidar(self):
        for name, method in LidarFun.__dict__.items():
            if 'test_' in name:
                getattr(self.lidar_fun, name)()

    def test_variable_values(self):
        assert self.nc.variables['wavelength'][:] == 1064.0
        assert self.nc.variables['zenith_angle'][:] == 5.0
        assert ma.max(self.nc.variables['depolarisation'][:]) < 1
        assert ma.min(self.nc.variables['depolarisation'][:]) > -0.1
        assert np.all(np.diff(self.nc.variables['time'][:]) > 0)

    def test_comments(self):
        assert 'SNR threshold applied: 2' in self.nc.variables['beta'].comment

    def test_global_attributes(self):
        assert self.nc.source == 'TROPOS PollyXT'
        assert self.nc.title == f"PollyXT Raman lidar from {site_meta['name']}"

    def test_tear_down(self):
        os.remove(self.output)
        self.nc.close()


def test_date_argument():
    output = 'dummy_output_file.nc'
    pollyxt2nc(filepath, output, site_meta, date='2021-09-17')
    nc = netCDF4.Dataset(output)
    assert len(nc.variables['time']) == 80
    assert nc.year == '2021'
    assert nc.month == '09'
    assert nc.day == '17'
    nc.close()
    with pytest.raises(ValidTimeStampError):
        pollyxt2nc(filepath, output, site_meta, date='2021-09-15')
    os.remove(output)
