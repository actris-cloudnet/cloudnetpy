import os
import glob
from cloudnetpy import concat_lib
from cloudnetpy.instruments import ceilo2nc
import netCDF4
import numpy as np
import numpy.ma as ma
import sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from lidar_fun import LidarFun
from all_products_fun import AllProductsFun

site_meta = {
    'name': 'Hyytiälä',
    'altitude': 123,
    'calibration_factor': 2.0,
    'latitude': 45.0,
    'longitude': 22.0
}
files = glob.glob(f'{SCRIPT_PATH}/data/cl61d/*.nc')
files.sort()
daily_file = 'dummy_cl61_daily_file.nc'
concat_lib.concatenate_files(files, daily_file, concat_dimension='profile')
date = '2021-08-28'


class TestCl61d:

    output = 'dummy_cl61_output_file.nc'
    uuid = ceilo2nc(daily_file, output, site_meta)
    nc = netCDF4.Dataset(output)
    lidar_fun = LidarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)

    def test_variable_names(self):
        keys = {'beta', 'beta_smooth', 'calibration_factor', 'range', 'height', 'zenith_angle',
                'time', 'depolarisation', 'altitude', 'latitude', 'longitude', 'wavelength'}
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
        assert abs(self.nc.variables['wavelength'][:] - 910.55) < 0.001
        assert self.nc.variables['zenith_angle'][:] == 3.0
        assert ma.max(self.nc.variables['depolarisation'][:]) < 1
        assert ma.min(self.nc.variables['depolarisation'][:]) > -0.1

    def test_comments(self):
        assert 'SNR threshold applied: 5' in self.nc.variables['beta'].comment

    def test_global_attributes(self):
        assert self.nc.source == 'Vaisala CL61d'
        assert self.nc.title == f'CL61d ceilometer file from {site_meta["name"]}'

    def test_tear_down(self):
        os.remove(self.output)
        os.remove(daily_file)
        self.nc.close()


def test_date_argument():
    output = 'dummy_asdfasdfa_output_file.nc'
    concat_lib.concatenate_files(files, daily_file, concat_dimension='profile')
    ceilo2nc(daily_file, output, site_meta, date='2021-08-30')
    nc = netCDF4.Dataset(output)
    assert len(nc.variables['time']) == 12
    assert np.all(np.diff(nc.variables['time'][:]) > 0)
    assert nc.year == '2021'
    assert nc.month == '08'
    assert nc.day == '30'
    nc.close()
    os.remove(output)
    os.remove(daily_file)

