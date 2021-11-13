import sys
import os
from os import path
import pytest
from cloudnetpy.instruments import basta2nc
import netCDF4
import numpy as np
from cloudnetpy.quality import Quality
from cloudnetpy.exceptions import ValidTimeStampError

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from radar_fun import RadarFun
from all_products_fun import AllProductsFun

site_meta = {
    'name': 'Palaiseau',
    'latitude': 50,
    'longitude': 104.5,
    'altitude': 50
}
filepath = f'{SCRIPT_PATH}/data/basta/'


class TestBASTA:
    date = '2021-08-27'
    output = 'dummy_basta_output_file.nc'
    output2 = 'dummy_temp_basta_file.nc'
    uuid = basta2nc(f'{filepath}/basta_1a_cldradLz1R025m_v03_20210827_000000.nc', output, site_meta)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()
    nc = netCDF4.Dataset(output)
    radar_fun = RadarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)

    def test_variable_names(self):
        keys = ('Zh',)
        for key in keys:
            assert key in self.nc.variables

    def test_variables(self):
        assert self.nc.variables['radar_frequency'][:].data == 95.0  # Hard coded

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if 'test_' in name:
                getattr(self.all_fun, name)()

    def test_common_radar(self):
        for name, method in RadarFun.__dict__.items():
            if 'test_' in name:
                getattr(self.radar_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_tear_down(self):
        os.remove(self.output)
        self.nc.close()
