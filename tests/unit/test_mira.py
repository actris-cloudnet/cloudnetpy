import sys
import os
from os import path
import pytest
from cloudnetpy.instruments import mira
import netCDF4
import numpy as np
from cloudnetpy.quality import Quality

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from radar_fun import RadarFun
from all_products_fun import AllProductsFun

site_meta = {
    'name': 'The_station',
    'latitude': 50,
    'longitude': 104.5,
    'altitude': 50
}
filepath = f'{SCRIPT_PATH}/data/mira/'


class TestMeasurementDate:

    correct_date = ['2020', '05', '24']

    @pytest.fixture(autouse=True)
    def _init(self, raw_mira_file):
        self.raw_radar = mira.Mira(raw_mira_file, {'name': 'Test'})

    def test_validate_date(self):
        self.raw_radar.screen_time('2020-05-24')
        assert self.raw_radar.date == self.correct_date

    def test_validate_date_fails(self):
        with pytest.raises(ValueError):
            self.raw_radar.screen_time('2020-05-23')


class TestMIRA2nc:
    date = '2021-01-02'
    n_time1 = 146
    n_time2 = 145
    output = 'dummy_mira_output_file.nc'
    output2 = 'dummy_temp_mira_file.nc'
    uuid = mira.mira2nc(f'{filepath}/20210102_0000.mmclx', output, site_meta)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()
    nc = netCDF4.Dataset(output)
    radar_fun = RadarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)

    def test_variable_names(self):
        keys = {'Zh', 'v', 'width', 'ldr', 'SNR', 'time', 'range', 'radar_frequency',
                'nyquist_velocity', 'latitude', 'longitude', 'altitude',
                'zenith_angle', 'height', 'rg0', 'nave', 'prf', 'nfft'}
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert self.nc.variables['radar_frequency'][:].data == 35.5  # Hard coded
        assert np.all(self.nc.variables['zenith_angle'][:].data) == 0

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

    def test_long_names(self):
        data = [
            ('nfft', 'Number of FFT points'),
            ('nave', 'Number of spectral averages (not accounting for overlapping FFTs)'),
            ('rg0', 'Number of lowest range gates'),
            ('prf', 'Pulse Repetition Frequency'),
            ('SNR', 'Signal-to-noise ratio')
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f'{value} != {expected}'

    def test_processing_of_one_nc_file(self):
        assert len(self.nc.variables['time'][:]) == self.n_time1

    def test_global_attributes(self):
        assert self.nc.source == 'METEK MIRA-35'
        assert self.nc.title == f'MIRA-35 cloud radar file from {site_meta["name"]}'

    def test_processing_of_several_nc_files(self):
        mira.mira2nc(filepath, self.output2, site_meta)
        nc = netCDF4.Dataset(self.output2)
        assert len(nc.variables['time'][:]) == self.n_time1 + self.n_time2
        nc.close()

    def test_correct_date_validation(self):
        mira.mira2nc(f'{filepath}/20210102_0000.mmclx', self.output2, site_meta, date='2021-01-02')

    def test_wrong_date_validation(self):
        with pytest.raises(ValueError):
            mira.mira2nc(f'{filepath}/20210102_0000.mmclx', self.output2, site_meta,
                         date='2021-01-03')

    def test_uuid_from_user(self):
        uuid_from_user = 'kissa'
        uuid = mira.mira2nc(f'{filepath}/20210102_0000.mmclx', self.output2,
                            site_meta, uuid=uuid_from_user)
        nc = netCDF4.Dataset(self.output2)
        assert nc.file_uuid == uuid_from_user
        assert uuid == uuid_from_user
        nc.close()

    def test_tear_down(self):
        os.remove(self.output)
        os.remove(self.output2)
        self.nc.close()
