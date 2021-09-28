import os
from cloudnetpy.instruments import disdrometer
import pytest
from tempfile import NamedTemporaryFile
import netCDF4

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def test_format_time():
    assert disdrometer._format_thies_date('3.10.20') == '2020-10-03'


class TestParsivel:
    file_path = f'{SCRIPT_PATH}/data/parsivel/'
    site_meta = {'name': 'Kumpula'}
    temp_file = NamedTemporaryFile()

    @pytest.fixture(autouse=True)
    def init_tests(self):
        self.file = f'{self.file_path}juelich.log'
        self.uuid = disdrometer.disdrometer2nc(self.file, self.temp_file.name, self.site_meta)
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    def test_global_attributes(self):
        assert 'Parsivel' in self.nc.source
        assert self.nc.cloudnet_file_type == 'disdrometer'
        assert self.nc.title == 'Disdrometer file from Kumpula'
        assert self.nc.year == '2021'
        assert self.nc.month == '03'
        assert self.nc.day == '18'
        assert self.nc.location == 'Kumpula'

    def test_dimensions(self):
        assert self.nc.dimensions['time'].size > 1000
        assert self.nc.dimensions['velocity'].size == 32
        assert self.nc.dimensions['diameter'].size == 32


class TestParsivel2:
    file = f'{SCRIPT_PATH}/data/parsivel/norunda.log'
    site_meta = {'name': 'Norunda'}

    def test_date_validation(self):
        temp_file = NamedTemporaryFile()
        disdrometer.disdrometer2nc(self.file, temp_file.name, self.site_meta, date='2019-11-09')

    def test_date_validation_fail(self):
        temp_file = NamedTemporaryFile()
        with pytest.raises(ValueError):
            disdrometer.disdrometer2nc(self.file, temp_file.name, self.site_meta, date='2022-04-05')


class TestParsivel3:
    file = f'{SCRIPT_PATH}/data/parsivel/ny-alesund.log'
    site_meta = {'name': 'Ny Alesund'}

    def test_date_validation(self):
        temp_file = NamedTemporaryFile()
        disdrometer.disdrometer2nc(self.file, temp_file.name, self.site_meta, date='2021-04-16')


class TestThies:

    file_path = f'{SCRIPT_PATH}/data/thies-lnm/'
    temp_file = NamedTemporaryFile()
    site_meta = {'name': 'Lindenberg', 'latitude': 34.6}

    @pytest.fixture(autouse=True)
    def init_tests(self):
        self.file = f'{self.file_path}2021091507.txt'
        self.uuid = disdrometer.disdrometer2nc(self.file, self.temp_file.name, self.site_meta)
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    def test_processing(self):
        temp_file = NamedTemporaryFile()
        disdrometer.disdrometer2nc(self.file, temp_file.name, self.site_meta, date='2021-09-15')
        assert self.nc.title == 'Disdrometer file from Lindenberg'
        assert self.nc.year == '2021'
        assert self.nc.month == '09'
        assert self.nc.day == '15'
        assert self.nc.location == 'Lindenberg'
        assert self.nc.cloudnet_file_type == 'disdrometer'
