import os
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.instruments import disdrometer
import pytest
from tempfile import NamedTemporaryFile
import netCDF4

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def test_find_between_substrings():
    x = 'askdjföasdlf<SPECTRUM>kissa;koira;marsu</SPECTRUM>'
    result = disdrometer._find_between_substrings(x, '<SPECTRUM>', '</SPECTRUM>')
    assert result == 'kissa;koira;marsu'


def test_format_time():
    assert disdrometer._format_date('3.10.20') == '2020-10-03'


class TestParsivel:
    file_path = f'{SCRIPT_PATH}/data/parsivel/'
    site_meta = {
        'name': 'Kumpula',
    }
    temp_file = NamedTemporaryFile()

    @pytest.fixture(autouse=True)
    def init_tests(self):
        self.file = f'{self.file_path}parsivel_palaiseau_20210405.txt'
        self.uuid = disdrometer.disdrometer2nc(self.file, self.temp_file.name, 'parsivel',
                                               self.site_meta)
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    def test_global_attributes(self):
        assert self.nc.source == 'Parsivel'
        assert self.nc.cloudnet_file_type == 'disdrometer'
        assert self.nc.title == 'Disdrometer file from Kumpula'
        assert self.nc.year == '2021'
        assert self.nc.month == '04'
        assert self.nc.day == '05'
        assert self.nc.location == 'Kumpula'

    def test_dimensions(self):
        assert self.nc.dimensions['time'].size > 1000
        assert self.nc.dimensions['velocity'].size == 32
        assert self.nc.dimensions['diameter'].size == 32

    def test_date_validation(self):
        temp_file = NamedTemporaryFile()
        disdrometer.disdrometer2nc(self.file, temp_file.name, 'parsivel', self.site_meta,
                                   date='2021-04-05')

    def test_date_validation_fail(self):
        temp_file = NamedTemporaryFile()
        with pytest.raises(ValueError):
            disdrometer.disdrometer2nc(self.file, temp_file.name, 'parsivel', self.site_meta,
                                       date='2022-04-05')
