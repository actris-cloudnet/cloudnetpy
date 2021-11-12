import os
import sys
from os import path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pytest
from numpy.testing import assert_array_equal
from cloudnetpy.instruments import rpg
from distutils.dir_util import copy_tree

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from radar_fun import RadarFun
from all_products_fun import AllProductsFun

site_meta = {
    'name': 'the_station',
    'altitude': 50,
    'latitude': 23,
    'longitude': 34.2
}
filepath = f'{SCRIPT_PATH}/data/rpg-fmcw-94'


class TestReduceHeader:
    n_points = 100
    header = {'a': n_points * [1], 'b': n_points * [2], 'c': n_points * [3]}

    def test_1(self):
        assert_array_equal(rpg._reduce_header(self.header), {'a': 1, 'b': 2, 'c': 3})

    def test_2(self):
        self.header['a'][50] = 10
        with pytest.raises(AssertionError):
            assert_array_equal(rpg._reduce_header(self.header), {'a': 1, 'b': 2, 'c': 3})


class TestRPG2nc:



    def test_default_processing(self):
        uuid, files = rpg.rpg2nc(self.file_path, self.temp_file.name,
                                 self.site_meta)
        assert len(files) == 3
        assert len(uuid) == 32

    def test_default_date_validation(self):
        _, files = rpg.rpg2nc(self.file_path, self.temp_file.name,
                              self.site_meta, date='2020-10-22')
        assert len(files) == 2

    def test_processing_of_one_file(self):
        _, files = rpg.rpg2nc(self.file_path, self.temp_file.name,
                              self.site_meta, date='2020-10-23')
        assert len(files) == 1

    def test_processing_of_no_files(self):
        _, files = rpg.rpg2nc(self.file_path, self.temp_file.name,
                              self.site_meta, date='2020-10-24')
        assert len(files) == 0

    def test_uuid_from_user(self):
        test_uuid = 'abc'
        uuid, _ = rpg.rpg2nc(self.file_path, self.temp_file.name,
                             self.site_meta, date='2020-10-23',
                             uuid=test_uuid)
        assert uuid == test_uuid

    def test_handling_of_corrupted_files(self):
        temp_dir = TemporaryDirectory()
        copy_tree(self.file_path, temp_dir.name)
        with open(f'{temp_dir.name}/foo.LV1', 'w') as f:
            f.write('kissa')
        _, files = rpg.rpg2nc(temp_dir.name, self.temp_file.name,
                              self.site_meta, date='2020-10-22')
        assert len(files) == 2
