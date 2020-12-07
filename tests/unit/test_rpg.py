from os import path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pytest
import shutil
from numpy.testing import assert_array_equal
from cloudnetpy.instruments import rpg
from distutils.dir_util import copy_tree

SCRIPT_PATH = path.dirname(path.realpath(__file__))


@pytest.fixture
def example_files(tmpdir):
    file_names = ['f.LV1', 'f.txt', 'f.LV0', 'f.lv1', 'g.LV1']
    folder = tmpdir.mkdir('data/')
    for name in file_names:
        with open(folder.join(name), 'wb') as f:
            f.write(b'abc')
    return folder


def test_get_rpg_files(example_files):
    dir_name = example_files.dirname + '/data'
    result = ['/'.join((dir_name, x)) for x in ('f.LV1', 'g.LV1')]
    assert rpg.get_rpg_files(dir_name) == result


class TestReduceHeader:
    n_points = 100
    header = {'a': n_points * [1], 'b': n_points * [2], 'c': n_points * [3]}

    def test_1(self):
        assert_array_equal(rpg._reduce_header(self.header),
                           {'a': 1, 'b': 2, 'c': 3})

    def test_2(self):
        self.header['a'][50] = 10
        with pytest.raises(AssertionError):
            assert_array_equal(rpg._reduce_header(self.header),
                               {'a': 1, 'b': 2, 'c': 3})


class TestRPG2nc:

    file_path = f'{SCRIPT_PATH}/data/rpg-fmcw-94'
    site_meta = {
        'name': 'the_station',
        'altitude': 50
    }
    temp_file = NamedTemporaryFile()

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
