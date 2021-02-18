from os import path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from cloudnetpy.instruments import hatpro
from distutils.dir_util import copy_tree

SCRIPT_PATH = path.dirname(path.realpath(__file__))


class TestHatpro2nc:

    file_path = f'{SCRIPT_PATH}/data/hatpro/'
    site_meta = {
        'name': 'the_station',
        'altitude': 50
    }
    temp_file = NamedTemporaryFile()

    def test_default_processing(self):
        uuid, files = hatpro.hatpro2nc(self.file_path, self.temp_file.name, self.site_meta)
        assert len(files) == 3
        assert len(uuid) == 32

    def test_processing_of_several_files(self):
        _, files = hatpro.hatpro2nc(self.file_path, self.temp_file.name, self.site_meta,
                                    date='2021-01-23')
        assert len(files) == 2

    def test_processing_of_one_file(self):
        _, files = hatpro.hatpro2nc(self.file_path, self.temp_file.name, self.site_meta,
                                    date='2020-07-23')
        assert len(files) == 1

    def test_processing_of_no_files(self):
        _, files = hatpro.hatpro2nc(self.file_path, self.temp_file.name, self.site_meta,
                                    date='2020-10-24')
        assert len(files) == 0

    def test_uuid_from_user(self):
        test_uuid = 'abc'
        uuid, _ = hatpro.hatpro2nc(self.file_path, self.temp_file.name, self.site_meta,
                                   date='2021-01-23', uuid=test_uuid)
        assert uuid == test_uuid

    def test_handling_of_corrupted_files(self):
        temp_dir = TemporaryDirectory()
        copy_tree(self.file_path, temp_dir.name)
        with open(f'{temp_dir.name}/foo.LV1', 'w') as f:
            f.write('kissa')
        _, files = hatpro.hatpro2nc(temp_dir.name, self.temp_file.name, self.site_meta,
                                    date='2021-01-23')
        assert len(files) == 2
