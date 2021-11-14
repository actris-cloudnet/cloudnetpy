from os import path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from cloudnetpy.instruments import hatpro
from distutils.dir_util import copy_tree
import netCDF4

SCRIPT_PATH = path.dirname(path.realpath(__file__))

file_path = f'{SCRIPT_PATH}/data/hatpro/'
site_meta = {
    'name': 'the_station',
    'altitude': 50
}


class TestHatpro2nc:

    temp_file = NamedTemporaryFile()

    def test_default_processing(self):
        uuid, files = hatpro.hatpro2nc(file_path, self.temp_file.name, site_meta)
        assert len(files) == 4
        assert len(uuid) == 32

    def test_processing_of_several_files(self):
        test_uuid = 'abc'
        uuid, files = hatpro.hatpro2nc(file_path, self.temp_file.name, site_meta, date='2021-01-23',
                                       uuid=test_uuid)
        assert len(files) == 2
        assert uuid == test_uuid
        nc = netCDF4.Dataset(self.temp_file.name)
        time = nc.variables['time']
        assert 'hours since' in time.units
        assert max(time[:]) < 24
        for ind, t in enumerate(time[:-1]):
            assert(time[ind+1] > t)
        assert 'LWP' in nc.variables
        assert 'g m-2' in nc.variables['LWP'].units
        nc.close()

    def test_processing_of_one_file(self):
        _, files = hatpro.hatpro2nc(file_path, self.temp_file.name, site_meta, date='2020-07-23')
        assert len(files) == 1

    def test_processing_of_no_files(self):
        _, files = hatpro.hatpro2nc(file_path, self.temp_file.name, site_meta, date='2020-10-24')
        assert len(files) == 0

    def test_handling_of_corrupted_files(self):
        temp_dir = TemporaryDirectory()
        copy_tree(file_path, temp_dir.name)
        with open(f'{temp_dir.name}/foo.LV1', 'w') as f:
            f.write('kissa')
        _, files = hatpro.hatpro2nc(temp_dir.name, self.temp_file.name, site_meta, date='2021-01-23')
        assert len(files) == 2
