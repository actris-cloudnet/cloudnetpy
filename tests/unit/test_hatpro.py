import sys
import os
from os import path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pytest
from distutils.dir_util import copy_tree
import netCDF4
from cloudnetpy.instruments import hatpro
from cloudnetpy_qc import Quality
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy_qc import Quality

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from all_products_fun import AllProductsFun


file_path = f'{SCRIPT_PATH}/data/hatpro/'
site_meta = {
    'name': 'the_station',
    'altitude': 50,
    'latitude': 23.0,
    'longitude': 123
}


class TestHatpro2nc:

    date = '2020-07-23'
    output = 'dummy_hatpro_output_file.nc'
    uuid, valid_files = hatpro.hatpro2nc(file_path, output, site_meta, date=date)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()
    nc = netCDF4.Dataset(output)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)

    temp_file = NamedTemporaryFile()

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if 'test_' in name:
                getattr(self.all_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_default_processing(self):
        uuid, files = hatpro.hatpro2nc(file_path, self.temp_file.name, site_meta)
        assert len(files) == 4
        assert len(uuid) == 36

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
        assert 'lwp' in nc.variables
        assert 'g m-2' in nc.variables['lwp'].units
        nc.close()

    def test_processing_of_one_file(self):
        date = '2020-07-23'
        uuid, files = hatpro.hatpro2nc(file_path, self.temp_file.name, site_meta, date=date)
        assert len(files) == 1

    def test_processing_of_no_files(self):
        with pytest.raises(ValidTimeStampError):
            hatpro.hatpro2nc(file_path, self.temp_file.name, site_meta, date='2020-10-24')

    def test_handling_of_corrupted_files(self):
        temp_dir = TemporaryDirectory()
        copy_tree(file_path, temp_dir.name)
        with open(f'{temp_dir.name}/foo.LV1', 'w') as f:
            f.write('kissa')
        _, files = hatpro.hatpro2nc(temp_dir.name, self.temp_file.name, site_meta,
                                    date='2021-01-23')
        assert len(files) == 2

    def test_cleanup(self):
        os.remove(self.output)
        self.nc.close()
