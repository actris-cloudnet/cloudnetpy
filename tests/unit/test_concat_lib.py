import numpy as np
from os import path
import os
import pytest
from cloudnetpy import concat_lib
from tempfile import NamedTemporaryFile
import netCDF4
import glob


SCRIPT_PATH = path.dirname(path.realpath(__file__))


class TestUpdateNc:

    files = glob.glob(f'{SCRIPT_PATH}/data/cl61d/*.nc')
    files.sort()

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.filename = 'dummy_test_file.nc'
        yield
        os.remove(self.filename)

    def test_does_append_to_end(self):
        concat_lib.concatenate_files(self.files[:2], self.filename, concat_dimension='profile')
        succ = concat_lib.update_nc(self.filename, self.files[2])
        assert succ == 1
        nc = netCDF4.Dataset(self.filename)
        time = nc.variables['time'][:]
        assert len(time) == 3 * 12
        for ind, timestamp in enumerate(time[:-1]):
            assert timestamp < time[ind+1]

    def test_does_not_append_to_beginning(self):
        concat_lib.concatenate_files(self.files[1:3], self.filename, concat_dimension='profile')
        succ = concat_lib.update_nc(self.filename, self.files[0])
        assert succ == 0
        nc = netCDF4.Dataset(self.filename)
        time = nc.variables['time'][:]
        assert len(time) == 2 * 12
        for ind, timestamp in enumerate(time[:-1]):
            assert timestamp < time[ind+1]

    def test_does_not_append_to_middle(self):
        files = [self.files[0], self.files[2]]
        concat_lib.concatenate_files(files, self.filename, concat_dimension='profile')
        succ = concat_lib.update_nc(self.filename, self.files[1])
        assert succ == 0
        nc = netCDF4.Dataset(self.filename)
        time = nc.variables['time'][:]
        assert len(time) == 2 * 12
        for ind, timestamp in enumerate(time[:-1]):
            assert timestamp < time[ind+1]


class TestConcat:

    files = [f'{SCRIPT_PATH}/data/chm15k/00100_A202010222015_CHM170137.nc',
             f'{SCRIPT_PATH}/data/chm15k/00100_A202010220005_CHM170137.nc']

    n_time = 10
    n_range = 1024
    n_range_hr = 32
    n_layer = 3

    output_file = NamedTemporaryFile()

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.concat = concat_lib.Concat(self.files, self.output_file.name)
        self.file = self.concat.concatenated_file
        yield
        self.concat.close()

    @pytest.mark.parametrize("size, result", [
        (np.zeros((n_time, n_range)), ('time', 'range')),
        (np.zeros((n_range, n_time)), ('range', 'time')),
        (np.zeros((n_range,)), ('range',)),
        (np.zeros((n_time,)), ('time',)),
    ])
    def test_get_dim(self, size, result, nc_file):
        assert self.concat._get_dim(size) == result

    def test_sorting_input_files(self):
        assert self.concat.filenames[0] == self.files[1]
        assert self.concat.filenames[1] == self.files[0]

    def test_create_dimension(self):
        for dim in ('time', 'range', 'range_hr', 'layer'):
            assert dim in self.file.dimensions

    def test_create_constants(self):
        self.concat.get_constants()
        for var in ('range', 'range_hr', 'layer', 'latitude', 'longitude'):
            assert var in self.concat.constants
        for var in ('time', 'life_time', 'beta_raw'):
            assert var not in self.concat.constants

    def test_create_global_attributes(self):
        self.concat.create_global_attributes(new_attributes={'kissa': 50, 'koira': '23'})
        for attr in ('day', 'title', 'month', 'comment', 'kissa', 'koira'):
            assert hasattr(self.file, attr)
        assert self.file.kissa == 50
        assert self.file.koira == '23'

    def test_concat_data(self):
        self.concat.get_constants()
        self.concat.concat_data()
        assert len(self.file.variables['time']) == 2 * self.n_time
        assert len(self.file.variables['range']) == self.n_range
        assert len(self.file.variables['layer']) == self.n_layer

    def test_concat_only_some_variables_data(self):
        self.concat.get_constants()
        variables = ['cbh', 'sci']
        self.concat.concat_data(variables)
        assert len(self.file.variables['time']) == 2 * self.n_time
        assert len(self.file.variables['range']) == self.n_range
        for var in variables:
            assert var in self.file.variables
        for var in ('cde', 'nn3'):
            assert var not in self.file.variables


def test_concatenate_files_with_mira():
    files = [f'{SCRIPT_PATH}/data/mira/20210102_1400.mmclx',
             f'{SCRIPT_PATH}/data/mira/20210102_0000.mmclx']
    output_file = NamedTemporaryFile()
    variables = ['microsec', 'SNRCorFaCo']
    concat_lib.concatenate_files(files, output_file.name, variables=variables,
                                 new_attributes={'kissa': 50})
    nc = netCDF4.Dataset(output_file.name)
    assert len(nc.variables['time']) == 145 + 146
    assert len(nc.variables['range']) == 413
    assert nc.data_model == 'NETCDF4_CLASSIC'
    for var in ('prf', 'microsec'):
        assert var in nc.variables
    for var in ('VELg', 'elv'):
        assert var not in nc.variables
    assert nc.kissa == 50
    nc.close()
