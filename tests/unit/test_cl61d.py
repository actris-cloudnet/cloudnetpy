import os
import glob
from cloudnetpy import concat_lib
from cloudnetpy.instruments import ceilo2nc
import pytest
import netCDF4

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class TestCl61d:

    site_meta = {
        'name': 'Hyytiälä',
        'altitude': 123,
        'calibration_factor': 2.0
    }
    files = glob.glob(f'{SCRIPT_PATH}/data/cl61d/*.nc')
    files.sort()

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.output = 'dummy_output_file.nc'
        self.filename = 'dummy_daily_file.nc'
        concat_lib.concatenate_files(self.files, self.filename, concat_dimension='profile')
        yield
        os.remove(self.filename)
        os.remove(self.output)

    def test_variables(self):
        ceilo2nc(self.filename, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        for key in ('beta', 'depolarisation', 'beta_smooth', 'depolarisation_smooth'):
            assert key in nc.variables
        for key in ('beta_raw', 'depolarisation_raw'):
            assert key not in nc.variables
        for key in ('altitude', 'calibration_factor'):
            assert nc.variables[key][:] == self.site_meta[key]
        nc.close()

    def test_global_attributes(self):
        uuid = ceilo2nc(self.filename, self.output, self.site_meta)
        nc = netCDF4.Dataset(self.output)
        assert nc.source == 'Vaisala CL61d'
        assert nc.location == self.site_meta['name']
        assert nc.title == f'Ceilometer file from {self.site_meta["name"]}'
        assert nc.file_uuid == uuid
        assert nc.cloudnet_file_type == 'lidar'
        nc.close()

    def test_date_argument(self):
        ceilo2nc(self.filename, self.output, self.site_meta, date='2021-08-30')
        nc = netCDF4.Dataset(self.output)
        assert len(nc.variables['time']) == 12
        assert nc.year == '2021'
        assert nc.month == '08'
        assert nc.day == '30'
        nc.close()
