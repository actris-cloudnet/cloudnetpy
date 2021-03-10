from os import path
import pytest
from cloudnetpy.instruments import mira
from tempfile import NamedTemporaryFile
import netCDF4


SCRIPT_PATH = path.dirname(path.realpath(__file__))


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

    file_path = f'{SCRIPT_PATH}/data/mira/'

    n_time1 = 146
    n_time2 = 145

    site_meta = {
        'name': 'The_station',
        'longitude': 104.3,
        'altitude': 50.2
    }

    @pytest.fixture(autouse=True)
    def _init(self):
        self.temp_file = NamedTemporaryFile()

    def test_processing_of_several_nc_files(self):
        mira.mira2nc(self.file_path, self.temp_file.name, self.site_meta)
        nc = netCDF4.Dataset(self.temp_file.name)
        assert len(nc.variables['time'][:]) == self.n_time1 + self.n_time2
        nc.close()

    def test_processing_of_one_nc_file(self):
        mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', self.temp_file.name, self.site_meta)
        nc = netCDF4.Dataset(self.temp_file.name)
        assert len(nc.variables['time'][:]) == self.n_time1
        nc.close()

    def test_correct_date_validation(self):
        mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', self.temp_file.name,
                     self.site_meta, date='2021-01-02')

    def test_wrong_date_validation(self):
        with pytest.raises(ValueError):
            mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', self.temp_file.name,
                         self.site_meta, date='2021-01-03')

    def test_uuid_from_user(self):
        uuid_from_user = 'kissa'
        uuid = mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', self.temp_file.name,
                            self.site_meta, uuid=uuid_from_user)
        nc = netCDF4.Dataset(self.temp_file.name)
        assert nc.file_uuid == uuid_from_user
        assert uuid == uuid_from_user
        nc.close()

    def test_global_attributes(self):
        mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', self.temp_file.name, self.site_meta)
        nc = netCDF4.Dataset(self.temp_file.name)
        assert nc.year == '2021'
        assert nc.month == '01'
        assert nc.day == '02'
        assert nc.Conventions == 'CF-1.7'
        assert nc.source == 'METEK MIRA-35'
        assert nc.location == 'The_station'
        assert nc.title == 'Radar file from The_station'
        assert nc.cloudnet_file_type == 'radar'
        nc.close()

    def test_variables(self):
        mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', self.temp_file.name, self.site_meta)
        nc = netCDF4.Dataset(self.temp_file.name)
        vars = ('Ze', 'v', 'width', 'ldr', 'SNR', 'time', 'range', 'radar_frequency',
                'nyquist_velocity', 'latitude', 'longitude', 'altitude')
        for var in vars:
            assert var in nc.variables
        assert nc.variables['radar_frequency'][:].data == 35.5  # Hard coded
        assert abs(nc.variables['latitude'][:].data-50.9085) < 0.1  # From input file
        for var in ('altitude', 'longitude'):
            assert abs(nc.variables[var][:]-self.site_meta[var]) < 0.1  # From user
        time = nc.variables['time'][:]
        assert min(time) > 0
        assert max(time) < 24
        nc.close()
