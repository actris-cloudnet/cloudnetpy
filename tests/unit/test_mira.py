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

    def test_validate_date_default(self):
        self.raw_radar.validate_date(None)
        assert self.raw_radar.date == self.correct_date

    def test_validate_date(self):
        self.raw_radar.validate_date('2020-05-24')
        assert self.raw_radar.date == self.correct_date

    def test_validate_date_fails(self):
        with pytest.raises(ValueError):
            self.raw_radar.validate_date('2020-05-23')


class TestMIRA2nc:

    file_path = f'{SCRIPT_PATH}/data/mira/'

    n_time1 = 146
    n_time2 = 145

    site_meta = {
        'name': 'the_station',
        'altitude': 50
    }

    def test_processing_of_several_nc_files(self):
        temp_file = NamedTemporaryFile()
        mira.mira2nc(self.file_path, temp_file.name, self.site_meta)
        nc = netCDF4.Dataset(temp_file.name)
        assert len(nc.variables['time'][:]) == self.n_time1 + self.n_time2
        nc.close()

    def test_processing_of_one_nc_file(self):
        temp_file = NamedTemporaryFile()
        mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', temp_file.name, self.site_meta)
        nc = netCDF4.Dataset(temp_file.name)
        assert len(nc.variables['time'][:]) == self.n_time1
        nc.close()

    def test_correct_date_validation(self):
        temp_file = NamedTemporaryFile()
        mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', temp_file.name,
                     self.site_meta, date='2021-01-02')

    def test_wrong_date_validation(self):
        temp_file = NamedTemporaryFile()
        with pytest.raises(ValueError):
            mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', temp_file.name,
                         self.site_meta, date='2021-01-03')

    def test_uuid_from_user(self):
        uuid_from_user = 'kissa'
        temp_file = NamedTemporaryFile()
        uuid = mira.mira2nc(f'{self.file_path}/20210102_0000.mmclx', temp_file.name,
                            self.site_meta, uuid=uuid_from_user)
        nc = netCDF4.Dataset(temp_file.name)
        assert nc.file_uuid == uuid_from_user
        assert uuid == uuid_from_user
        nc.close()
