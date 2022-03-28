import sys
from os import path
import pytest
from cloudnetpy.instruments import basta2nc
import netCDF4
from cloudnetpy_qc import Quality
from cloudnetpy.exceptions import ValidTimeStampError
from tempfile import NamedTemporaryFile

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from radar_fun import RadarFun
from all_products_fun import AllProductsFun

site_meta = {"name": "Palaiseau", "latitude": 50, "longitude": 104.5, "altitude": 50}
filename = f"{SCRIPT_PATH}/data/basta/basta_1a_cldradLz1R025m_v03_20210827_000000.nc"


class TestBASTA:
    date = "2021-08-27"
    temp_file = NamedTemporaryFile()
    uuid = basta2nc(filename, temp_file.name, site_meta)
    quality = Quality(temp_file.name)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    def test_variable_names(self):
        keys = {
            "Zh",
            "v",
            "time",
            "range",
            "radar_frequency",
            "height",
            "nyquist_velocity",
            "latitude",
            "longitude",
            "altitude",
            "zenith_angle",
            "radar_pitch",
            "radar_yaw",
            "radar_roll",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert self.nc.variables["radar_frequency"][:].data == 95.0  # Hard coded

    def test_common(self):
        all_fun = AllProductsFun(self.nc, site_meta, self.date, self.uuid)
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(all_fun, name)()

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_geolocation_from_source_file(self):
        temp_file = NamedTemporaryFile()
        meta_without_geolocation = {"name": "Kumpula"}
        basta2nc(filename, temp_file.name, meta_without_geolocation)
        nc = netCDF4.Dataset(temp_file.name)
        for key in ("latitude", "longitude", "altitude"):
            assert key in nc.variables
            assert nc.variables[key][:] > 0
        nc.close()

    def test_global_attributes(self):
        assert self.nc.source == "BASTA"
        assert self.nc.title == f'BASTA cloud radar from {site_meta["name"]}'

    def test_wrong_date_validation(self):
        temp_file = NamedTemporaryFile()
        with pytest.raises(ValidTimeStampError):
            basta2nc(filename, temp_file.name, site_meta, date="2021-01-04")

    def test_uuid_from_user(self):
        temp_file = NamedTemporaryFile()
        uuid_from_user = "kissa"
        uuid = basta2nc(filename, temp_file.name, site_meta, uuid=uuid_from_user)
        nc = netCDF4.Dataset(temp_file.name)
        assert nc.file_uuid == uuid_from_user
        assert uuid == uuid_from_user
        nc.close()
