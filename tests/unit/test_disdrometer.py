import os
from cloudnetpy.instruments import disdrometer
import pytest
import netCDF4
from cloudnetpy_qc import Quality
from tempfile import NamedTemporaryFile

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def test_format_time():
    assert disdrometer._format_thies_date("3.10.20") == "2020-10-03"


class TestParsivel:
    temp_file = NamedTemporaryFile()
    site_meta = {"name": "Kumpula"}
    filename = f"{SCRIPT_PATH}/data/parsivel/juelich.log"
    uuid = disdrometer.disdrometer2nc(filename, temp_file.name, site_meta)

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    def test_global_attributes(self):
        assert "Parsivel" in self.nc.source
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.title == "Disdrometer file from Kumpula"
        assert self.nc.year == "2021"
        assert self.nc.month == "03"
        assert self.nc.day == "18"
        assert self.nc.location == "Kumpula"

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size > 1000
        assert self.nc.dimensions["velocity"].size == 32
        assert self.nc.dimensions["diameter"].size == 32

    def test_qc(self):
        check_qc(self.temp_file.name)


class TestParsivel2:
    temp_file = NamedTemporaryFile()
    filename = f"{SCRIPT_PATH}/data/parsivel/norunda.log"
    site_meta = {"name": "Norunda"}

    def test_date_validation(self):
        disdrometer.disdrometer2nc(
            self.filename, self.temp_file.name, self.site_meta, date="2019-11-09"
        )

    def test_date_validation_fail(self):
        temp_file = NamedTemporaryFile()
        with pytest.raises(ValueError):
            disdrometer.disdrometer2nc(
                self.filename, temp_file.name, self.site_meta, date="2022-04-05"
            )

    def test_qc(self):
        check_qc(self.temp_file.name)


class TestParsivel3:
    temp_file = NamedTemporaryFile()
    filename = f"{SCRIPT_PATH}/data/parsivel/ny-alesund.log"
    site_meta = {"name": "Ny Alesund"}
    disdrometer.disdrometer2nc(filename, temp_file.name, site_meta, date="2021-04-16")

    def test_qc(self):
        check_qc(self.temp_file.name)


class TestThies:
    temp_file = NamedTemporaryFile()
    filename = f"{SCRIPT_PATH}/data/thies-lnm/2021091507.txt"
    site_meta = {"name": "Lindenberg", "latitude": 34.6, "altitude": 20}
    uuid = disdrometer.disdrometer2nc(filename, temp_file.name, site_meta, date="2021-09-15")

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    def test_processing(self):
        assert self.nc.title == "Disdrometer file from Lindenberg"
        assert self.nc.year == "2021"
        assert self.nc.month == "09"
        assert self.nc.day == "15"
        assert self.nc.location == "Lindenberg"
        assert self.nc.cloudnet_file_type == "disdrometer"

    def test_qc(self):
        check_qc(self.temp_file.name)


def check_qc(filename: str):
    quality = Quality(filename)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()
    assert quality.n_metadata_test_failures == 0, res_metadata
    assert quality.n_data_test_failures == 0, res_data
