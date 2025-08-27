from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import mrr2nc
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
filename = f"{SCRIPT_PATH}/data/mrr/20220124_180000.nc"


class TestMrrPro(Check):
    site_meta = {
        "name": "Palaiseau",
        "latitude": 50,
        "longitude": 104.5,
        "altitude": 50,
    }
    date = "2022-01-24"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    uuid = mrr2nc(filename, temp_path, site_meta)

    def test_variable_names(self):
        keys = {
            "Zh",
            "v",
            "width",
            "pia",
            "lwc",
            "rainfall_rate",
            "time",
            "range",
            "radar_frequency",
            "height",
            "latitude",
            "longitude",
            "altitude",
            "zenith_angle",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert self.nc.variables["radar_frequency"].units == "GHz"
        assert (
            abs(self.nc.variables["radar_frequency"][:].data - 24.23) < 0.001
        )  # Hard coded
        assert np.all(self.nc.variables["altitude"][:] == 50)

    def test_global_attributes(self):
        assert self.nc.source == "METEK MRR-PRO"
        assert self.nc.title == "MRR-PRO rain radar from Palaiseau"
        assert self.nc.serial_number == "0511107367"

    def test_wrong_date(self, tmp_path):
        with pytest.raises(ValidTimeStampError):
            mrr2nc(
                filename,
                tmp_path / "invalid.nc",
                self.site_meta,
                date="2021-01-04",
            )

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "e58134f9-073d-4b10-b6c9-a91cf2229322"
        uuid = mrr2nc(filename, test_path, self.site_meta, uuid=uuid_from_user)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert uuid == uuid_from_user
