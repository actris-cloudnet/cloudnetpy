import os
from tempfile import TemporaryDirectory

from cloudnetpy.cloudnetarray import CloudnetArray
import pytest

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import rain_e_h32nc
from tests.unit.all_products_fun import Check
import numpy as np
from numpy import ma

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SITE_META = {
    "name": "Palaiseau",
    "latitude": 50,
    "longitude": 104.5,
    "altitude": 50,
}


class RainEH3(Check):
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"

    def test_rainfall_amount(self):
        assert self.nc.variables["rainfall_amount"][0] == 0.0
        assert (np.diff(self.nc.variables["rainfall_amount"][:]) >= 0).all()

    def test_global_attributes(self):
        assert self.nc.cloudnet_file_type == "rain-gauge"
        assert self.nc.title == f"rain[e]H3 rain-gauge from {self.site_meta['name']}"
        assert self.nc.source == "LAMBRECHT meteo GmbH rain[e]H3"
        assert self.nc.year == self.date[:4]
        assert self.nc.month == self.date[5:7]
        assert self.nc.day == self.date[8:10]
        assert self.nc.location == self.site_meta["name"]


class TestRainEH3(RainEH3):
    date = "2024-12-31"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/rain_e_h3/20241231_raine_lindenberg.csv"
    uuid = rain_e_h32nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 22


class TestRainEH3File2(RainEH3):
    date = "2023-05-14"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/rain_e_h3/Lindenberg_RainE_20230514.txt"
    uuid = rain_e_h32nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 18


class TestDateArgument(RainEH3):
    date = "2024-12-31"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/rain_e_h3/20241231_raine_lindenberg.csv"
    site_meta = SITE_META
    uuid = rain_e_h32nc(filename, temp_path, site_meta, date=date)

    def test_invalid_date(self):
        with pytest.raises(ValidTimeStampError):
            rain_e_h32nc(
                self.filename,
                self.temp_path,
                SITE_META,
                date="2022-01-05",
            )
