import os
from tempfile import TemporaryDirectory

import pytest
from numpy.testing import assert_array_equal
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import fd12p2nc
from tests.unit.all_products_fun import Check
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SITE_META = {
    "name": "Lindenberg",
    "latitude": 52.208,
    "longitude": 14.118,
    "altitude": 104,
}


class FD12P(Check):
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"

    def test_rainfall_amount(self):
        assert self.nc.variables["precipitation_amount"][0] == 0.0
        assert (np.diff(self.nc.variables["precipitation_amount"][:]) >= 0).all()

    def test_global_attributes(self):
        assert self.nc.cloudnet_file_type == "weather-station"
        assert (
            self.nc.title
            == f"FD12P present weather sensor from {self.site_meta['name']}"
        )
        assert self.nc.source == "Vaisala FD12P"
        assert self.nc.year == self.date[:4]
        assert self.nc.month == self.date[5:7]
        assert self.nc.day == self.date[8:10]
        assert self.nc.location == self.site_meta["name"]


class TestFD12P_1(FD12P):
    date = "2004-04-06"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/fd12p/PW040406.DAT"
    uuid = fd12p2nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 3


class TestFD12P_2(FD12P):
    date = "2004-07-12"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/fd12p/PW040712.DAT"
    uuid = fd12p2nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 2

    def test_visibility(self):
        assert_array_equal(self.nc["visibility"], [3430, 3693])


class TestFD12P_3(FD12P):
    date = "2013-01-21"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    temp_path2 = temp_dir.name + "/test2.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/fd12p/PW130121.DAT"
    uuid = fd12p2nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 2

    def test_invalid_date(self):
        with pytest.raises(ValidTimeStampError):
            fd12p2nc(
                self.filename,
                self.temp_path2,
                SITE_META,
                date="2022-01-05",
            )
