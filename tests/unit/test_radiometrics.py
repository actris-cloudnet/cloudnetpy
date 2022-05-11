import sys
from os import path
from tempfile import TemporaryDirectory

import pytest

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import radiometrics2nc

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from all_products_fun import Check

file = f"{SCRIPT_PATH}/data/radiometrics/2021-07-18_00-00-00_lv2.csv"


class TestHatpro2nc(Check):
    site_meta = {"name": "the_station", "altitude": 50, "latitude": 23.0, "longitude": 123}
    date = "2021-07-18"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(file, temp_path, site_meta, date=date)

    def test_default_processing(self, tmp_path):
        test_path = tmp_path / "default.nc"
        radiometrics2nc(file, test_path, self.site_meta)

    def test_processing_of_one_file(self, tmp_path):
        test_path = tmp_path / "one.nc"
        radiometrics2nc(file, test_path, self.site_meta, date=self.date)

    def test_processing_of_no_files(self, tmp_path):
        test_path = tmp_path / "no.nc"
        with pytest.raises(ValidTimeStampError):
            radiometrics2nc(file, test_path, self.site_meta, date="2021-07-19")
