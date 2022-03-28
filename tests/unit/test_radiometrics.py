import sys
import os
from os import path
from tempfile import NamedTemporaryFile
import pytest
import netCDF4
from cloudnetpy.instruments import radiometrics2nc
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy_qc import Quality

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from all_products_fun import AllProductsFun


file = f"{SCRIPT_PATH}/data/radiometrics/2021-07-18_00-00-00_lv2.csv"
site_meta = {"name": "the_station", "altitude": 50, "latitude": 23.0, "longitude": 123}


class TestHatpro2nc:

    date = "2021-07-18"
    temp_file = NamedTemporaryFile()
    uuid = radiometrics2nc(file, temp_file.name, site_meta, date=date)

    def test_common(self):
        nc = netCDF4.Dataset(self.temp_file.name)
        all_fun = AllProductsFun(nc, site_meta, self.date, self.uuid)
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(all_fun, name)()
        nc.close()

    def test_qc(self):
        quality = Quality(self.temp_file.name)
        res_data = quality.check_data()
        res_metadata = quality.check_metadata()
        assert quality.n_metadata_test_failures == 0, res_metadata
        assert quality.n_data_test_failures == 0, res_data

    def test_default_processing(self):
        temp_file = NamedTemporaryFile()
        radiometrics2nc(file, temp_file.name, site_meta)

    def test_processing_of_one_file(self):
        temp_file = NamedTemporaryFile()
        radiometrics2nc(file, temp_file.name, site_meta, date=self.date)

    def test_processing_of_no_files(self):
        temp_file = NamedTemporaryFile()
        with pytest.raises(ValidTimeStampError):
            radiometrics2nc(file, temp_file.name, site_meta, date="2021-07-19")
