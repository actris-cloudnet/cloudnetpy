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
    output = "dummy_radiometrics_output_file.nc"
    uuid = radiometrics2nc(file, output, site_meta, date=date)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()
    nc = netCDF4.Dataset(output)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)
    temp_file = NamedTemporaryFile()

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(self.all_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_default_processing(self):
        radiometrics2nc(file, self.temp_file.name, site_meta)

    def test_processing_of_one_file(self):
        radiometrics2nc(file, self.temp_file.name, site_meta, date=self.date)

    def test_processing_of_no_files(self):
        with pytest.raises(ValidTimeStampError):
            radiometrics2nc(file, self.temp_file.name, site_meta, date="2021-07-19")

    def test_cleanup(self):
        os.remove(self.output)
        self.nc.close()
