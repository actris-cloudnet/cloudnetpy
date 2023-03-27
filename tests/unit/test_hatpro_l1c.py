from os import path
from tempfile import TemporaryDirectory

import netCDF4

from cloudnetpy.instruments import hatpro
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
file_path = f"{SCRIPT_PATH}/data/hatpro-mwrpy/"


class TestHatpro2nc(Check):
    site_meta = {
        "name": "the_station",
        "altitude": 50,
        "latitude": 23.0,
        "longitude": 123,
        "coeffs_dir": "hyytiala",
    }
    date = "2023-04-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    uuid = hatpro.hatpro2l1c(file_path, temp_path, site_meta, date=date)

    def test_default_processing(self, tmp_path):
        with netCDF4.Dataset(self.temp_path) as nc:
            assert nc.cloudnet_file_type == "mwr-l1c"
