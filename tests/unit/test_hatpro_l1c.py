import glob
from os import path
from tempfile import TemporaryDirectory

import netCDF4

from cloudnetpy.instruments import hatpro
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
file_path = f"{SCRIPT_PATH}/data/hatpro-mwrpy/"


class TestHatpro2nc(Check):
    coeff_files = glob.glob(f"{SCRIPT_PATH}/data/hatpro-mwrpy-coeffs/*.ret")

    site_meta = {
        "name": "hyytiala",
        "latitude": 61.844,
        "longitude": 24.287,
        "altitude": 150,
        "coefficientFiles": coeff_files,
        "coefficientLinks": coeff_files,
    }

    date = "2023-04-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    uuid = hatpro.hatpro2l1c(file_path, temp_path, site_meta, date=date)

    def test_default_processing(self, tmp_path):
        with netCDF4.Dataset(self.temp_path) as nc:
            assert nc.cloudnet_file_type == "mwr-l1c"
