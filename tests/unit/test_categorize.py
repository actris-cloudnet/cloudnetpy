import os
import sys
import warnings
from tempfile import TemporaryDirectory

import netCDF4

from cloudnetpy.categorize import generate_categorize
from cloudnetpy.instruments import ceilo2nc, mira2nc
from tests.unit.all_products_fun import Check

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
filepath = f"{SCRIPT_PATH}/../source_data"


class TestCategorize(Check):
    date = "2021-11-20"
    site_meta = {
        "name": "Munich",
        "altitude": 538,
        "latitude": 48.5,
        "longitude": 11.5,
    }

    temp_dir = TemporaryDirectory()
    radar_path = temp_dir.name + "/radar.nc"
    lidar_path = temp_dir.name + "/lidar.nc"

    uuid_radar = mira2nc(f"{filepath}/raw_mira_radar.mmclx", radar_path, site_meta)
    uuid_lidar = ceilo2nc(f"{filepath}/raw_chm15k_lidar.nc", lidar_path, site_meta)

    input_files = {
        "radar": radar_path,
        "lidar": lidar_path,
        "mwr": f"{filepath}/hatpro_mwr.nc",
        "model": f"{filepath}/ecmwf_model.nc",
    }

    temp_path = temp_dir.name + "/categorize.nc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        uuid = generate_categorize(input_files, temp_path)

    def test_global_attributes(self):
        with netCDF4.Dataset(self.temp_path) as nc:
            assert nc.title == "Cloud categorization products from Munich"
