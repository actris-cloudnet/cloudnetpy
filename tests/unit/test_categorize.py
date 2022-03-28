import os
import sys
from tempfile import NamedTemporaryFile
import pytest
from cloudnetpy.instruments import ceilo2nc, mira2nc
from cloudnetpy.categorize import generate_categorize
import netCDF4
import warnings
from cloudnetpy_qc import Quality

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from all_products_fun import AllProductsFun

site_meta = {"name": "Munich", "altitude": 538, "latitude": 48.5, "longitude": 11.5}
filepath = f"{SCRIPT_PATH}/../source_data/"
date = "2021-11-20"


class TestCategorize:

    radar_file = NamedTemporaryFile()
    lidar_file = NamedTemporaryFile()

    uuid_radar = mira2nc(f"{filepath}raw_mira_radar.mmclx", radar_file.name, site_meta)
    uuid_lidar = ceilo2nc(f"{filepath}raw_chm15k_lidar.nc", lidar_file.name, site_meta)

    input_files = {
        "radar": radar_file.name,
        "lidar": lidar_file.name,
        "mwr": f"{filepath}hatpro_mwr.nc",
        "model": f"{filepath}ecmwf_model.nc",
    }

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    temp_file = NamedTemporaryFile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        uuid = generate_categorize(input_files, temp_file.name)

    def test_common(self):
        all_fun = AllProductsFun(self.nc, site_meta, date, self.uuid)
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(all_fun, name)()

    def test_global_attributes(self):
        nc = netCDF4.Dataset(self.temp_file.name)
        assert nc.title == "Cloud categorization products from Munich"
        nc.close()

    def test_qc(self):
        quality = Quality(self.temp_file.name)
        res_data = quality.check_data()
        res_metadata = quality.check_metadata()
        assert quality.n_metadata_test_failures == 0, res_metadata
        assert quality.n_data_test_failures == 0, res_data
