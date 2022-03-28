import sys
from os import path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pytest
from distutils.dir_util import copy_tree
import netCDF4
from cloudnetpy.instruments import hatpro
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy_qc import Quality

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from all_products_fun import AllProductsFun


file_path = f"{SCRIPT_PATH}/data/hatpro/"
site_meta = {"name": "the_station", "altitude": 50, "latitude": 23.0, "longitude": 123}


class TestHatpro2nc:

    date = "2020-07-23"
    temp_file = NamedTemporaryFile()
    uuid, valid_files = hatpro.hatpro2nc(file_path, temp_file.name, site_meta, date=date)

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
        uuid, files = hatpro.hatpro2nc(file_path, temp_file.name, site_meta)
        assert len(files) == 4
        assert len(uuid) == 36

    def test_processing_of_several_files(self):
        test_uuid = "abc"
        temp_file = NamedTemporaryFile()
        uuid, files = hatpro.hatpro2nc(
            file_path, temp_file.name, site_meta, date="2021-01-23", uuid=test_uuid
        )
        assert len(files) == 2
        assert uuid == test_uuid
        nc = netCDF4.Dataset(temp_file.name)
        time = nc.variables["time"]
        assert "hours since" in time.units
        assert max(time[:]) < 24
        for ind, t in enumerate(time[:-1]):
            assert time[ind + 1] > t
        assert "lwp" in nc.variables
        assert "g m-2" in nc.variables["lwp"].units
        nc.close()

    def test_processing_of_one_file(self):
        temp_file = NamedTemporaryFile()
        date = "2020-07-23"
        uuid, files = hatpro.hatpro2nc(file_path, temp_file.name, site_meta, date=date)
        assert len(files) == 1

    def test_processing_of_no_files(self):
        temp_file = NamedTemporaryFile()
        with pytest.raises(ValidTimeStampError):
            hatpro.hatpro2nc(file_path, temp_file.name, site_meta, date="2020-10-24")

    def test_handling_of_corrupted_files(self):
        temp_dir = TemporaryDirectory()
        copy_tree(file_path, temp_dir.name)
        with open(f"{temp_dir.name}/foo.LV1", "w") as f:
            f.write("kissa")
        temp_file = NamedTemporaryFile()
        _, files = hatpro.hatpro2nc(temp_dir.name, temp_file.name, site_meta, date="2021-01-23")
        assert len(files) == 2
