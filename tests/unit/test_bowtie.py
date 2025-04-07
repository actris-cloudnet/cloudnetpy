from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import bowtie2nc
from tests.unit.all_products_fun import Check
from tests.unit.radar_fun import RadarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/bowtie/bowtie-trunc.nc"


class TestCopernicus2nc(Check):
    site_meta = {
        "altitude": 16,
        "latitude": 6.1,
        "longitude": -25.9,
        "name": "RV Meteor",
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/bowtie.nc"
    uuid = bowtie2nc(FILEPATH, temp_path, site_meta)
    date = "2024-08-22"

    def test_variables(self):
        assert np.isclose(
            self.nc.variables["radar_frequency"][:].data,
            94,
        )

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_global_attributes(self):
        assert self.nc.source == "RPG-Radiometer Physics RPG-FMCW-94"
        assert self.nc.title == f'RPG-FMCW-94 cloud radar from {self.site_meta["name"]}'

    def test_range(self):
        for key in ("range", "height"):
            assert np.all(self.nc.variables[key][:] > 0)

    def test_correct_date_validation(self, tmp_path):
        test_path = tmp_path / "date.nc"
        bowtie2nc(FILEPATH, test_path, self.site_meta, date=self.date)

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            bowtie2nc(
                FILEPATH,
                test_path,
                self.site_meta,
                date="2021-01-03",
            )
