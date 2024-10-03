import os
from datetime import timedelta
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.instruments import disdrometer
from tests.unit.all_products_fun import Check

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SITE_META = {
    "name": "Kumpula",
    "latitude": 50,
    "longitude": 104.5,
    "altitude": 50,
}

# fmt: off
TELEGRAM = [19, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13,
            14, 16, 17, 18, 22, 24, 25, 90, 91, 93]
TELEGRAM2 = [19, 1] + [None] * 16 + [90, 91, 93]
TELEGRAM3 = [21, 20, 1, 2, 3, 5, 6, 7, 8, 10,
            11, 12, 16, 17, 34, 18, 93]
# fmt: on


class TestParsivel(Check):
    date = "2021-03-18"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/parsivel/juelich.log"
    uuid = disdrometer.parsivel2nc(
        filename,
        temp_path,
        site_meta,
        telegram=TELEGRAM,
    )

    def test_global_attributes(self):
        assert "Parsivel" in self.nc.source
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.title == f'Parsivel2 disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2021"
        assert self.nc.month == "03"
        assert self.nc.day == "18"
        assert self.nc.location == "Kumpula"
        assert self.nc.serial_number == "403479"

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size > 1000
        assert self.nc.dimensions["velocity"].size == 32
        assert self.nc.dimensions["diameter"].size == 32

    def test_variables(self):
        assert "rainfall_rate" in self.nc.variables
        assert "radar_reflectivity" in self.nc.variables
        assert "visibility" in self.nc.variables
        assert "T_sensor" in self.nc.variables
        assert "I_heating" in self.nc.variables
        assert "V_power_supply" in self.nc.variables


class TestParsivelUnknownValue(Check):
    date = "2021-03-18"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/parsivel/juelich.log"
    uuid = disdrometer.parsivel2nc(filename, temp_path, site_meta, telegram=TELEGRAM2)

    def test_global_attributes(self):
        assert "Parsivel" in self.nc.source
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.title == f'Parsivel2 disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2021"
        assert self.nc.month == "03"
        assert self.nc.day == "18"
        assert self.nc.location == "Kumpula"

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size > 1000
        assert self.nc.dimensions["velocity"].size == 32
        assert self.nc.dimensions["diameter"].size == 32

    def test_variables(self):
        assert "rainfall_rate" in self.nc.variables
        assert "radar_reflectivity" not in self.nc.variables
        assert "visibility" not in self.nc.variables
        assert "T_sensor" not in self.nc.variables
        assert "I_heating" not in self.nc.variables
        assert "V_power_supply" not in self.nc.variables


class TestParsivel2(Check):
    date = "2019-11-09"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/norunda.log"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(
        filename,
        temp_path,
        site_meta,
        date=date,
        telegram=TELEGRAM,
    )

    def test_date_validation_fail(self, tmp_path):
        with pytest.raises(DisdrometerDataError):
            disdrometer.parsivel2nc(
                self.filename,
                tmp_path / "invalid.nc",
                self.site_meta,
                date="2022-04-05",
                telegram=TELEGRAM,
            )


class TestParsivel3(Check):
    date = "2021-04-16"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/ny-alesund.log"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(
        filename,
        temp_path,
        site_meta,
        date=date,
        telegram=TELEGRAM,
    )


class TestParsivel4(Check):
    date = "2019-11-15"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/palaiseau.txt"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 3


class TestParsivel5(Check):
    date = "2021-08-06"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/warsaw.txt"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(
        filename,
        temp_path,
        site_meta,
        date=date,
        telegram=TELEGRAM3,
    )

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 3


class TestParsivel6(Check):
    date = "2021-02-08"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/granada.dat"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 3


class TestParsivel7:
    date = "2023-10-25"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/bucharest_0000000123_20231025221800.txt"
    site_meta = SITE_META
    with pytest.raises(DisdrometerDataError):
        disdrometer.parsivel2nc(filename, temp_path, site_meta, date=date)


class TestParsivel8(Check):
    date = "2023-12-04"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/Lindenberg_Parsivel_20231204.log"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(
        filename,
        temp_path,
        site_meta,
        date=date,
        telegram=TELEGRAM,
    )

    def test_dimensions(self):
        assert self.nc.serial_number == "451221"
        assert np.allclose(
            self.nc["time"][:],
            [
                timedelta(hours=0, minutes=0, seconds=47) / timedelta(hours=1),
                timedelta(hours=0, minutes=1, seconds=47) / timedelta(hours=1),
                timedelta(hours=0, minutes=2, seconds=47) / timedelta(hours=1),
            ],
        )


class TestParsivel9(Check):
    date = "2024-01-14"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/hyytiala.txt"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.serial_number == "291923"
        assert np.allclose(
            self.nc["time"][:],
            [
                timedelta(hours=0, minutes=0, seconds=0) / timedelta(hours=1),
                timedelta(hours=0, minutes=1, seconds=0) / timedelta(hours=1),
                timedelta(hours=0, minutes=2, seconds=0) / timedelta(hours=1),
            ],
        )


class TestParsivel10(Check):
    date = "2014-01-04"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/hyytiala2.txt"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.serial_number == "295160"
        assert np.allclose(
            self.nc["time"][:],
            [
                timedelta(hours=10, minutes=1, seconds=0) / timedelta(hours=1),
                timedelta(hours=10, minutes=2, seconds=0) / timedelta(hours=1),
            ],
        )
        assert_array_equal(
            self.nc["data_raw"][:].mask, [[[1] * 32] * 32, [[0] * 32] * 32]
        )


class TestParsivel11(Check):
    date = "2024-02-19"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/bucharest-20240219.cvs"
    site_meta = SITE_META
    uuid = disdrometer.parsivel2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 4


class TestThies(Check):
    date = "2021-09-15"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/thies-lnm/2021091507.txt"
    site_meta = SITE_META
    uuid = disdrometer.thies2nc(filename, temp_path, site_meta, date=date)

    def test_processing(self):
        assert self.nc.title == f'LNM disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2021"
        assert self.nc.month == "09"
        assert self.nc.day == "15"
        assert self.nc.location == "Kumpula"
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.serial_number == "1025"


class TestThies2(Check):
    date = "2024-04-14"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/thies-lnm/THIES_LMP_2024_04_14_00_00.dat"
    site_meta = SITE_META
    uuid = disdrometer.thies2nc(filename, temp_path, site_meta, date=date)

    def test_processing(self):
        assert self.nc.title == f'LNM disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2024"
        assert self.nc.month == "04"
        assert self.nc.day == "14"
        assert self.nc.location == "Kumpula"
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.serial_number == "2965"
        assert np.allclose(
            self.nc["time"][:],
            [
                timedelta(hours=0, minutes=1, seconds=0) / timedelta(hours=1),
                timedelta(hours=0, minutes=2, seconds=0) / timedelta(hours=1),
                timedelta(hours=0, minutes=3, seconds=0) / timedelta(hours=1),
            ],
        )


class TestThies3(Check):
    date = "2024-04-17"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/thies-lnm/2024041723-leipzig-lim.txt"
    site_meta = SITE_META
    uuid = disdrometer.thies2nc(
        filename, temp_path, {**site_meta, "truncate_columns": 23}, date=date
    )

    def test_processing(self):
        assert self.nc.title == f'LNM disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2024"
        assert self.nc.month == "04"
        assert self.nc.day == "17"
        assert self.nc.location == "Kumpula"
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.serial_number == "0747"
        assert np.allclose(
            self.nc["time"][:],
            [
                timedelta(hours=23, minutes=0, seconds=50) / timedelta(hours=1),
                timedelta(hours=23, minutes=1, seconds=50) / timedelta(hours=1),
                timedelta(hours=23, minutes=2, seconds=50) / timedelta(hours=1),
            ],
        )


class TestThies4(Check):
    date = "2024-07-02"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/thies-lnm/PAY_DDTH01_20240702_actris.csv"
    site_meta = SITE_META
    uuid = disdrometer.thies2nc(filename, temp_path, site_meta, date=date)

    def test_processing(self):
        assert self.nc.title == f'LNM disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2024"
        assert self.nc.month == "07"
        assert self.nc.day == "02"
        assert self.nc.location == "Kumpula"
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.serial_number == "3486"
        assert np.allclose(
            self.nc["time"][:],
            [
                timedelta(hours=0, minutes=0, seconds=0) / timedelta(hours=1),
                timedelta(hours=0, minutes=1, seconds=0) / timedelta(hours=1),
                timedelta(hours=0, minutes=2, seconds=0) / timedelta(hours=1),
            ],
        )


class TestThies5(Check):
    date = "2024-10-03"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/thies-lnm/20241003_lnm_lindenberg.csv"
    site_meta = SITE_META
    uuid = disdrometer.thies2nc(filename, temp_path, site_meta, date=date)

    def test_processing(self):
        assert self.nc.title == f'LNM disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2024"
        assert self.nc.month == "10"
        assert self.nc.day == "03"
        assert self.nc.location == "Kumpula"
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.serial_number == "1025"
        assert np.allclose(
            self.nc["time"][:],
            [
                timedelta(hours=0, minutes=0, seconds=41) / timedelta(hours=1),
                timedelta(hours=0, minutes=1, seconds=41) / timedelta(hours=1),
                timedelta(hours=0, minutes=2, seconds=41) / timedelta(hours=1),
            ],
        )


class TestInvalidCharacters(Check):
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/parsivel_bad.log"
    site_meta = SITE_META
    date = "2019-04-10"
    uuid = disdrometer.parsivel2nc(
        filename,
        temp_path,
        site_meta,
        date=date,
        telegram=TELEGRAM,
    )

    def test_skips_invalid_row(self):
        assert len(self.nc.variables["time"]) == 4
