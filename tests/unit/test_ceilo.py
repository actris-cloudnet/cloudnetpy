"""This module contains unit tests for ceilo-module."""
from os import path

import netCDF4
import pytest
from numpy.testing import assert_almost_equal

from cloudnetpy.instruments import ceilo

SCRIPT_PATH = path.dirname(path.realpath(__file__))


def test_find_ceilo_model_jenoptik():
    file = f"{SCRIPT_PATH}/data/chm15k/00100_A202010220005_CHM170137.nc"
    assert ceilo._find_ceilo_model(file) == "chm15k"


def test_find_ceilo_model_cl61d():
    file = f"{SCRIPT_PATH}/data/cl61d/live_20210829_224520.nc"
    assert ceilo._find_ceilo_model(file) == "cl61d"


@pytest.mark.parametrize(
    "fix, result",
    [
        ("CL01", "cl31_or_cl51"),
        ("CL02", "cl31_or_cl51"),
        ("CT02", "ct25k"),
    ],
)
def test_find_ceilo_model_vaisala(fix, result, tmpdir):
    file_name = "/".join((str(tmpdir), "ceilo.txt"))
    f = open(file_name, "w")
    f.write("row\n")
    f.write("\n")
    f.write("-2020-04-10 00:00:58\n")
    f.write(f"\x01{fix}\n")
    f.close()
    assert ceilo._find_ceilo_model(str(file_name)) == result


def test_cl51_reading(tmp_path):
    output_file = tmp_path / "cl51.nc"
    file = f"{SCRIPT_PATH}/data/vaisala/cl51.DAT"
    ceilo.ceilo2nc(file, output_file, {"name": "Norunda", "altitude": 100})
    with netCDF4.Dataset(output_file) as nc:
        assert nc.source == "Vaisala CL51"
        assert nc.cloudnet_file_type == "lidar"
        assert nc.location == "Norunda"
        assert nc.year == "2020"
        assert nc.month == "11"
        assert nc.day == "15"
        assert nc.variables["time"].shape == (2,)
        assert nc.variables["zenith_angle"][:].all() < 5
        assert_almost_equal(nc.variables["altitude"][:], 100)


def test_cl31_reading(tmp_path):
    output_file = tmp_path / "cl31.nc"
    file = f"{SCRIPT_PATH}/data/vaisala/cl31.DAT"
    ceilo.ceilo2nc(file, output_file, {"name": "Norunda", "altitude": 100})
    with netCDF4.Dataset(output_file) as nc:
        assert nc.source == "Vaisala CL31"
        assert_almost_equal(nc.variables["wavelength"][:], 910)
