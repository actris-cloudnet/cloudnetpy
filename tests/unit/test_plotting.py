import os

import numpy as np
import pytest
from cloudnetpy.plotting import plotting
from cloudnetpy.instruments import basta2nc
from os import path
import netCDF4

SCRIPT_PATH = path.dirname(path.realpath(__file__))

@pytest.mark.parametrize(
    "numbers, result",
    [
        ((1e-6, 1e-5), [-6, -5]),
        ((1e-1,), [-1]),
    ],
)
def test_lin2log(numbers, result):
    assert plotting.lin2log(*numbers) == result


@pytest.mark.parametrize(
    "vmin, vmax, result",
    [
        (-7, -5, ["10$^{-7}$", "10$^{-6}$", "10$^{-5}$"]),
    ],
)
def test_generate_log_cbar_ticklabel_list(vmin, vmax, result):
    assert plotting.get_log_cbar_tick_labels(vmin, vmax) == result

@pytest.fixture(scope="session")
def basta_nc(tmpdir_factory) -> str:
    basta_raw = f"{SCRIPT_PATH}/data/basta/basta_1a_cldradLz1R025m_v03_20210827_000000.nc"
    site_meta = {
        "name": "Palaiseau",
        "latitude": 50,
        "longitude": 104.5,
        "altitude": 50,
    }
    filename = tmpdir_factory.mktemp("data").join("file.nc")
    basta2nc(basta_raw, filename, site_meta)
    return filename


def test_figure_data(basta_nc):
    options = plotting.PlotParameters()
    with netCDF4.Dataset(basta_nc) as nc:
        figure_data = plotting.FigureData(nc, ["Zh", "v", "kissa"], options)
        assert len(figure_data) == 2
        assert figure_data.height is not None
        assert np.max(figure_data.height) < 25


def test_generate_figure(basta_nc):
    plotting.generate_figure(basta_nc, ["Zh"], show=False)
    image_name = "test_23142134.png"
    plotting.generate_figure(basta_nc, ["Zh"], show=False, output_filename=image_name)
    assert path.exists(image_name)
    os.remove(image_name)
