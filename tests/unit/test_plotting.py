import os

import numpy as np
import numpy.ma as ma
import pytest
from cloudnetpy.plotting import plotting
from cloudnetpy.instruments import basta2nc
from cloudnetpy.exceptions import PlottingError
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
    basta_raw = (
        f"{SCRIPT_PATH}/data/basta/basta_1a_cldradLz1R025m_v03_20210827_000000.nc"
    )
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


def test_screen_completely_masked_profiles_with_no_mask():
    time = np.array([1, 2, 3])
    data = ma.array([4, 5, 6])
    result_time, result_data = plotting.screen_completely_masked_profiles(time, data)
    assert np.array_equal(result_time, time)
    assert np.array_equal(result_data, data)


def test_screen_completely_masked_profiles_with_nothing_masked():
    time = np.array([1, 2, 3])
    data = ma.array([[4, 5], [4, 5], [4, 5]], mask=False)
    result_time, result_data = plotting.screen_completely_masked_profiles(time, data)
    assert np.array_equal(result_time, time)
    assert np.array_equal(result_data, data)


def test_screen_completely_masked_profiles_with_nothing_masked_2():
    time = np.array([1, 2, 3])
    data = ma.array(
        [[4, 5], [4, 5], [4, 5]], mask=[[False, False], [False, False], [False, False]]
    )
    result_time, result_data = plotting.screen_completely_masked_profiles(time, data)
    assert np.array_equal(result_time, time)
    assert np.array_equal(result_data, data)


def test_screen_completely_masked_profiles_with_partial_mask():
    time = np.array([1, 2, 3, 4, 5, 6])
    data = ma.array(
        [[4, 5], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]],
        mask=[
            [False, True],
            [False, False],
            [False, False],
            [False, False],
            [True, True],
            [True, True],
        ],
    )
    result_time, result_data = plotting.screen_completely_masked_profiles(time, data)
    assert np.array_equal(result_time, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(
        result_data, ma.array([[4, 5], [6, 7], [7, 8], [8, 9], [9, 10]])
    )


def test_screen_completely_masked_profiles_with_all_masked():
    time = np.array([1, 2, 3])
    data = ma.array([[4, 5], [4, 5], [4, 5]], mask=True)
    with pytest.raises(PlottingError, match="All values masked in the file."):
        plotting.screen_completely_masked_profiles(time, data)


def test_screen_completely_masked_profiles_with_all_masked_2():
    time = np.array([1, 2, 3])
    data = ma.array(
        [[4, 5], [4, 5], [4, 5]], mask=[[True, True], [True, True], [True, True]]
    )
    with pytest.raises(PlottingError, match="All values masked in the file."):
        plotting.screen_completely_masked_profiles(time, data)


@pytest.mark.parametrize(
    "units, result",
    [
        ("1", ""),
        ("mu m", "$\\mu$m"),
        ("m-3", "m$^{-3}$"),
        ("m s-1", "m s$^{-1}$"),
        ("sr-1 m-1", "sr$^{-1}$ m$^{-1}$"),
        ("kg m-2", "kg m$^{-2}$"),
        ("kg m-3", "kg m$^{-3}$"),
        ("g m-3", "g m$^{-3}$"),
        ("g m-2", "g m$^{-2}$"),
        ("kg m-2 s-1", "kg m$^{-2}$ s$^{-1}$"),
        ("dB km-1", "dB km$^{-1}$"),
        ("rad km-1", "rad km$^{-1}$"),
    ],
)
def test_reformat_units(units, result):
    assert plotting._reformat_units(units) == result
