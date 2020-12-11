from scipy.signal import lfilter
import numpy as np
import pytest
import numpy.testing as testing
from numpy.testing import assert_array_equal
from cloudnetpy.plotting import plotting
from datetime import date


@pytest.mark.parametrize("numbers, result", [
    ((1e-6, 1e-5), [-6, -5]),
    ((1e-1,), [-1]),
])
def test_lin2log(numbers, result):
    assert plotting._lin2log(*numbers) == result


@pytest.mark.parametrize("reso, result", [
    (4, ['', '04:00', '08:00', '12:00', '16:00', '20:00', '']),
    (2, ['', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00',
         '14:00', '16:00', '18:00', '20:00', '22:00', '']),
    ])
def test_get_standard_time_ticks(reso, result):
    assert plotting._get_standard_time_ticks(reso) == result


@pytest.mark.parametrize("vmin, vmax, result", [
    (-7, -5, ['10$^{-7}$', '10$^{-6}$', '10$^{-5}$']),
    ])
def test__generate_log_cbar_ticklabel_list(vmin, vmax, result):
    assert plotting._generate_log_cbar_ticklabel_list(vmin, vmax) == result


def test_get_subtitle_text():
    case_date = date(2019, 5, 7)
    site_name = 'Mace-Head'
    assert plotting._get_subtitle_text(case_date, site_name) == 'Mace Head, 7 May 2019'


def test_read_location(nc_file):
    assert plotting._read_location(nc_file) == 'Kumpula'


def test_read_data(nc_file):
    assert plotting._read_date(nc_file) == date(2019, 5, 23)


def test_create_save_name(file_metadata):
    path = '/foo/bar/'
    case_date = file_metadata['case_date']
    datestr = file_metadata['year'] + file_metadata['month'] + file_metadata['day']
    fields = ['ldr', 'z']
    assert plotting._create_save_name(path, case_date, fields) == f"/foo/bar/{datestr}_ldr_z.png"


def test_remove_timestamps_of_next_date():
    times = [1, 2, 3, 4, 5, 6, 1, 2, 3]
    data = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    x, y = plotting._remove_timestamps_of_next_date(times, data, 8)
    assert x == times[:-3]
    assert y == data[:-3]


def test_calculate_rolling_mean():
    time = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    r_window = np.blackman(4)
    r_mean = np.convolve(data, r_window, 'valid') / np.sum(r_window)
    x, y = plotting._calculate_rolling_mean(time, data)
    testing.assert_array_almost_equal(x, r_mean)


def test_filter_noise():
    data = np.array([1, 1, 5, -5, 1, 1, 5, -5, 1, 1, -5, 5])
    x = plotting._filter_noise(data, 3)
    b = [1.0 / 3] * 3
    data = lfilter(b, 1, data)
    testing.assert_array_almost_equal(x, data)


class TestReadAxValues:

    def test_1(self, nc_file, test_array):
        assert_array_equal(plotting._read_ax_values(nc_file)[0], test_array)

    def test_2(self, nc_file, test_array):
        assert_array_equal(plotting._read_ax_values(nc_file)[1], test_array/1000)
