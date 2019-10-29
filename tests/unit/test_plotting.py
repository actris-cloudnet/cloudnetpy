import pytest
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


class TestReadAxValues:

    def test_1(self, nc_file, test_array):
        assert_array_equal(plotting._read_ax_values(nc_file)[0], test_array)

    def test_2(self, nc_file, test_array):
        assert_array_equal(plotting._read_ax_values(nc_file)[1], test_array/1000)

    def test_3(self, nc_file, test_array):
        assert_array_equal(plotting._read_ax_values(nc_file, 'model')[0], test_array)

    def test_4(self, nc_file, test_array):
        assert_array_equal(plotting._read_ax_values(nc_file, 'model')[1], test_array/1000)
