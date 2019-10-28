import pytest
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
