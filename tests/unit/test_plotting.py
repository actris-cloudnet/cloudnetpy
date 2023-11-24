from datetime import date

import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.signal import filtfilt

from cloudnetpy.plotting import plotting


@pytest.mark.parametrize(
    "numbers, result",
    [
        ((1e-6, 1e-5), [-6, -5]),
        ((1e-1,), [-1]),
    ],
)
def test_lin2log(numbers, result):
    assert plotting.lin2log(*numbers) == result


# @pytest.mark.parametrize(
#     "reso, result",
#     [
#         (4, ["", "04:00", "08:00", "12:00", "16:00", "20:00", ""]),
#         (
#             2,
#             [
#                 "",
#                 "02:00",
#                 "04:00",
#                 "06:00",
#                 "08:00",
#                 "10:00",
#                 "12:00",
#                 "14:00",
#                 "16:00",
#                 "18:00",
#                 "20:00",
#                 "22:00",
#                 "",
#             ],
#         ),
#     ],
# )
# def test_get_standard_time_ticks(reso, result):
#     assert plotting._get_standard_time_ticks(reso) == result


@pytest.mark.parametrize(
    "vmin, vmax, result",
    [
        (-7, -5, ["10$^{-7}$", "10$^{-6}$", "10$^{-5}$"]),
    ],
)
def test_generate_log_cbar_ticklabel_list(vmin, vmax, result):
    assert plotting.get_log_cbar_tick_labels(vmin, vmax) == result


# def test_find_time_gap_indices():
#     time = np.array([0.01, 0.02, 0.04, 0.13, 0.14, 0.23, 0.24])
#     indices = (2, 4)
#     gaps = plotting._find_time_gap_indices(time, 5)
#     assert_array_almost_equal(gaps, indices)
#
#
# def test_mark_gaps():
#     time = np.array([1.0, 2, 3, 10, 11, 20, 21, 22])
#     data = ma.ones((len(time), 3))
#     data.mask = np.zeros(data.shape)
#     time_new, data_new = plotting._mark_gaps(time, data, 60)
#     expected_mask = np.array(
#         [
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [1, 1, 1],
#             [0, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [1, 1, 1],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [1, 1, 1],
#         ],
#     )
#     expected_data = np.array(
#         [
#             [1, 1, 1],
#             [1, 1, 1],
#             [1, 1, 1],
#             [0, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [1, 1, 1],
#             [0, 0, 0],
#             [0, 0, 0],
#             [1, 1, 1],
#             [1, 1, 1],
#             [1, 1, 1],
#             [0, 0, 0],
#             [0, 0, 0],
#         ],
#     )
#     expected_time = np.array(
#         [
#             1,
#             2,
#             3,
#             3.001,
#             9.999,
#             10,
#             11,
#             11.001,
#             19.999,
#             20,
#             21,
#             22,
#             22.001,
#             23.999,
#         ],
#     )
#     assert_array_equal(data_new.data, expected_data)
#     assert_array_equal(data_new.mask, expected_mask)
#     assert_array_almost_equal(expected_time, time_new)
