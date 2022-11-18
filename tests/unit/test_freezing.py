import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import assert_array_equal

from cloudnetpy.categorize import freezing
from cloudnetpy.constants import T0


def test_find_t0_alt():
    temperature = np.array(
        [[290, 280, T0, 260], [320, T0 + 10, T0 - 10, 220], [240, 230, 220, 210]]
    )
    height = np.array([10, 20, 30, 40])
    res = [30, 25, 10]
    cnet = freezing._find_t0_alt(temperature, height)
    assert_array_equal(cnet, res)


@pytest.mark.parametrize(
    "mean_melting_alt, t0_alt, height, expected_result",
    [
        (ma.array([1, 2, 3], mask=[1, 1, 1]), np.array([1, 1]), np.array([1, 2]), True),
        (ma.array([1, 2, 3], mask=[1, 0, 1]), np.array([1, 1]), np.array([1, 2]), False),
        (ma.array([1, 2, 3], mask=[1, 1, 1]), np.array([1.2, 1]), np.array([1, 2]), False),
        (ma.array([1, 2, 3], mask=[0, 1, 1]), np.array([1.2, 1]), np.array([1, 2]), False),
    ],
)
def test_is_all_freezing(mean_melting_alt, t0_alt, height, expected_result):
    result = freezing._is_all_freezing(mean_melting_alt, t0_alt, height)
    assert result == expected_result


class Obs:
    def __init__(self):

        self.time = np.linspace(0, 24, 13)  # 2h resolution
        self.height = np.linspace(0, 5, 6) * 100

        self.tw = ma.array(
            [
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 270, 260, 250, 240],
                [290, 280, 280, 260, 250, 240],
            ]
        )


def test_find_mean_melting_alt():
    obs = Obs()
    obs.time = np.array([0.0, 1.0])
    is_melting = np.array([[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]], dtype=bool)
    result = np.array([250, 300])
    assert_array_equal(freezing._find_mean_melting_alt(obs, is_melting), result)  # type: ignore


def test_find_freezing_region():
    obs = Obs()
    is_melting = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    expected = np.array(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )

    result = freezing.find_freezing_region(obs, is_melting)  # type: ignore
    assert_array_equal(result, expected)
