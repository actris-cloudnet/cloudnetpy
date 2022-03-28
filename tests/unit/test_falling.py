import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import falling
import pytest


class Obs:
    def __init__(self):

        self.z = ma.array(
            [[0.4, 0.5, 0.6, 0.7, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
            mask=[[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]],
        )

        self.is_clutter = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])

        self.beta = ma.array(
            [[1e-8, 1e-8, 1e-5, 1e-8, 1e-8, 1e-3], [1e-8, 1e-8, 1e-5, 1e-5, 1e-8, 1e-8]],
            mask=[[0, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0]],
        )

        self.tw = ma.array([[250, 250, 250, 250, 250, 250], [250, 250, 250, 250, 250, 270]])


def test_find_falling_from_radar():
    obs = Obs()
    is_insects = np.array([[0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0]])
    result = np.array([[0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 1, 1]])
    assert_array_equal(falling._find_falling_from_radar(obs, is_insects), result)


def test_find_cold_aerosols():
    obs = Obs()
    result = np.array([[1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 0]])
    is_liquid = np.array([[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0]])
    assert_array_equal(falling._find_cold_aerosols(obs, is_liquid), result)


@pytest.mark.parametrize(
    "z, ind_top, result",
    [
        (ma.masked_array([1, 1, 1, 1], mask=[0, 0, 0, 0]), 2, False),
        (ma.masked_array([1, 1, 1, 1], mask=[0, 0, 0, 0]), 2, False),
        (ma.masked_array([1, 1, 1, 1], mask=[0, 0, 0, 0]), 3, False),
        (ma.masked_array([1, 1, 1, 1], mask=[1, 1, 1, 1]), 3, False),
        (ma.masked_array([1, 1, 1, 1], mask=[0, 0, 0, 1]), 2, True),
    ],
)
def test_is_z_missing_above_liquid(z, ind_top, result):
    assert falling._is_z_missing_above_liquid(z, ind_top) == result


@pytest.mark.parametrize(
    "z, ind_base, ind_top, result",
    [
        (ma.masked_array([1, 1, 1, 1], mask=[0, 0, 0, 0]), 1, 2, False),
        (ma.masked_array([1, 1, 2, 1], mask=[0, 0, 0, 0]), 1, 2, True),
        (ma.masked_array([1, 2, 1, 1], mask=[0, 0, 0, 0]), 1, 2, False),
        (ma.masked_array([1, 1, 2, 3], mask=[0, 0, 1, 0]), 1, 3, True),
        (ma.masked_array([1, 2, 3, 4], mask=[0, 1, 1, 1]), 1, 3, False),
    ],
)
def test_is_z_increasing(z, ind_top, ind_base, result):
    assert falling._is_z_increasing(z, ind_base, ind_top) == result


def test_fix_liquid_dominated_radar():
    obs = Obs()

    is_liquid = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]])

    falling_from_radar = np.array([[0, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]])

    result = np.array([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]])

    fixed = falling._fix_liquid_dominated_radar(obs, falling_from_radar, is_liquid)

    assert_array_equal(fixed, result)
