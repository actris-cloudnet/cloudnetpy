import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import falling


class Obs:
    def __init__(self):

        self.z = ma.array([[.5, .5, .5, .5, .5, .5],
                           [.5, .5, .5, .5, .5, .5]],
                          mask=[[0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0]])

        self.is_clutter = np.array([[1, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0, 0]])

        self.beta = ma.array([[1e-8, 1e-8, 1e-5, 1e-8, 1e-8, 1e-3],
                              [1e-8, 1e-8, 1e-5, 1e-5, 1e-8, 1e-8]],
                             mask=[[0, 0, 0, 1, 0, 0],
                                   [0, 1, 1, 0, 0, 0]])

        self.tw = np.array([[250, 250, 250, 250, 250, 250],
                           [250, 250, 250, 250, 250, 270]])


def test_find_falling_from_radar():
    obs = Obs()
    is_insects = np.array([[0, 1, 0, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0]])
    result = np.array([[0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1]])
    assert_array_equal(falling._find_falling_from_radar(obs, is_insects), result)


def test_find_falling_from_lidar():
    obs = Obs()
    result = np.array([[0, 0, 1, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0]])
    is_liquid = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0]])
    assert_array_equal(falling._find_falling_from_lidar(obs, is_liquid), result)


def test_find_cold_aerosols():
    obs = Obs()
    result = np.array([[0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 0]])
    is_liquid = np.array([[0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 1, 0, 0]])
    assert_array_equal(falling._find_cold_aerosols(obs, is_liquid), result)

