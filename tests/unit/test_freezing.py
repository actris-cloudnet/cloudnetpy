import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import freezing
from cloudnetpy.constants import T0


def test_find_t0_alt():
    temperature = np.array([[290, 280, T0, 260],
                           [320, T0+10, T0-10, 220],
                           [240, 230, 220, 210]])
    height = np.array([10, 20, 30, 40])
    res = [30, 25, 10]
    cnet = freezing._find_t0_alt(temperature, height)
    assert_array_equal(cnet, res)


class Obs:
    def __init__(self):

        self.time = np.linspace(0, 24, 13)  # 2h resolution
        self.height = np.linspace(0, 5, 6)

        self.tw = ma.array([[290, 280, 270, 260, 250, 240],
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
                            [290, 280, 280, 260, 250, 240]])


def test_find_mean_melting_alt():
    obs = Obs()
    obs.time = np.arange(2)
    is_melting = np.array([[0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0]], dtype=bool)
    result = np.array([2.5, 3])
    assert_array_equal(freezing._find_mean_melting_alt(obs, is_melting), result)


def test_find_freezing_region():
    obs = Obs()
    is_melting = np.array([[0, 0, 1, 0, 0, 0],
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
                           [0, 0, 0, 0, 0, 0]], dtype=bool)

    result = np.array([[0, 0, 0, 1, 1, 1],
                       [0, 0, 0, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 1],
                       [0, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1],
                       [0, 0, 1, 1, 1, 1],
                       [0, 0, 1, 1, 1, 1],
                       [0, 0, 1, 1, 1, 1],
                       [0, 0, 1, 1, 1, 1],
                       [0, 0, 1, 1, 1, 1],
                       [0, 0, 1, 1, 1, 1],
                       [0, 0, 0, 1, 1, 1]], dtype=bool)

    assert_array_equal(freezing.find_freezing_region(obs, is_melting), result)
