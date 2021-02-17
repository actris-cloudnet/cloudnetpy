""" This module contains unit tests for classify-module. """
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import classify
from cloudnetpy.categorize import containers


class Obs:
    def __init__(self):
        self.beta = ma.array([[1, 1], [1, 1], [1, 1], [1, 1]],
                             mask=[[0, 0], [0, 0], [1, 1], [1, 1]])


def test_find_aerosols():
    obs = Obs()
    is_falling = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
    is_liquid = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    result = np.array([[0, 1], [0, 0], [0, 0], [0, 0]])
    assert_array_equal(classify._find_aerosols(obs, is_falling, is_liquid), result)


def test_bits_to_integer():
    b0 = [[1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    b1 = [[0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
          [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    b2 = [[0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
    b3 = [[0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
          [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
    bits = [b0, b1, b2, b3]
    re = [[1, 2, 4, 8, 3, 7, 15, 5, 10, 9],
          [6, 12, 14, 0, 0, 0, 0, 0, 0, 0]]
    assert_array_equal(classify._bits_to_integer(bits), re)


class TestFindRain:
    time = np.linspace(0, 24, 2880)  # 30 s resolution
    z = np.zeros((len(time), 10))

    def test_1(self):
        result = np.zeros(len(self.time))
        assert_array_equal(containers._find_rain(self.z, self.time), result)

    def test_2(self):
        self.z[:, 3] = 0.1
        result = np.ones(len(self.time))
        assert_array_equal(containers._find_rain(self.z, self.time), result)

    def test_3(self):
        self.z[5, 3] = 0.1
        result = np.ones(len(self.time))
        result[3:7] = 1
        assert_array_equal(containers._find_rain(self.z, self.time, time_buffer=1), result)

    def test_4(self):
        self.z[1440, 3] = 0.1
        result = np.ones(len(self.time))
        assert_array_equal(containers._find_rain(self.z, self.time, time_buffer=1500), result)


def test_find_clutter():
    is_rain = np.array([0, 0, 0, 1, 1], dtype=bool)
    v = np.ones((5, 12)) * 0.1
    v = ma.array(v)
    v[:, 5] = 0.04
    result = np.zeros(v.shape)
    result[:3, 5] = 1
    assert_array_equal(containers._find_clutter(v, is_rain), result)


def test_find_drizzle_and_falling():
    is_liquid = np.array([[0, 0, 1, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0]])

    is_falling = np.array([[0, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 1]])

    is_freezing = np.array([[0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1]])

    result = ma.array([[0, 2, 0, 0, 1, 0],
                       [0, 0, 0, 2, 1, 1]],
                      mask=[[1, 0, 1, 1, 0, 1],
                            [1, 1, 1, 0, 0, 0]])

    assert_array_equal(classify._find_drizzle_and_falling(is_liquid, is_falling,
                                                          is_freezing), result)


def test_find_profiles_with_undetected_melting():
    is_liquid = np.array([[0, 0, 1, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])

    is_falling = np.array([[0, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1]])

    is_freezing = np.array([[0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0]])

    is_melting = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

    result = np.array([0, 1, 1, 0])
    bits = [is_liquid, is_falling, is_freezing, is_melting]
    undetected = classify._find_profiles_with_undetected_melting(bits)
    assert_array_equal(undetected.data, result)

