""" This module contains unit tests for classify-module. """
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import classify


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

