""" This module contains unit tests for classify-module. """
import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import assert_array_equal

from cloudnetpy.categorize import classify, containers


class Obs:
    def __init__(self):
        self.beta = ma.array(
            [[1, 1], [1, 1], [1, 1], [1, 1]], mask=[[0, 0], [0, 0], [1, 1], [1, 1]]
        )


def test_find_aerosols():
    obs = Obs()
    is_falling = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
    is_liquid = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    result = np.array([[0, 1], [0, 0], [0, 0], [0, 0]])
    assert_array_equal(classify._find_aerosols(obs, is_falling, is_liquid), result)  # type: ignore


def test_bits_to_integer():
    b0 = [[1, 0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    b1 = [[0, 1, 0, 0, 1, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    b2 = [[0, 0, 1, 0, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
    b3 = [[0, 0, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
    bits = [b0, b1, b2, b3]
    re = [[1, 2, 4, 8, 3, 7, 15, 5, 10, 9], [6, 12, 14, 0, 0, 0, 0, 0, 0, 0]]
    assert_array_equal(classify._bits_to_integer(bits), re)


class TestFindRain:
    time = np.linspace(0, 24, 2880)  # 30 s resolution

    @pytest.fixture(autouse=True)
    def run_before_tests(self):
        self.z = np.zeros((len(self.time), 10))
        yield

    def test_low_values(self):
        result = np.zeros(len(self.time))
        assert_array_equal(containers._find_rain_from_radar_echo(self.z, self.time), result)

    def test_threshold_value(self):
        self.z[:, 3] = 0.1
        result = np.ones(len(self.time))
        assert_array_equal(containers._find_rain_from_radar_echo(self.z, self.time), result)

    def test_hot_pixel_removal(self):
        self.z[5, 3] = 0.1
        result = np.zeros(len(self.time))
        assert_array_equal(
            containers._find_rain_from_radar_echo(self.z, self.time, time_buffer=1), result
        )

    def test_rain_spreading(self):
        self.z[10:12, 3] = 0.1
        result = np.zeros(len(self.time))
        result[8:14] = 1
        assert_array_equal(
            containers._find_rain_from_radar_echo(self.z, self.time, time_buffer=1), result
        )


def test_find_clutter():
    is_rain = np.array([0, 0, 0, 1, 1], dtype=bool)
    v = np.ones((5, 12)) * 0.1
    vm = ma.array(v)
    vm[:, 5] = 0.04
    result = np.zeros(vm.shape)
    result[:3, 5] = 1
    assert_array_equal(containers._find_clutter(vm, is_rain), result)


def test_find_drizzle_and_falling():
    is_liquid = np.array([[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=bool)

    is_falling = np.array([[0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1]], dtype=bool)

    is_freezing = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]], dtype=bool)

    expected = ma.array(
        [[0, 2, 0, 1, 1, 0], [0, 0, 0, 2, 1, 1]], mask=[[1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0]]
    )

    result = classify._find_drizzle_and_falling(is_liquid, is_falling, is_freezing)
    assert_array_equal(expected.data, result.data)
    assert isinstance(result, ma.MaskedArray)
    assert_array_equal(expected.mask, result.mask)


def test_fix_undetected_melting_layer():
    is_liquid = np.array(
        [[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool
    )

    is_falling = np.array(
        [[0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=bool
    )

    is_freezing = np.array(
        [[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=bool
    )

    is_melting = np.array(
        [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool
    )

    expected = np.array(
        [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool
    )

    bits = [is_liquid, is_falling, is_freezing, is_melting]
    result = classify._fix_undetected_melting_layer(bits)
    assert_array_equal(expected.data, result.data)
