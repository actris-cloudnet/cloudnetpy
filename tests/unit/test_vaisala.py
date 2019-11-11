""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import vaisala
import pytest
import numpy as np
from numpy.testing import assert_equal


@pytest.mark.parametrize("input, result", [
    ('01:30:00', 1.5),
    ('02:00:00', 2),
    ('13:15:00', 13.25),
])
def test_time_to_fraction_hour(input, result):
    assert vaisala.time_to_fraction_hour(input) == result


@pytest.mark.parametrize("keys, values, result", [
    (('a', 'b'), [[1, 2], [1, 2], [1, 2]],
     {'a': np.array([1, 1, 1]), 'b': np.array([2, 2, 2])}),
])
def test_values_to_dict(keys, values, result):
    assert_equal(vaisala.values_to_dict(keys, values), result)


@pytest.mark.parametrize("string, indices, result", [
    ('abcd', [3, 4], ['d']),
    ('abcd', [0, 4], ['abcd']),
    ('abcdedfg', [1, 2, 4, 5], ['b', 'cd', 'e']),
])
def test_split_string(string, indices, result):
    assert_equal(vaisala.split_string(string, indices), result)
