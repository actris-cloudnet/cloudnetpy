""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import ceilo
import pytest
import numpy as np
from numpy.testing import assert_equal


@pytest.mark.parametrize("input, result", [
    ('A line', False),
    ('', False),
    ('\n', True),
    ('\r\n', True),
])
def test_is_empty_line(input, result):
    assert ceilo.is_empty_line(input) == result


@pytest.mark.parametrize("input, result", [
    ('01:30:00', 1.5),
    ('02:00:00', 2),
    ('13:15:00', 13.25),
])
def test_time_to_fraction_hour(input, result):
    assert ceilo.time_to_fraction_hour(input) == result


@pytest.mark.parametrize("input, result", [
    ('-2019-02-13 23:04:50', True),
    ('2019-02-13 23:04:50', False),
    ('2019-02-13', False),
])
def test_is_timestamp(input, result):
    assert ceilo.is_timestamp(input) == result


@pytest.mark.parametrize("keys, values, result", [
    (('a', 'b'), [[1, 2], [1, 2], [1, 2]],
     {'a': np.array([1, 1, 1]), 'b': np.array([2, 2, 2])}),
])
def test_values_to_dict(keys, values, result):
    assert_equal(ceilo.values_to_dict(keys, values), result)


@pytest.mark.parametrize("string, indices, result", [
    ('abcd', [3, 4], ['d']),
    ('abcd', [0, 4], ['abcd']),
    ('abcdedfg', [1, 2, 4, 5], ['b', 'cd', 'e']),
])
def test_split_string(string, indices, result):
    assert_equal(ceilo.split_string(string, indices), result)
