""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import ceilo
import pytest


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
