""" This module contains unit tests for ceilo-module. """
import numpy as np
import numpy.ma as ma
from cloudnetpy.instruments import ceilo
from numpy.testing import assert_array_almost_equal
import pytest


def test_is_empty_line():
    assert ceilo.is_empty_line('A line') is False
    assert ceilo.is_empty_line('') is False
    assert ceilo.is_empty_line('\n') is True
    assert ceilo.is_empty_line('\r\n') is True


def test_time_to_fraction_hour():
    assert ceilo.time_to_fraction_hour('01:30:00') == 1.5
    assert ceilo.time_to_fraction_hour('02:00:00') == 2
    assert ceilo.time_to_fraction_hour('13:15:00') == 13.25


def test_is_timestamp():
    assert ceilo.is_timestamp('-2019-02-13 23:04:50') is True
    assert ceilo.is_timestamp('2019-02-13 23:04:50') is False
    assert ceilo.is_timestamp('-2019-02-13') is False

