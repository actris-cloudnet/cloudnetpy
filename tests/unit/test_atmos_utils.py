"""This module contains unit tests for atmos-module."""

import atmoslib
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cloudnetpy.categorize import atmos_utils


def test_calc_adiabatic_lwc():
    lwc_dz = np.array([[0, 0, 2.1, 2.1, 0, 3.2, 3.2], [0, 2.0, 2.0, 0, 1.5, 1.5, 0]])

    height = np.array([10, 12, 14, 16, 18, 20, 22])

    adiabatic_lwc = atmos_utils.calc_adiabatic_lwc(lwc_dz, height)

    expected = np.array([[0, 0, 4.2, 8.4, 0, 6.4, 12.8], [0, 4, 8, 0, 3, 6, 0]])

    assert_array_almost_equal(adiabatic_lwc, expected, decimal=1)
