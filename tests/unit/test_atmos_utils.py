"""This module contains unit tests for atmos-module."""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cloudnetpy.categorize import atmos_utils


@pytest.mark.parametrize(
    "t, res",
    [
        (300, 3533),
        (280, 991),
    ],
)
def test_saturation_vapor_pressure(t, res):
    """Unit tests for atmos.saturation_vapor_pressure()."""
    cnet = atmos_utils.calc_saturation_vapor_pressure(np.array(t))
    assert_array_almost_equal(cnet, res, decimal=0)


@pytest.mark.parametrize(
    "t, p, q, res",
    [
        (280, 101330, 0.001, 273.21),
        (250, 90000, 0.001, 251.03),
    ],
)
def test_wet_bulb(t, p, q, res):
    """Unit tests for atmos.wet_bulb()."""
    model = {
        "temperature": np.array(t),
        "pressure": np.array(p),
        "q": np.array(q),
    }
    cnet = atmos_utils.calc_wet_bulb_temperature(model)
    assert_array_almost_equal(cnet / 10, res / 10, decimal=1)


def test_calc_adiabatic_lwc():

    lwc_dz = np.array([[0, 0, 2.1, 2.1, 0, 3.2, 3.2],
                       [0, 2.0, 2.0, 0, 1.5, 1.5, 0]])

    height = np.array([10, 12, 14, 16, 18, 20, 22])

    adiabatic_lwc = atmos_utils.calc_adiabatic_lwc(lwc_dz, height)

    expected = np.array([[ 0,  0,  0,  4.2,  0,  0, 6.4],
                         [ 0,  0,  4, 0, 0,  3, 0]])

    assert_array_almost_equal(adiabatic_lwc, expected, decimal=1)
