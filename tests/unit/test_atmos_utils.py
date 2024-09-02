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
    "vapor_pressure, res",
    [
        (500, 270.37),
        (300, 263.68),
        (200, 258.63),
        (100, 250.48),
    ],
)
def test_dew_point_temperature(vapor_pressure, res):
    """Unit tests for atmos.dew_point()."""
    assert_array_almost_equal(
        atmos_utils.calc_dew_point_temperature(np.array(vapor_pressure)),
        res,
        decimal=1,
    )


@pytest.mark.parametrize(
    "t, p, rh, res",
    [
        (280, 101330, 0.2, 273.05),
        (250, 90000, 0.01, 248.73),
    ],
)
def test_wet_bulb(t, p, rh, res):
    """Unit tests for atmos.wet_bulb()."""
    model = {
        "temperature": np.array(t),
        "pressure": np.array(p),
        "rh": np.array(rh),
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
