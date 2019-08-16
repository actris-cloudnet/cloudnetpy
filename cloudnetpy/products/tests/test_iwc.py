import warnings
import pytest
import numpy as np
import numpy.ma as ma
from numpy import testing
import netCDF4
from cloudnetpy.products.iwc import IwcSource
from tests.test import get_test_file_name

warnings.filterwarnings("ignore")


@pytest.fixture
def iwc_source():
    fname = get_test_file_name('categorize')
    return IwcSource(fname)


@pytest.fixture
def iwc_data():
    nc_file = get_test_file_name('iwc')
    iwc_data = netCDF4.Dataset(nc_file).variables['iwc'][:]
    return iwc_data[iwc_data == iwc_data.filled()] == ma.masked


@pytest.fixture
def freq():
    return [0, 1]


def test_wl_band(iwc_source, freq):
    errors = []
    for f in freq:
        if not iwc_source.wl_band == f:
            errors.append("Wavelength band not correct")
        else:
            correct = True
    if 'correct' in locals():
        errors = []
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("liq_att", [{0: 1.0, 1: 4.5}])
def test_get_approximative_specific_liquid_atten(iwc_source, liq_att):
    testing.assert_equal(iwc_source.spec_liq_atten, liq_att[iwc_source.wl_band])


@pytest.mark.parametrize("coeff", [
    {0: (0.878, 0.000242, -0.0186, 0.0699, -1.63),
     1: (0.669, 0.000580, -0.00706, 0.0923, -0.992)}])
def test_get_iwc_coeffs(iwc_source, coeff):
    testing.assert_equal(iwc_source.coeffs[0:], coeff[iwc_source.wl_band])


@pytest.mark.parametrize("factor", [{0: -0.24988, 1: -1.43056}])
def test_get_z_factor(iwc_source, factor):
    testing.assert_equal(round(iwc_source.z_factor, 5), factor[iwc_source.wl_band])


def test_get_mean_temperature(iwc_source):
    testing.assert_array_equal(iwc_source.mean_temperature,
                               np.mean(iwc_source.temperature, axis=0))


@pytest.mark.parametrize("coeff, z, T, result", [
    (0.1, 2.0, 3.0, 5.75),
    (0.02, 0.8, 0.4, 1.05)])
def test_z_to_iwc(coeff, z, T, result):
    x = 10 ** (coeff*z*T + coeff*0.2*T + coeff*0.3*z + coeff*0.4)
    testing.assert_equal(round(x, 2), result)


@pytest.mark.parametrize("coeff, result", [
    (1, 5),
    (5, 25)])
def test_calc_error_in_uncorrected_ice(coeff, result):
    x = 250 * coeff * 2 * 1e-3 * 10
    testing.assert_equal(x, result)


def test_iwc_quality(iwc_data):
    testing.assert_array_less(iwc_data, 1e-4)
    assert not iwc_data.any() <= 1e-7



