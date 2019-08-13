import pytest
from cloudnetpy.products.iwc import IwcSource
import numpy as np
import numpy.ma as ma
import netCDF4
from numpy import testing
from tests.test import initialize_test_data
import warnings

warnings.filterwarnings("ignore")

test_data_path = initialize_test_data()


# Test iwcSource class
@pytest.fixture
def iwcS():
    fname = test_data_path + '/test_data_categorize.nc'
    return IwcSource(fname)


@pytest.fixture
def freq():
    return [0, 1]


def test_wl_band(iwcS, freq):
    errors = []
    for f in freq:
        if not iwcS.wl_band == f:
            errors.append("Wavelength band not correct")
        else:
            correct = True
    if 'correct' in locals():
        errors = []
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("liq_att", [{0: 1.0, 1: 4.5}])
def test_get_approximative_specific_liquid_atten(iwcS, liq_att):
    testing.assert_equal(iwcS.spec_liq_atten, liq_att[iwcS.wl_band])


@pytest.mark.parametrize("coeff", [
    {0: (0.878, 0.000242, -0.0186, 0.0699, -1.63),
    1: (0.669, 0.000580, -0.00706, 0.0923, -0.992)}])
def test_get_iwc_coeffs(iwcS, coeff):
    testing.assert_equal(iwcS.coeffs[0:], coeff[iwcS.wl_band])


@pytest.mark.parametrize("factor", [{0: -0.24988, 1: -1.43056}])
def test_get_z_factor(iwcS, factor):
    testing.assert_equal(round(iwcS.z_factor, 5), factor[iwcS.wl_band])


def test_get_subzero_temperatures(iwcS):
    testing.assert_array_less(np.max(iwcS.temperature), 0)


def test_get_mean_temperature(iwcS):
    testing.assert_array_equal(iwcS.mean_temperature,
                               np.mean(iwcS.temperature, axis=0))


# Other methods from iwc
@pytest.mark.parametrize("coeff, z, T, result",[
    (0.1, 2.0, 3.0, 5.75),
    (0.02, 0.8, 0.4, 1.05)])
def test_z_to_iwc(coeff, z, T, result):
    x = 10 ** (coeff*z*T + coeff*0.2*T + coeff*0.3*z + coeff*0.4)
    testing.assert_equal(round(x, 2), result)


@pytest.mark.parametrize("coeff, result",[
    (1, 5),
    (5, 25)])
def test_calc_error_in_uncorrected_ice(coeff, result):
    x = 250 * coeff * 2 * 1e-3 * 10
    testing.assert_equal(x, result)


@pytest.fixture
def iwc_data():
    nc_file = test_data_path + 'test_data_iwc.nc'
    data_file = netCDF4.Dataset(nc_file).variables['iwc'][:]
    return data_file[data_file == data_file.filled()] == ma.masked


def test_iwc_quality(iwc_data):
    #iwc data quality testing
    # checks if array values are between wanted boundaries
    testing.assert_array_less(iwc_data, 1e-4)
    assert not iwc_data.any() <= 1e-7



