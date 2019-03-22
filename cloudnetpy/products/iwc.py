"""Module for creating Cloudnet ice water content file
using Z-T method.
"""
from collections import namedtuple
import numpy as np
import numpy.ma as ma
import cloudnetpy.utils as utils
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
import cloudnetpy.products.product_tools as p_tools
import cloudnetpy.atmos as atmos

class IwcSource(DataSource):
    """Class containing data needed in the ice water content Z-T method."""
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(self.getvar('radar_frequency'))
        self.spec_liq_atten = self._get_approximative_specific_liquid_atten()
        self.coeffs = self._get_iwc_coeffs()
        self.z_factor = self._get_z_factor()
        self.temperature = self._get_subzero_temperatures()
        self.mean_temperature = self._get_mean_temperature()

    def _get_z_factor(self):
        """Returns empirical scaling factor for radar echo."""
        return utils.lin2db(self.coeffs.K2liquid0 / 0.93)

    def _get_iwc_coeffs(self):
        """Returns coefficients for ice water content retrieval.

        References:
            Hogan et.al. 2006, https://doi.org/10.1175/JAM2340.1
        """
        Coefficients = namedtuple('Coefficients', 'K2liquid0 ZT T Z c')
        if self.wl_band == 0:
            return Coefficients(0.878, 0.000242, -0.0186, 0.0699, -1.63)
        return Coefficients(0.669, 0.000580, -0.00706, 0.0923, -0.992)

    def _get_approximative_specific_liquid_atten(self):
        """Returns approximative liquid water attenuation (dB).

        Returns estimate of the liquid water attenuation for
        pixels that are affected by it but not corrected
        for some reason. The amount of attenuation depends on
        the radar wavelength.

        """
        if self.wl_band == 0:
            return 1.0
        return 4.5

    def _get_subzero_temperatures(self):
        """Returns freezing temperatures in Celsius."""
        temperature = utils.interpolate_2d(self.getvar('model_time'),
                                           self.getvar('model_height'),
                                           self.getvar('temperature'),
                                           self.time, self.getvar('height'))
        temperature = atmos.k2c(temperature)
        temperature[temperature > 0] = ma.masked
        return temperature

    def _get_mean_temperature(self):
        """Returns mean subzero temperatures."""
        return ma.mean(self.temperature, axis=0)


class _IceClassification:
    """Class storing the information about different ice types."""
    def __init__(self, iwc_data):
        self.iwc_data = iwc_data
        self.quality_bits = p_tools.read_quality_bits(iwc_data)
        self.category_bits = p_tools.read_category_bits(iwc_data)
        self.is_ice = self._find_ice()
        self.would_be_ice = self._find_would_be_ice()
        self.corrected_ice = self._find_corrected_ice()
        self.uncorrected_ice = self._find_uncorrected_ice()
        self.ice_above_rain = self._find_ice_above_rain()
        self.cold_above_rain = self._find_cold_above_rain()

    def _find_ice(self):
        return (self.category_bits['falling'] & self.category_bits['cold']
                & ~self.category_bits['melting'] & ~self.category_bits['insect'])

    def _find_would_be_ice(self):
        return (self.category_bits['falling'] & ~self.category_bits['cold']
                & ~self.category_bits['insect'])

    def _find_corrected_ice(self):
        return (self.is_ice & self.quality_bits['attenuated'] &
                self.quality_bits['corrected'])

    def _find_uncorrected_ice(self):
        return (self.is_ice & self.quality_bits['attenuated'] &
                ~self.quality_bits['corrected'])

    def _find_ice_above_rain(self):
        is_rain = self._transpose_rain()
        return (self.is_ice * is_rain) > 0

    def _find_cold_above_rain(self):
        is_cold = self.category_bits['cold']
        is_rain = self._transpose_rain()
        return (is_cold * is_rain) > 0

    def _transpose_rain(self):
        return utils.transpose(self.iwc_data.getvar('is_rain'))


def _z_to_iwc(iwc_data, z_variable):
    """Calculates temperature weighted z, i.e. ice water content."""
    def _get_correct_temperature():
        if z_variable == 'Z':
            return iwc_data.temperature
        return iwc_data.mean_temperature

    temperature = _get_correct_temperature()
    z_scaled = iwc_data.getvar(z_variable) + iwc_data.z_factor
    coeffs = iwc_data.coeffs
    return 10 ** (coeffs.ZT*z_scaled*temperature
                  + coeffs.T*temperature
                  + coeffs.Z*z_scaled
                  + coeffs.c) * 0.001


def _append_iwc_including_rain(iwc_data, ice_class):
    """Calculates ice water content (including ice above rain)."""
    iwc_including_rain = _z_to_iwc(iwc_data, 'Z')
    iwc_including_rain[~ice_class.is_ice] = ma.masked
    iwc_data.append_data(iwc_including_rain, 'iwc_inc_rain')


def _append_iwc(iwc_data, ice_class):
    """Masks ice clouds above rain from ice water content."""
    iwc = ma.copy(iwc_data.data['iwc_inc_rain'][:])
    iwc[ice_class.ice_above_rain] = ma.masked
    iwc_data.append_data(iwc, 'iwc')


def _append_iwc_error(iwc_data, ice_class):
    """Estimates error of ice water content."""

    def _calc_random_error():
        scaled_temperature = iwc_data.coeffs.ZT * iwc_data.temperature
        scaled_temperature += iwc_data.coeffs.Z
        return iwc_data.getvar('Z_error') * scaled_temperature * 10

    def _calc_error_in_uncorrected_ice():
        lwp_prior = 250  # g / m-2
        liq_atten_scaled = iwc_data.spec_liq_atten * iwc_data.coeffs.Z
        return lwp_prior * liq_atten_scaled * 2 * 1e-3 * 10

    retrieval_uncertainty = 1.7  # dB
    random_error = _calc_random_error()
    error_uncorrected = _calc_error_in_uncorrected_ice()
    iwc_error = utils.l2norm(retrieval_uncertainty, random_error)
    iwc_error[ice_class.uncorrected_ice] = utils.l2norm(retrieval_uncertainty,
                                                        error_uncorrected)
    iwc_error[(~ice_class.is_ice | ice_class.ice_above_rain)] = ma.masked
    iwc_data.append_data(iwc_error, 'iwc_error')


def _append_iwc_sensitivity(iwc_data):
    """Calculates sensitivity of ice water content."""
    iwc_sensitivity = _z_to_iwc(iwc_data, 'Z_sensitivity')
    iwc_data.append_data(iwc_sensitivity, 'iwc_sensitivity')


def _append_iwc_bias(iwc_data):
    """Calculates bias of ice water content."""
    iwc_bias = iwc_data.getvar('Z_bias') * iwc_data.coeffs.Z * 10
    iwc_data.append_data(iwc_bias, 'iwc_bias')


def _append_iwc_status(iwc_data, ice_class):
    """Returns information about the status of iwc retrieval."""
    iwc = iwc_data.data['iwc'][:]
    retrieval_status = np.zeros(iwc.shape, dtype=int)
    is_iwc = ~iwc.mask
    retrieval_status[is_iwc] = 1
    retrieval_status[is_iwc & ice_class.uncorrected_ice] = 2
    retrieval_status[is_iwc & ice_class.corrected_ice] = 3
    retrieval_status[~is_iwc & ice_class.is_ice] = 4
    retrieval_status[ice_class.cold_above_rain] = 6
    retrieval_status[ice_class.ice_above_rain] = 5
    retrieval_status[ice_class.would_be_ice & (retrieval_status == 0)] = 7
    iwc_data.append_data(retrieval_status, 'iwc_retrieval_status')


def generate_iwc(categorize_file, output_file):
    """High level API to generate Cloudnet ice water content product.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.

    Examples:
        >>> from cloudnetpy.products.iwc import generate_iwc
        >>> generate_iwc('categorize.nc', 'iwc.nc')

    """
    iwc_data = IwcSource(categorize_file)
    ice_class = _IceClassification(iwc_data)
    _append_iwc_including_rain(iwc_data, ice_class)
    _append_iwc(iwc_data, ice_class)
    _append_iwc_bias(iwc_data)
    _append_iwc_error(iwc_data, ice_class)
    _append_iwc_sensitivity(iwc_data)
    _append_iwc_status(iwc_data, ice_class)
    output.update_attributes(iwc_data.data)
    _save_data_and_meta(iwc_data, output_file)


def _save_data_and_meta(iwc_data, output_file):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(iwc_data.time),
            'height': len(iwc_data.variables['height'])}
    rootgrp = output.init_file(output_file, dims, iwc_data.data, zlib=True)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height')
    output.copy_variables(iwc_data.dataset, rootgrp, vars_from_source)
    rootgrp.title = f"Ice water content file from {iwc_data.dataset.location}"
    rootgrp.source = f"Categorize file: {p_tools.get_source(iwc_data)}"
    output.copy_global(iwc_data.dataset, rootgrp, ('location', 'day',
                                                   'month', 'year'))
    output.merge_history(rootgrp, 'ice water content', iwc_data)
    rootgrp.close()
