"""Module for creating Cloudnet ice water content file
using Z-T method.
"""
from typing import Union
from collections import namedtuple
import numpy as np
import numpy.ma as ma
from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos
from cloudnetpy.metadata import MetaData
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.categorize import DataSource
from cloudnetpy.products.product_tools import ProductClassification

G_TO_KG = 0.001


def generate_iwc(categorize_file: str,
                 output_file: str,
                 keep_uuid: bool = False,
                 uuid: Union[str, None] = None):
    """Generates Cloudnet ice water content product.

    This function calculates ice water content using the so-called Z-T method.
    In this method, ice water content is calculated from attenuated-corrected
    radar reflectivity and model temperature. The results are written in a
    netCDF file.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_iwc
        >>> generate_iwc('categorize.nc', 'iwc.nc')

    References:
        Hogan, R.J., M.P. Mittermaier, and A.J. Illingworth, 2006:
        The Retrieval of Ice Water Content from Radar Reflectivity Factor and
        Temperature and Its Use in Evaluating a Mesoscale Model.
        J. Appl. Meteor. Climatol., 45, 301–317, https://doi.org/10.1175/JAM2340.1

    """
    iwc_data = IwcSource(categorize_file)
    ice_class = _IceClassification(categorize_file)
    _append_iwc_including_rain(iwc_data, ice_class)
    _append_iwc(iwc_data, ice_class)
    _append_iwc_bias(iwc_data)
    _append_iwc_error(iwc_data, ice_class)
    _append_iwc_sensitivity(iwc_data)
    _append_iwc_status(iwc_data, ice_class)
    date = iwc_data.get_date()
    attributes = output.add_time_attribute(IWC_ATTRIBUTES, date)
    output.update_attributes(iwc_data.data, attributes)
    uuid = output.save_product_file('iwc', iwc_data, output_file, keep_uuid, uuid)
    iwc_data.close()
    return uuid


class IwcSource(DataSource):
    """Class containing data needed in the ice water content Z-T method."""
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(self.getvar('radar_frequency'))
        self.spec_liq_atten = self._get_approximate_specific_liquid_atten()
        self.coeffs = self._get_iwc_coeffs()
        self.z_factor = self._get_z_factor()
        self.temperature = self._get_temperature_field(categorize_file)
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

    def _get_approximate_specific_liquid_atten(self):
        """Returns approximate liquid water attenuation (dB).

        Returns estimate of the liquid water attenuation for
        pixels that are affected by it but not corrected
        for some reason.

        """
        if self.wl_band == 0:
            return 1.0
        return 4.5

    @staticmethod
    def _get_temperature_field(cat_file):
        """Returns interpolated temperatures in Celsius."""
        temperature = p_tools.interpolate_model(cat_file, 'temperature')
        return atmos.k2c(temperature)

    def _get_mean_temperature(self):
        """Returns mean temperature for each altitude."""
        return ma.mean(self.temperature, axis=0)


class _IceClassification(ProductClassification):
    """Class storing the information about different ice types.
       Child of ProductClassification().
    """
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.is_ice = self._find_ice()
        self.would_be_ice = self._find_would_be_ice()
        self.corrected_ice = self._find_corrected_ice()
        self.uncorrected_ice = self._find_uncorrected_ice()
        self.ice_above_rain = self._find_ice_above_rain()
        self.cold_above_rain = self._find_cold_above_rain()

    def _find_ice(self):
        return (self.category_bits['falling']
                & self.category_bits['cold']
                & ~self.category_bits['melting']
                & ~self.category_bits['insect'])

    def _find_would_be_ice(self):
        warm_falling = (self.category_bits['falling']
                        & ~self.category_bits['cold']
                        & ~self.category_bits['insect'])
        return warm_falling | self.category_bits['melting']

    def _find_corrected_ice(self):
        return (self.is_ice
                & self.quality_bits['attenuated']
                & self.quality_bits['corrected'])

    def _find_uncorrected_ice(self):
        return (self.is_ice
                & self.quality_bits['attenuated']
                & ~self.quality_bits['corrected'])

    def _find_ice_above_rain(self):
        is_rain = utils.transpose(self.is_rain)
        return (self.is_ice * is_rain) == 1

    def _find_cold_above_rain(self):
        is_cold = self.category_bits['cold']
        is_rain = utils.transpose(self.is_rain)
        return (((is_cold * is_rain) == 1)
                & ~self.category_bits['melting'])


def _z_to_iwc(iwc_data, z_variable):
    """Calculates temperature weighted z, i.e. ice water content (kg m-3)."""
    def _get_correct_temperature():
        if z_variable == 'Z':
            return iwc_data.temperature
        return iwc_data.mean_temperature

    temperature = _get_correct_temperature()
    z_scaled = iwc_data.getvar(z_variable) + iwc_data.z_factor
    coeffs = iwc_data.coeffs
    return 10 ** (coeffs.ZT * z_scaled * temperature
                  + coeffs.T * temperature
                  + coeffs.Z * z_scaled
                  + coeffs.c) * G_TO_KG


def _append_iwc_including_rain(iwc_data, ice_class):
    """Calculates ice water content (including ice above rain)."""
    iwc_including_rain = _z_to_iwc(iwc_data, 'Z')
    iwc_including_rain[~ice_class.is_ice] = ma.masked
    iwc_data.append_data(iwc_including_rain, 'iwc_inc_rain')


def _append_iwc(iwc_data, ice_class):
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
    iwc_sensitivity = _z_to_iwc(iwc_data, 'Z_sensitivity')
    iwc_data.append_data(iwc_sensitivity, 'iwc_sensitivity')


def _append_iwc_bias(iwc_data):
    iwc_bias = iwc_data.getvar('Z_bias') * iwc_data.coeffs.Z * 10
    iwc_data.append_data(iwc_bias, 'iwc_bias')


def _append_iwc_status(iwc_data, ice_class):
    """Returns information about the status of iwc retrieval."""
    iwc = iwc_data.data['iwc'][:]
    retrieval_status = np.zeros(iwc.shape, dtype=int)
    is_iwc = ~iwc.mask
    retrieval_status[is_iwc] = 1
    retrieval_status[is_iwc & ice_class.corrected_ice] = 2
    retrieval_status[is_iwc & ice_class.uncorrected_ice] = 3
    retrieval_status[~is_iwc & ice_class.is_ice] = 4
    retrieval_status[ice_class.cold_above_rain] = 6
    retrieval_status[ice_class.ice_above_rain] = 5
    retrieval_status[ice_class.would_be_ice & (retrieval_status == 0)] = 7
    iwc_data.append_data(retrieval_status, 'iwc_retrieval_status')


COMMENTS = {
    'iwc':
        ('This variable was calculated from the radar reflectivity factor, after\n'
         'correction for gaseous and liquid attenuation, and temperature taken\n'
         'from a forecast model, using an empirical formula.'),

    'iwc_error':
        ('This variable is an estimate of the one-standard-deviation random error\n'
         'in ice water content due to both the uncertainty of the retrieval\n'
         '(about +50%/-33%, or 1.7 dB), and the random error in radar reflectivity\n'
         'factor from which ice water content was calculated. When liquid water is\n'
         'present beneath the ice but no microwave radiometer data were available to\n'
         'correct for the associated attenuation, the error also includes a\n'
         'contribution equivalent to approximately 250 g m-2 of liquid water path\n'
         'being uncorrected for.'),

    'iwc_bias':
        ('This variable is an estimate of the possible systematic error in \n'
         'ice water content due to the calibration error of the radar \n'
         'reflectivity factor from which it was calculated.'),

    'iwc_sensitivity':
        ('This variable is an estimate of the minimum detectable ice water\n'
         'content as a function of height.'),

    'iwc_retrieval_status':
        ('This variable describes whether a retrieval was performed\n'
         'for each pixel, and its associated quality.'),

    'iwc_inc_rain':
        ('This variable is the same as iwc but it also contains iwc values\n'
         'above rain. The iwc values above rain have been severely affected\n'
         'by attenuation and should be used when the effect of attenuation\n'
         'is being studied.'),
}

DEFINITIONS = {
    'iwc_retrieval_status':
    ('\n'
     'Value 0: No ice present.\n'
     'Value 1: Reliable retrieval.\n'
     'Value 2: Unreliable retrieval: Radar corrected using liquid water path\n'
     '         data which can be inaccurate.'
     'Value 3: Unreliable retrieval: Uncorrected liquid attenuation due to\n'
     '         missing liquid water path data.'
     'Value 4: No retrieval: Ice detected only by the lidar.\n'
     'Value 5: No retrieval: Rain below the detected ice leads to large\n'
     '         uncertainties.\n'
     'Value 6: Clear sky above rain and wet-bulb temperature less than 0degC: '
     '         if rain attenuation is strong, ice could be present but undetected.\n'
     'Value 7: Drizzle or rain that would have been classified as ice if the\n'
     '         wet-bulb temperature were less than 0degC.')
}

IWC_ATTRIBUTES = {
    'iwc': MetaData(
        long_name='Ice water content',
        units='kg m-3',
        comment=COMMENTS['iwc'],
        ancillary_variables='iwc_sensitivity iwc_bias'
    ),
    'iwc_error': MetaData(
        long_name='Random error in ice water content, one standard deviation',
        units='dB',
        comment=COMMENTS['iwc_error']
    ),
    'iwc_bias': MetaData(
        long_name='Possible bias in ice water content, one standard deviation',
        units='dB',
        comment=COMMENTS['iwc_bias']
    ),
    'iwc_sensitivity': MetaData(
        long_name='Minimum detectable ice water content',
        units='kg m-3',
        comment=COMMENTS['iwc_sensitivity']
    ),
    'iwc_retrieval_status': MetaData(
        long_name='Ice water content retrieval status',
        comment=COMMENTS['iwc_retrieval_status'],
        definition=DEFINITIONS['iwc_retrieval_status'],
    ),
    'iwc_inc_rain': MetaData(
        long_name='Ice water content including rain',
        units='kg m-3',
        comment=COMMENTS['iwc_inc_rain'],
        ancillary_variables='iwc_sensitivity iwc_bias'
    )
}
