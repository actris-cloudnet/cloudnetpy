"""Module for creating Cloudnet ice water content file using Z-T method."""
from typing import Optional
from collections import namedtuple
import numpy as np
import numpy.ma as ma
from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos
from cloudnetpy.metadata import MetaData
from cloudnetpy.products import product_tools
from cloudnetpy.categorize import DataSource
from cloudnetpy.products.product_tools import ProductClassification

G_TO_KG = 0.001

Coefficients = namedtuple('Coefficients', 'K2liquid0 ZT T Z c')


def generate_iwc(categorize_file: str,
                 output_file: str,
                 keep_uuid: Optional[bool] = False,
                 uuid: Optional[str] = None) -> str:
    """Generates Cloudnet ice water content product.

    This function calculates ice water content using the so-called Z-T method.
    In this method, ice water content is calculated from attenuated-corrected
    radar reflectivity and model temperature. The results are written in a
    netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False when
            new UUID is generated.
        uuid: Set specific UUID for the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_iwc
        >>> generate_iwc('categorize.nc', 'iwc.nc')

    References:
        Hogan, R.J., M.P. Mittermaier, and A.J. Illingworth, 2006:
        The Retrieval of Ice Water Content from Radar Reflectivity Factor and
        Temperature and Its Use in Evaluating a Mesoscale Model.
        J. Appl. Meteor. Climatol., 45, 301–317, https://doi.org/10.1175/JAM2340.1

    """
    iwc_source = IwcSource(categorize_file)
    ice_classification = IceClassification(categorize_file)
    iwc_source.append_iwc_including_rain(ice_classification)
    iwc_source.append_iwc(ice_classification)
    iwc_source.append_bias()
    iwc_source.append_sensitivity()
    iwc_source.append_error(ice_classification)
    iwc_source.append_status(ice_classification)
    date = iwc_source.get_date()
    attributes = output.add_time_attribute(IWC_ATTRIBUTES, date)
    output.update_attributes(iwc_source.data, attributes)
    uuid = output.save_product_file('iwc', iwc_source, output_file, keep_uuid, uuid)
    iwc_source.close()
    return uuid


class IceClassification(ProductClassification):
    """Class storing the information about different ice types.
       Child of ProductClassification().
    """
    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.is_ice = self._find_ice()
        self.would_be_ice = self._find_would_be_ice()
        self.corrected_ice = self._find_corrected_ice()
        self.uncorrected_ice = self._find_uncorrected_ice()
        self.ice_above_rain = self._find_ice_above_rain()
        self.cold_above_rain = self._find_cold_above_rain()

    def _find_ice(self) -> np.ndarray:
        return (self.category_bits['falling']
                & self.category_bits['cold']
                & ~self.category_bits['melting']
                & ~self.category_bits['insect'])

    def _find_would_be_ice(self) -> np.ndarray:
        warm_falling = (self.category_bits['falling']
                        & ~self.category_bits['cold']
                        & ~self.category_bits['insect'])
        return warm_falling | self.category_bits['melting']

    def _find_corrected_ice(self) -> np.ndarray:
        return (self.is_ice
                & self.quality_bits['attenuated']
                & self.quality_bits['corrected'])

    def _find_uncorrected_ice(self) -> np.ndarray:
        return (self.is_ice
                & self.quality_bits['attenuated']
                & ~self.quality_bits['corrected'])

    def _find_ice_above_rain(self) -> np.ndarray:
        is_rain = utils.transpose(self.is_rain)
        return (self.is_ice * is_rain) == 1

    def _find_cold_above_rain(self) -> np.ndarray:
        is_cold = self.category_bits['cold']
        is_rain = utils.transpose(self.is_rain)
        is_cold_rain = (is_cold * is_rain) == 1
        return is_cold_rain & ~self.category_bits['melting']


class IwcSource(DataSource):

    """Data container for ice water content calculations."""
    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(self.getvar('radar_frequency'))
        self.coeffs = self._get_iwc_coeffs()
        self.z_factor = self._get_z_factor()
        self.temperature = self._get_temperature(categorize_file)

    def append_sensitivity(self) -> None:
        """Calculates iwc sensitivity."""
        iwc_sensitivity = self._z_to_iwc('Z_sensitivity')
        self.append_data(iwc_sensitivity, 'iwc_sensitivity')

    def append_bias(self) -> None:
        """Calculates iwc bias."""
        iwc_bias = self.getvar('Z_bias') * self.coeffs.Z * 10
        self.append_data(iwc_bias, 'iwc_bias')

    def append_iwc_including_rain(self, ice_classification: IceClassification) -> None:
        """Calculates ice water content (including ice above rain)."""
        iwc_including_rain = self._z_to_iwc('Z')
        iwc_including_rain[~ice_classification.is_ice] = ma.masked
        self.append_data(iwc_including_rain, 'iwc_inc_rain')

    def append_iwc(self, ice_classification: IceClassification) -> None:
        """Calculates ice water content"""
        iwc = ma.copy(self.data['iwc_inc_rain'][:])
        iwc[ice_classification.ice_above_rain] = ma.masked
        self.append_data(iwc, 'iwc')

    def append_error(self, ice_classification: IceClassification) -> None:
        """Estimates error of ice water content."""

        def _calc_random_error() -> np.ndarray:
            scaled_temperature = self.coeffs.ZT * self.temperature
            scaled_temperature += self.coeffs.Z
            return self.getvar('Z_error') * scaled_temperature * 10

        def _calc_error_in_uncorrected_ice() -> np.ndarray:
            spec_liq_atten = 1.0 if self.wl_band == 0 else 4.5
            liq_atten_scaled = spec_liq_atten * self.coeffs.Z
            lwp_prior = 250  # g / m-2
            return lwp_prior * liq_atten_scaled * 2 * 1e-3 * 10

        retrieval_uncertainty = 1.7  # dB
        random_error = _calc_random_error()
        error_uncorrected = _calc_error_in_uncorrected_ice()
        iwc_error = utils.l2norm(retrieval_uncertainty, random_error)
        iwc_error[ice_classification.uncorrected_ice] = utils.l2norm(retrieval_uncertainty,
                                                                     error_uncorrected)
        iwc_error[(~ice_classification.is_ice | ice_classification.ice_above_rain)] = ma.masked
        self.append_data(iwc_error, 'iwc_error')

    def append_status(self, ice_classification: IceClassification) -> None:
        """Returns information about the status of iwc retrieval."""
        iwc = self.data['iwc'][:]
        retrieval_status = np.zeros(iwc.shape, dtype=int)
        is_iwc = ~iwc.mask
        retrieval_status[is_iwc] = 1
        retrieval_status[is_iwc & ice_classification.corrected_ice] = 2
        retrieval_status[is_iwc & ice_classification.uncorrected_ice] = 3
        retrieval_status[~is_iwc & ice_classification.is_ice] = 4
        retrieval_status[ice_classification.cold_above_rain] = 6
        retrieval_status[ice_classification.ice_above_rain] = 5
        retrieval_status[ice_classification.would_be_ice & (retrieval_status == 0)] = 7
        self.append_data(retrieval_status, 'iwc_retrieval_status')

    def _get_iwc_coeffs(self) -> Coefficients:
        """Returns coefficients for ice water content retrieval.

        References:
            Hogan et.al. 2006, https://doi.org/10.1175/JAM2340.1
        """
        if self.wl_band == 0:
            return Coefficients(0.878, 0.000242, -0.0186, 0.0699, -1.63)
        return Coefficients(0.669, 0.000580, -0.00706, 0.0923, -0.992)

    def _get_z_factor(self) -> float:
        """Returns empirical scaling factor for radar echo."""
        return utils.lin2db(self.coeffs.K2liquid0 / 0.93)

    @staticmethod
    def _get_temperature(categorize_file: str) -> np.ndarray:
        """Returns interpolated temperatures in Celsius."""
        temperature = product_tools.interpolate_model(categorize_file, 'temperature')
        return atmos.k2c(temperature)

    def _z_to_iwc(self, z_variable: str) -> np.ndarray:
        """Calculates temperature weighted z, i.e. ice water content (kg m-3)."""
        if z_variable == 'Z':
            temperature = self.temperature
        else:
            temperature = ma.mean(self.temperature, axis=0)
        z_scaled = self.getvar(z_variable) + self.z_factor
        coeffs = self.coeffs
        return 10 ** (coeffs.ZT * z_scaled * temperature
                      + coeffs.T * temperature
                      + coeffs.Z * z_scaled
                      + coeffs.c) * G_TO_KG


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
