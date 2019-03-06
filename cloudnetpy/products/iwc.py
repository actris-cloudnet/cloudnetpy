from collections import namedtuple
import numpy as np
import numpy.ma as ma
import cloudnetpy.utils as utils
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
import cloudnetpy.products.product_tools as p_tools
import cloudnetpy.atmos as atmos


class DataCollector(DataSource):
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(self.getvar('radar_frequency'))
        self.spec_liq_atten = self._get_approximative_specific_liquid_atten()
        self.coeffs = self._get_iwc_coeffs()
        self.T, self.meanT = self._get_subzero_temperatures()
        self.Z_factor = self._get_z_factor()

    def _get_z_factor(self):
        """Returns empirical scaling factor for radar echo."""
        return utils.lin2db(self.coeffs.K2liquid0 / 0.93)

    def _get_approximative_specific_liquid_atten(self):
        if self.wl_band == 0:
            return 1.0
        return 4.5

    def _get_iwc_coeffs(self):
        """Returns empirical coefficients for ice water content retrieval."""
        Coefficients = namedtuple('Coefficients', 'K2liquid0 ZT T Z c')
        if self.wl_band == 0:
            return Coefficients(0.878, 0.000242, -0.0186, 0.0699, -1.63)
        return Coefficients(0.669, 0.000580, -0.00706, 0.0923, -0.992)

    def _get_subzero_temperatures(self):
         """Returns freezing temperatures as Celsius."""
        temperature = utils.interpolate_2d(self.getvar('model_time'),
                                           self.getvar('model_height'),
                                           self.getvar('temperature'),
                                           self.time, self.getvar('height'))
        temperature = atmos.k2c(temperature)
        temperature[temperature > 0] = ma.masked
        mean_temperature = ma.mean(temperature, axis=0)
        return temperature, mean_temperature


class IceClassification:
    """Class storing the information about different ice types."""
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.quality_bits = p_tools.read_quality_bits(data_handler)
        self.category_bits = p_tools.read_category_bits(data_handler)
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
        return utils.transpose(self.data_handler.getvar('is_rain'))


def calc_iwc_including_rain(data_handler, ice_class):
    """Calculates ice water content (including ice above rain)."""
    coeffs = data_handler.coeffs
    z = data_handler.getvar('Z') + data_handler.Z_factor
    iwc_including_rain = 10 ** (coeffs.ZT*z*data_handler.T
                                + coeffs.T*data_handler.T
                                + coeffs.Z*z
                                + coeffs.c) * 0.001

    iwc_including_rain[~ice_class.is_ice] = ma.masked
    data_handler.append_data(iwc_including_rain, 'iwc_inc_rain')
    return iwc_including_rain


def calc_iwc(iwc_including_rain, ice_class, data_handler):
    """Returns ice water content ignoring the ice clouds above rain."""
    iwc = ma.copy(iwc_including_rain)
    iwc[ice_class.ice_above_rain] = ma.masked
    data_handler.append_data(iwc, 'iwc')
    return iwc


def calc_iwc_error(data_handler, ice_class):
    """Estimates error of ice water content."""
    coeffs = data_handler.coeffs
    error = data_handler.getvar('Z_error')*(coeffs.ZT*data_handler.T
                                            + coeffs.Z) * 10
    error = utils.l2norm(1.7, error)
    lwp_prior = 250
    error_uncorrected = lwp_prior*0.001*2*data_handler.spec_liq_atten*coeffs.Z * 10
    error[ice_class.uncorrected_ice] = utils.l2norm(1.7, error_uncorrected)
    error[~ice_class.is_ice] = ma.masked
    error[ice_class.ice_above_rain] = ma.masked
    data_handler.append_data(error, 'iwc_error')


def calc_iwc_sensitivity(data_handler):
    """Calculates sensitivity of ice water content."""
    coeffs = data_handler.coeffs
    z_sensitivity = data_handler.getvar('Z_sensitivity') + data_handler.Z_factor
    iwc_sensitivity = 10 ** (coeffs.ZT*z_sensitivity*data_handler.meanT
                             + coeffs.T*data_handler.meanT
                             + coeffs.Z*z_sensitivity
                             + coeffs.c) * 0.001
    data_handler.append_data(iwc_sensitivity, 'iwc_sensitivity')


def calc_iwc_bias(data_handler):
    """Calculates bias of ice water content."""
    iwc_bias = data_handler.getvar('Z_bias')*data_handler.coeffs.Z * 10
    data_handler.append_data(iwc_bias, 'iwc_bias')


def calc_iwc_status(iwc, ice_class, data_handler):
    """Returns information about the status of iwc retrieval."""
    retrieval_status = np.zeros(iwc.shape, dtype=int)

    is_iwc = ~iwc.mask
    retrieval_status[is_iwc] = 1
    retrieval_status[is_iwc & ice_class.uncorrected_ice] = 2
    retrieval_status[is_iwc & ice_class.corrected_ice] = 3
    retrieval_status[~is_iwc & ice_class.is_ice] = 4
    retrieval_status[ice_class.cold_above_rain] = 6
    retrieval_status[ice_class.ice_above_rain] = 5
    retrieval_status[ice_class.would_be_ice & (retrieval_status == 0)] = 7

    data_handler.append_data(retrieval_status, 'iwc_retrieval_status')


def generate_iwc(categorize_file, output_file):
    """High level API to generate Cloudnet ice water content file.

    Args:
        categorize_file (str): Categorize file.
        output_file (str): Output file name.

    """
    data_handler = DataCollector(categorize_file)
    ice_class = IceClassification(data_handler)
    iwc_including_rain = calc_iwc_including_rain(data_handler, ice_class)
    iwc = calc_iwc(iwc_including_rain, ice_class, data_handler)
    calc_iwc_bias(data_handler)
    calc_iwc_error(data_handler, ice_class)
    calc_iwc_sensitivity(data_handler)
    calc_iwc_status(iwc, ice_class, data_handler)

    output.update_attributes(data_handler.data)
    _save_data_and_meta(data_handler, output_file)


def _save_data_and_meta(data_handler, output_file):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(data_handler.time),
            'height': len(data_handler.variables['height'])}
    rootgrp = output.init_file(output_file, dims, data_handler.data, zlib=True)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height')
    output.copy_variables(data_handler.dataset, rootgrp, vars_from_source)
    rootgrp.title = f"Ice water content file from {data_handler.dataset.location}"
    rootgrp.source = f"Categorize file: {p_tools.get_source(data_handler)}"
    output.copy_global(data_handler.dataset, rootgrp, ('location', 'day',
                                                       'month', 'year'))
    output.merge_history(rootgrp, 'ice water content', data_handler)
    rootgrp.close()
