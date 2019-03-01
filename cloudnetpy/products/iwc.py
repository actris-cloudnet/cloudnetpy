import numpy as np
import numpy.ma as ma
import cloudnetpy.utils as utils
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
import cloudnetpy.products.product_tools as p_tools
import cloudnetpy.atmos as atmos


class DataCollecter(DataSource):
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.radar_frequency = self._getvar('radar_frequency')
        self.wl_band = utils.get_wl_band(self.radar_frequency)
        self.spec_liq_atten = self._get_sla()
        self.coeffs = self._get_iwc_coeffs()
        self.T, self.meanT = self._get_subzero_temperatures()
        self.Z_factor = self._get_z_factor()

    def _get_z_factor(self):
        return utils.lin2db(self.coeffs['K2liquid0'] / 0.93)

    def _get_sla(self):
        """ specific liquid attenuation """
        if self.wl_band == 0:
            return 1.0
        return 4.5

    def _get_iwc_coeffs(self):
        if self.wl_band == 0:
            a = 0.878
            b = 0.000242
            c = -0.0186
            d = 0.0699
            e = -1.63
        else:
            a = 0.669
            b = 0.000580
            c = -0.00706
            d = 0.0923
            e = -0.992
        return {'K2liquid0': a, 'cZT': b, 'cT': c, 'cZ': d, 'c': e}

    def _get_subzero_temperatures(self):
        """Returns freezing wet-bulb temperatures as Celsius.

        Notes:
            Positive values are masked.

        """
        temperature = atmos.k2c(self._getvar('Tw'))
        temperature[temperature > 0] = ma.masked
        mean_temperature = ma.mean(temperature, axis=0)
        return temperature, mean_temperature


def generate_iwc(categorize_file, output_file):
    """High level API to generate Cloudnet ice water content file.

    Args:
        categorize_file (str): Categorize file.
        output_file (str): Output file name.

    """
    data_handler = DataCollecter(categorize_file)
    ice_class = classify_ice(data_handler)
    ice_above_rain, cold_above_rain = get_raining(data_handler,
                                                  ice_class['is_ice'])
    iwc = calc_iwc(data_handler, ice_class['is_ice'], ice_above_rain)
    calc_iwc_bias(data_handler)
    calc_iwc_error(data_handler, ice_class, ice_above_rain)
    calc_iwc_sens(data_handler)
    calc_iwc_status(iwc, ice_class, ice_above_rain, cold_above_rain,
                    data_handler)

    output.update_attributes(data_handler.data)
    _save_data_and_meta(data_handler, output_file)


def calc_iwc(data_handler, is_ice, ice_above_rain):
    """Calculates ice water content."""
    z = data_handler.dataset.variables['Z'][:] + data_handler.Z_factor

    iwc = 10**(data_handler.coeffs['cZT']*z*data_handler.T +
               data_handler.coeffs['cT']*data_handler.T +
               data_handler.coeffs['cZ']*z + data_handler.coeffs['c']) * 0.001

    iwc[~is_ice] = ma.masked
    iwc_inc_rain = ma.copy(iwc)
    iwc[ice_above_rain] = ma.masked

    data_handler.append_data(iwc, 'iwc')
    data_handler.append_data(iwc_inc_rain, 'iwc_inc_rain')

    return iwc


def calc_iwc_error(data_handler, ice_class, ice_above_rain):
    """
    TODO:
        Mikä on missing LWP? voiko hakea muualta?
        Mikä on 1.7
    """
    #MISSING_LWP = 250
    error = data_handler.variables['Z_error'][:] * (data_handler.coeffs['cZT']*data_handler.T
                                                    + data_handler.coeffs['cZ'])

    error = utils.l2norm(1.7, error*10)

    lwb_square = 250*0.001*2*data_handler.spec_liq_atten*data_handler.coeffs['cZ']*10

    error[ice_class['uncorrected_ice']] = utils.l2norm(1.7, lwb_square)
    error[~ice_class['is_ice']] = ma.masked
    error[ice_above_rain] = ma.masked

    data_handler.append_data(error, 'iwc_error')


def calc_iwc_bias(data_handler):
    iwc_bias = data_handler.variables['Z_bias'][:] * data_handler.coeffs['cZ']*10
    data_handler.append_data(iwc_bias, 'iwc_bias')


def calc_iwc_sens(data_handler):
    Z = data_handler.variables['Z_sensitivity'][:] + data_handler.Z_factor
    sensitivity = 10 ** (data_handler.coeffs['cZT']*Z*data_handler.meanT +
                         data_handler.coeffs['cT']*data_handler.meanT +
                         data_handler.coeffs['cZ']*Z + data_handler.coeffs['c']) * 0.001
    data_handler.append_data(sensitivity, 'iwc_sensitivity')


def calc_iwc_status(iwc, ice_class, rain_below_cold, rain_below_ice, data_handler):
    retrieval_status = np.zeros(iwc.shape)

    retrieval_status[iwc > 0] = 1
    retrieval_status[iwc > 0 & ice_class['uncorrected_ice']] = 2
    retrieval_status[iwc > 0 & ice_class['corrected_ice']] = 3
    retrieval_status[iwc > 0 & ice_class['is_ice']] = 4
    retrieval_status[rain_below_ice] = 5
    retrieval_status[rain_below_cold] = 6
    retrieval_status[ice_class['would_be_ice'] & (retrieval_status == 0)] = 7

    data_handler.append_data(retrieval_status, 'iwc_retrieval_status')


def classify_ice(data_handler):

    cb = data_handler.variables['category_bits'][:]
    qb = data_handler.variables['quality_bits'][:]

    c_keys = p_tools.get_categorize_keys()
    c_bits = p_tools.check_active_bits(cb, c_keys)
    q_keys = p_tools.get_status_keys()
    q_bits = p_tools.check_active_bits(qb, q_keys)

    is_ice = c_bits['falling'] & c_bits['cold'] & ~c_bits['melting'] & ~c_bits['insect']
    would_be_ice = c_bits['falling'] & ~c_bits['cold'] & ~c_bits['insect']
    corrected_ice = q_bits['attenuated'] & q_bits['corrected'] & is_ice
    uncorrected_ice = q_bits['attenuated'] & ~q_bits['corrected'] & is_ice

    return {'is_ice': is_ice, 'would_be_ice': would_be_ice,
            'corrected_ice': corrected_ice, 'uncorrected_ice': uncorrected_ice}


def get_raining(data_handler, is_ice):
    """ True or False fields indicating raining below a) ice b) cold
    """
    is_cold = utils.isbit(data_handler.variables['category_bits'][:], 2)
    is_rain = data_handler.variables['is_rain'][:] == 1
    is_rain = np.tile(is_rain, (len(data_handler.variables['height'][:]), 1)).T

    ice_above_rain = is_rain & is_ice
    cold_above_rain = is_rain & is_cold

    return ice_above_rain, cold_above_rain


def _save_data_and_meta(data_handler, output_file):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(data_handler.time),
            'height': len(data_handler.variables['height'])}
    rootgrp = output.init_file(output_file, dims, data_handler.data, zlib=True)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height')
    output.copy_variables(data_handler.dataset, rootgrp, vars_from_source)
    rootgrp.title = f"Classification file from {data_handler.dataset.location}"
    rootgrp.source = f"Categorize file: {_get_source(data_handler)}"
    output.copy_global(data_handler.dataset, rootgrp, ('location', 'day',
                                                       'month', 'year'))
    output.merge_history(rootgrp, 'classification', data_handler)
    rootgrp.close()


def _get_source(data_handler):
    """Returns uuid (or filename if uuid not found) of the source file."""
    return getattr(data_handler.dataset, 'file_uuid', data_handler.filename)
