import numpy as np
from scipy.interpolate import interp1d
import cloudnetpy.utils as utils
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource

class Data_collecter(DataSource):
    def __init__(self, catfile):
        super().__init__(catfile)
        self.radar_frequency = float(self._getvar('radar_frequency',
                                                  'frequency'))
        self.is35 = self._get_freq()
        self.spec_liq_atten = self._get_sla()
        self.coeffs = self._get_iwc_coeffs()
        self.T, self.meanT = self._get_T()
        self.Z_factor = self.Z_scalefactor()


    def _get_freq(self):
        """ Read and return radar frequency """
        is35 = None
        if 30 < self.radar_frequency < 40:
            is35 = True
        elif 90 < self.radar_frequency < 100:
            is35 = False
        return is35


    def _get_sla(self):
        """ specific liquid attenuation """
        if self.is35:
            return 1.0
        else:
            return 4.5


    def _get_iwc_coeffs(self):
        if self.is35:
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


    def _get_T(self):
        """ linear interpolation of model temperature into target grid """
        f = interp1d(np.array(self.variables['model_height'][:]),
                     np.array(self.variables['temperature'][:]))
        t_height = f(np.array(self.variables['height'][:])) - 273.15
        t_mean = np.mean(t_height, axis=0)

        # TODO: miksi plus-arvot pois?
        t_height[t_height > 0] = 0
        t_mean[t_mean > 0] = 0

        return t_height, t_mean

    def Z_scalefactor(self):
        return 10 * np.log10(self.coeffs['K2liquid0'] / 0.93)


def generate_iwc(cat_file,output_file):
    data_handler = Data_collecter(cat_file)

    ice_class = classificate_ice(data_handler)
    rain_below_ice, rain_below_cold = get_raining(data_handler, ice_class['is_ice'])
    iwc = calc_iwc(data_handler, ice_class['is_ice'], rain_below_ice)

    calc_iwc_bias(data_handler)
    calc_iwc_error(data_handler, ice_class, rain_below_ice)
    calc_iwc_sens(data_handler)
    calc_iwc_status(iwc, ice_class, rain_below_ice,
                   rain_below_cold, data_handler)

    output.update_attributes(data_handler.data)
    _save_data_and_meta(data_handler, output_file)


def calc_iwc(data_handler, is_ice, rain_below_ice):
    """ calculation of ice water content """
    Z = data_handler.dataset.variables['Z'][:] + data_handler.Z_factor
    iwc = 10 ** (data_handler.coeffs['cZT']*Z*data_handler.T +
                 data_handler.coeffs['cT']*data_handler.T +
                 data_handler.coeffs['cZ']*Z + data_handler.coeffs['c']) * 0.001
    iwc[is_ice] = 0.0
    iwc_inc_rain = np.copy(iwc) 
    iwc[rain_below_ice] = np.nan

    data_handler.append_data(iwc, 'iwc')
    data_handler.append_data(iwc_inc_rain, 'iwc_inc_rain')
    return iwc


def calc_iwc_error(data_handler, ice_class, rain_below_ice):
    #MISSING_LWP = 250
    error = data_handler.variables['Z_error'][:] * \
            (data_handler.coeffs['cZT']*data_handler.T + data_handler.coeffs['cZ'])
    error = 1.7**2 + (error * 10)**2
    error[error > 0] = np.sqrt(error[error > 0])

    error[ice_class['uncorrected_ice']] = \
        np.sqrt(1.7**2 + ((250*0.001*2*data_handler.spec_liq_atten)*
                   data_handler.coeffs['cZ']*10)**2)

    error[ice_class['is_ice']] = np.nan
    error[rain_below_ice] = np.nan
    data_handler.append_data(error, 'iwc_error')


def calc_iwc_bias(data_handler):
    iwc_bias = data_handler.variables['Z_bias'][:] * \
           data_handler.coeffs['cZ'] * 10
    data_handler.append_data(iwc_bias,'iwc_bias')


def calc_iwc_sens(data_handler):
    Z = data_handler.variables['Z_sensitivity'][:] + data_handler.Z_factor
    sensitivity = 10 ** (data_handler.coeffs['cZT']*Z*data_handler.meanT +
                         data_handler.coeffs['cT']*data_handler.meanT +
                         data_handler.coeffs['cZ']*Z + data_handler.coeffs['c']) * 0.001
    data_handler.append_data(sensitivity,'iwc_sensitivity')


def check_active_bits(cb, keys):
    """
    Check is observed bin active or not, returns boolean array of
    active and unactive bin index
    """
    bits = {}
    for i, key in enumerate(keys):
        bits[key] = utils.isbit(cb, i)
    return bits


def calc_iwc_status(iwc, ice_class, rain_below_cold, rain_below_ice, data_handler):
    retrieval_status = np.zeros(iwc.shape)

    retrieval_status[iwc > 0] = 1
    retrieval_status[(iwc > 0 & ice_class['uncorrected_ice'])] = 2
    retrieval_status[(iwc > 0 & ice_class['corrected_ice'])] = 3
    retrieval_status[(iwc > 0 & ice_class['is_ice'])] = 4
    retrieval_status[rain_below_ice] = 5
    retrieval_status[rain_below_cold] = 6
    retrieval_status[(ice_class['would_be_ice'] & (retrieval_status == 0))] = 7

    data_handler.append_data(retrieval_status, 'iwc_retrieval_status')


def classificate_ice(data_handler):
    cb, qb = data_handler.variables['category_bits'][:],\
             data_handler.variables['quality_bits'][:]

    keys = ('b1','b2','b4','b8','b16','b32')
    c_bits = check_active_bits(cb, keys)
    qb16 = utils.isbit(qb, 4)
    qb32 = utils.isbit(qb, 5)

    is_ice = c_bits['b2'] & c_bits['b4'] & (c_bits['b8'] == 0) & \
             (c_bits['b32'] == 0)
    would_be_ice = c_bits['b2'] & (c_bits['b4'] == 0) & (c_bits['b32'] == 0)
    corrected_ice = qb16 & qb32 & is_ice
    uncorrected_ice = qb16 & (qb32 == 0) & is_ice

    return {'is_ice':is_ice, 'would_be_ice':would_be_ice,
            'corrected_ice':corrected_ice, 'uncorrected_ice':uncorrected_ice}


def get_raining(data_handler, is_ice):
    """ True or False fields indicating raining below a) ice b) cold """
    a = (data_handler.variables['category_bits'][:] & 4) > 0
    rate = data_handler.variables['rainrate'][:] > 0
    rate = np.tile(rate,(len(data_handler.variables['height'][:]),1)).T
    rain_below_ice = rate & is_ice
    rain_below_cold = rate & a
    return (rain_below_ice, rain_below_cold)



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
