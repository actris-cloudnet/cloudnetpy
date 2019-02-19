import netCDF4
import numpy as np
from scipy.interpolate import interp1d
import cloudnetpy.utils as utils
from cloudnetpy.categorize import DataSource
import cloudnetpy.products.ncf as ncf
from cloudnetpy.products.ncf import CnetVar

"""
Tää koko homma on vähän palapeli, jossa on kaikki osat vähän levällään.
Pitää katsoa, mitä saa pois, minkä siirrettyä toisaalle ym.
Aika paljon yksittäisiä funktioita moduuli täynnä.
"""
def generate_iwc(cat_file):
    data_handler = DataSource(cat_file)
    vrs = data_handler.dataset.variables             # variables
    nalt = data_handler.dataset.dimensions['height']

    # Määrittää mittalaitekorkeudelle interpoloitu lämpötila ja y-akselin keskiarvo
    (T, meanT) = get_T(vrs)

    # Palauttaa sanakirjan, jossa on halutut jäätilanne luokat
    ice_class = classificate_ice(vrs)

    # Noutaa radar frekvenssi ja boolean is35
    (freq, is35) = ncf.get_radar_freq(vrs)

    # Palauttaa random numeroita
    spec_liq_atten = get_sla(is35)

    # Palautaa random kertoimia
    coeffs = get_iwc_coeffs(is35)

    # tuskin tarvii muutoksia
    iwc_bias = calc_iwc_bias(vrs, coeffs)

    # Vain muutama asia, minkä toteuttaa, hyvä sellaisenaan
    (rain_below_ice, rain_below_cold) = get_raining(vrs, ice_class['is_ice'], nalt)


    iwc_error = calc_iwc_error(vrs, coeffs, T, ice_class, spec_liq_atten, rain_below_ice)    
    (iwc, iwc_inc_rain) = calc_iwc(vrs['Z'][:], coeffs, T, ice_class['is_ice'], rain_below_ice)
    iwc_sensitivity = calc_iwc_sens(vrs['Z_sensitivity'][:], coeffs, meanT)
    retrieval_status = calc_iwc_status(iwc, ice_class, rain_below_ice, rain_below_cold)
    obs = lwc2cnet({'iwc_inc_rain': iwc_inc_rain,
                    'radar_frequency':freq,
                    'iwc':iwc, 'iwc_bias':iwc_bias,
                    'iwc_error':iwc_error,
                    'iwc_sensitivity':iwc_sensitivity,
                    'iwc_retrieval_status': retrieval_status})
    return (data_handler.dataset, obs)


def lwc2cnet(vars_in):
    """ Defines IWC Cloudnet objects """
    log, lin = 'logarithmic', 'linear'
    obs = []

    #freq = str(int(vars_in['radar_frequency']))
    
    s, lname = 'iwc', 'Ice water content'
    obs.append(CnetVar(s, vars_in[s], long_name=lname, units='kg m-3',
                       plot_scale=log, plot_range=(1e-7, 0.001), error_variable=True, bias_variable=True,
                       comment=("comment"),
                       extra_attributes={'sensitivity_variable':'iwc_sensitivity'}))

    s = 'iwc_error'
    obs.append(CnetVar(s, vars_in[s], plot_scale=lin, plot_range=(0, 3), long_name=ncf.err_name(lname), units='dB',
                       comment=("comment")))

    s = 'iwc_bias'
    obs.append(CnetVar(s, vars_in[s], long_name=ncf.bias_name(lname), units='dB', size=(), fill_value=None, comment=ncf.bias_comm(lname)))
                           
    s = 'iwc_sensitivity'
    obs.append(CnetVar(s, vars_in[s], long_name="Minimum detectable ice water content", units='kg m-3', size=('height'),
                       comment=("This variable is an estimate of the minimum detectable ice water content as a function of height.")))

    s = 'iwc_retrieval_status'
    obs.append(CnetVar(s, vars_in[s], long_name=ncf.status_name(lname), units='', data_type='b', fill_value=None, plot_range=(0,7),
                       comment=("comment"),
                       extra_attributes = {'definition':'definition'}))

    s = 'iwc_inc_rain'
    obs.append(CnetVar(s, vars_in[s], long_name=lname, units='kg m-3',
                       plot_scale=log, plot_range=(1e-7, 0.001), error_variable='iwc_error', bias_variable='iwc_bias', 
                       comment=("comment"),
                       extra_attributes={'sensitivity_variable':'iwc_sensitivity'}))
    return obs

            
def calc_iwc_status(iwc, ice_class, rain_below_cold, rain_below_ice):
    retrieved_ice = iwc > 0    
    retrieval_status = np.zeros((len(iwc),len(iwc[0])))
    retrieval_status[retrieved_ice] = 1
    retrieval_status[(retrieved_ice & ice_class['uncorrected_ice'])] = 2
    retrieval_status[(retrieved_ice & ice_class['corrected_ice'])] = 3
    retrieval_status[(~retrieved_ice & ice_class['is_ice'])] = 4
    retrieval_status[rain_below_cold] = 6
    retrieval_status[rain_below_ice] = 5    
    retrieval_status[(ice_class['would_be_ice'] & (retrieval_status == 0))] = 7
    return retrieval_status
    
def calc_iwc_sens(Z_sensitivity, coeffs, meanT):
    Z = Z_sensitivity + Z_scalefactor(coeffs['K2liquid0'])
    sensitivity = 10 ** (coeffs['cZT']*Z*meanT + coeffs['cT']*meanT + coeffs['cZ']*Z + coeffs['c']) * 0.001
    return sensitivity

def calc_iwc(Z, coeffs, T, is_ice, rain_below_ice):
    """ calculation of ice water content """
    Z = Z + Z_scalefactor(coeffs['K2liquid0'])
    iwc = 10 ** (coeffs['cZT']*Z*T + coeffs['cT']*T + coeffs['cZ']*Z + coeffs['c']) * 0.001
    iwc[~is_ice] = 0.0
    iwc_inc_rain = np.copy(iwc) 
    iwc[rain_below_ice] = np.nan
    return (iwc, iwc_inc_rain)
    
def get_raining(vrs, is_ice, nalt):
    """ True or False fields indicating raining below a) ice b) cold """
    a = (vrs['category_bits'][:] & 4) > 0
    rate = vrs['rainrate'][:] > 0  # True / False vector
    rate = np.tile(rate,(nalt,1)).T
    rain_below_ice = rate & is_ice
    rain_below_cold = rate & a
    return (rain_below_ice, rain_below_cold)


def check_active_bits(cb, keys):
    """
    Check is observed bin active or not, returns boolean array of
    active and unactive bin index
    """
    bits = {}
    for i, key in enumerate(keys):
        bits[key] = utils.isbit(cb, i)
    return bits


def classificate_ice(vrs):
    cb, qb = vrs['category_bits'][:], vrs['quality_bits'][:]
    keys = ('cb1','cb2','cb4','cb8','cb16','cb32')
    print(type(keys))
    c_bits = check_active_bits(cb, keys)
    q_bits = check_active_bits(qb, keys)

    is_ice = c_bits['cb2'] & c_bits['cb4'] & c_bits['cb8'] == 0 & c_bits['cb32'] == 0
    would_be_ice = c_bits['cb2'] & c_bits['cb4'] == 0 & c_bits['cb32'] == 0
    corrected_ice = q_bits['cb16'] & q_bits['cb32'] & is_ice
    uncorrected_ice = q_bits['cb16'] & q_bits['cb32'] == 0 & is_ice

    # Lisätään vielä data
    # do classification:
    #is_ice = (cb2 > 0) & (cb4 > 0) & (cb8 == 0) & (cb32 == 0)
    #would_be_ice = (cb2 > 0) & (cb4 == 0) & (cb32 == 0)
    #corrected_ice = (qb16 > 0) & (qb32 > 0) & is_ice
    #uncorrected_ice = (qb16 > 0) & (qb32 == 0) & is_ice


    return {'is_ice':is_ice, 'would_be_ice':would_be_ice, 'corrected_ice':corrected_ice, 'uncorrected_ice':uncorrected_ice}


def calc_iwc_bias(vrs, coeffs):
    return vrs['Z_bias'][:] * coeffs['cZ'] * 10

def calc_iwc_error(vrs, coeffs, T, ice_class, spec_liq_atten, rain_below_ice):
    # iwc_error = calc_iwc_error(vrs, coeffs, T, ice_class, spec_liq_atten, rain_below_ice)
    MISSING_LWP = 250
    error = vrs['Z_error'][:] * (coeffs['cZT']*T + coeffs['cZ'])
    error = 1.7**2 + (error * 10)**2
    error[error > 0] = np.sqrt(error[error > 0])
    error[ice_class['uncorrected_ice']] = np.sqrt(1.7**2 + ((MISSING_LWP*0.001*2*spec_liq_atten)*coeffs['cZ']*10)**2)
    error[ice_class['is_ice']] = np.nan
    error[rain_below_ice] = np.nan
    return error

def Z_scalefactor(K2liquid0):
    return 10 * np.log10(K2liquid0 / 0.93)

def get_sla(is35):
    """ specific liquid attenuation """
    if is35:
        return 1.0
    else:
        return 4.5
    
def get_T(vrs):
    """ linear interpolation of model temperature into target grid """
    # Interpoloidaan mallikentän korkeus ja havaintojen lämpötila
    # Tuottaa olion
    f = interp1d(np.array(vrs['model_height'][:]), np.array(vrs['temperature'][:]))
    T_height = f(np.array(vrs['height'][:])) - 273.15
    T_mean = np.mean(T_height, axis=0)

    #TODO: miksi plus-arvot pois?
    T_height[T_height > 0] = 0
    T_mean[T_mean > 0] = 0

    return (T_height, T_mean)

def get_iwc_coeffs(is35):
    if is35:
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
    return {'K2liquid0':a, 'cZT':b, 'cT':c, 'cZ':d, 'c':e}




    
