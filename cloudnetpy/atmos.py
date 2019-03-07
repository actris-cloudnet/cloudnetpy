""" This module contains functions to calculate
various atmospheric parameters.
"""
import numpy as np
import numpy.ma as ma
from cloudnetpy import constants as con
from cloudnetpy import lwc, utils


def c2k(temp):
    """Converts Celsius to Kelvins."""
    return ma.array(temp) + 273.15


def k2c(temp):
    """Converts Kelvins to Celsius."""
    return ma.array(temp) - 273.15


VAISALA_PARAMS_OVER_WATER = (6.116441, 7.591386, 240.7263)


def saturation_vapor_pressure(temp_kelvin):
    """Returns approximate water vapour saturation pressure.

    Args:
        temp_kelvin (ndarray): Temperature in K.

    Returns:
        ndarray: Vapor saturation pressure in Pa.

    References:
        Vaisala's white paper: "Humidity conversion formulas".

        """
    a, m, tn = VAISALA_PARAMS_OVER_WATER
    temp_celsius = k2c(temp_kelvin)
    return a * 10**((m*temp_celsius) / (temp_celsius+tn)) * 100


def dew_point_temperature(vapor_pressure):
    """ Returns dew point temperature.

    Args:
        vapor_pressure (ndarray): Water vapor pressure (Pa).

    Returns:
        ndarray: Dew point temperature (K).

    Notes:
        Method from Vaisala's white paper: "Humidity conversion formulas".

    """
    a, m, tn = VAISALA_PARAMS_OVER_WATER
    dew_point_celsius = tn / ((m/np.log10(vapor_pressure/100/a))-1)
    return c2k(dew_point_celsius)


def wet_bulb(model_data):
    """Returns wet bulb temperature.

    Returns wet bulb temperature for given temperature,
    pressure and relative humidity. Algorithm is based on a Taylor
    expansion of a simple expression for the saturated vapour pressure.

    Args:
        model_data (dict): Model variables 'temperature', 'pressure', 'rh'.

    Returns:
        ndarray: Wet bulb temperature (K).

    References:
        J. Sullivan and L. D. Sanders: Method for obtaining wet-bulb
        temperatures by modifying the psychrometric formula.

    """
    def _screen_rh():
        rh = model_data['rh']
        rh[rh < 1e-5] = 1e-5
        return rh

    def _vapor_derivatives():
        m = 17.269
        tn = 35.86
        a = m*(tn - con.T0)
        b = dew_point - tn
        first = -vapor_pressure*a/(b**2)
        second = vapor_pressure*((a/(b**2))**2 + 2*a/(b**3))
        return first, second

    def _psychrometric_constant():
        return (model_data['pressure'] * con.specific_heat
                / (con.latent_heat * con.mw_ratio))

    relative_humidity = _screen_rh()
    saturation_pressure = saturation_vapor_pressure(model_data['temperature'])
    vapor_pressure = saturation_pressure * relative_humidity
    dew_point = dew_point_temperature(vapor_pressure)
    first_der, second_der = _vapor_derivatives()
    psychrometric_const = _psychrometric_constant()
    a = 0.5*second_der
    b = first_der + psychrometric_const - dew_point*second_der
    c = (-model_data['temperature']*psychrometric_const
         - dew_point*first_der
         + 0.5*dew_point**2*second_der)
    return (-b+np.sqrt(b*b-4*a*c))/(2*a)


def gas_atten(model, cat_bits, height):
    """Returns gas attenuation (assumes saturation inside liquid droplets).

    Args:
        model (dict): Interpolated 2-D model fields {'gas_atten',
            'specific_gas_atten', 'specific_saturated_gas_atten'}.
        cat_bits (ndarray): 2-D array of integers containing
            categorize flag bits.
        height (ndarray): 1-D altitude grid (m).

    Returns:
        dict: 'radar_gas_atten' containing the attenuation
            due to atmospheric gases.

    Notes:
        Could be combined with liquid_atten if a common
            Attenuations class is created.

    """
    dheight = utils.mdiff(height)
    is_liquid = utils.isbit(cat_bits, 0)
    spec = np.copy(model['specific_gas_atten'][:])
    spec[is_liquid] = model['specific_saturated_gas_atten'][:][is_liquid]
    layer1_att = model['gas_atten'][:][:, 0]
    gas_att = 2*np.cumsum(spec.T, axis=0)*dheight*1e-3 + layer1_att
    gas_att = np.insert(gas_att.T, 0, layer1_att, axis=1)[:, :-1]
    return {'radar_gas_atten': gas_att}


def liquid_atten(mwr, model, classification, height):
    """Calculates attenuation due to liquid water.

    Args:
        mwr (Mwr): Mwr data container.
        model (Model): Model data container.
        classification (ClassificationResult): Classification container.
        height (ndarray): 1-D altitude grid (m).

    Returns:
        Dict containing

        - **radar_liquid_atten** (*MaskedArray*): Amount of liquid attenuation.
        - **liquid_atten_err** (*MaskedArray*): Error in the liquid attenuation.
        - **liquid_corrected** (*ndarray*): Boolean array denoting where liquid
          attenuation is present and we can compute its value.
        - **liquid_uncorrected** (*ndarray*): Boolean array denoting where liquid
          attenuation is present but we can not compute its value.

    Notes:
        Too complicated function! Needs to be broken into a class.

    """

    spec_liq = model['specific_liquid_atten']
    is_liq = utils.isbit(classification.category_bits, 0)
    lwc_dz, lwc_dz_err, liq_att, liq_att_err, lwp_norm, lwp_norm_err = utils.init(6, is_liq.shape)
    ind = np.where(classification.liquid_bases)
    lwc_dz[ind] = lwc.adiabatic_lwc(model['temperature'][ind],
                                    model['pressure'][ind])
    lwc_dz_err[is_liq] = utils.ffill(lwc_dz[is_liq])
    ind_from_base = utils.cumsumr(is_liq, axis=1)
    lwc_adiab = ind_from_base*lwc_dz_err*utils.mdiff(height)*1e3
    ind = np.isfinite(mwr['lwp'][:]) & np.any(is_liq, axis=1)
    lwp_norm[ind, :] = (lwc_adiab[ind, :].T*mwr['lwp'][:][ind]
                        / np.sum(lwc_adiab[ind, :], axis=1)).T
    lwp_norm_err[ind, :] = (lwc_dz_err[ind, :].T*mwr['lwp_error'][:][ind]
                            / np.sum(lwc_dz_err[ind, :], axis=1)).T
    liq_att[:, 1:] = 2e-3*np.cumsum(lwp_norm[:, :-1]*spec_liq[:, :-1], axis=1)
    liq_att_err[:, 1:] = 2e-3*np.cumsum(lwp_norm_err[:, :-1]*spec_liq[:, :-1],
                                        axis=1)
    liq_att, corr_atten, uncorr_atten = _screen_liq_atten(liq_att, classification)
    return {'radar_liquid_atten': liq_att,
            'liquid_atten_err': liq_att_err,
            'liquid_corrected': corr_atten,
            'liquid_uncorrected': uncorr_atten}


def _screen_liq_atten(liq_atten, classification):
    """Removes corrupted data from liquid attenuation.

    Args:
        liq_atten (ndarray): Liquid attenuation.
        classification (ClassificationResult): Classification container.

    Returns:
        tuple: 3-element tuple containing:
        
        - MaskedArray: Screened liquid attenuation.
        - ndarray: Boolean array denoting where liquid attenuation was corrected.
        - ndarray: Boolean array denoting where liquid attenuation was present but not corrected.

    """
    melting_layer = utils.isbit(classification.category_bits, 3)
    uncorr_atten = np.cumsum(melting_layer, axis=1) >= 1
    uncorr_atten[classification.is_rain, :] = True
    corr_atten = (liq_atten > 0).filled(False) & ~uncorr_atten
    liq_atten[uncorr_atten] = ma.masked
    return liq_atten, corr_atten, uncorr_atten


def get_attenuations(model, mwr, classification, height):
    """Wrapper for attenuations."""
    gas = gas_atten(model.data_dense, classification.category_bits, height)
    liquid = liquid_atten(mwr.data, model.data_dense, classification, height)
    return {**gas, **liquid}
