""" This module contains functions to calculate
various atmospheric parameters.
"""

import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy import lwc
from cloudnetpy import constants as con
from cloudnetpy.cloudnetarray import CloudnetArray

def c2k(temp):
    """Converts Celsius to Kelvins."""
    return np.array(temp) + 273.15


def k2c(temp):
    """Converts Kelvins to Celsius."""
    return np.array(temp) - 273.15


def _vaisala_params():
    """Returns parameters for Vaisala's empirical formulas."""
    A = 6.116441
    m = 7.591386
    Tn = 240.7263
    return A, m, Tn


def saturation_vapor_pressure(T, kind='accurate'):
    """Returns water vapour saturation pressure (over liquid).

    Args:
        T (ndarray): Temperature (K).
        kind (str, optional): Specifies the used method as a string
           ('accurate', 'fast'), where 'accurate' is the
           IAPWS-95 formulation and 'fast' is Vaisala's
           simpler approximation. Default is 'accurate'.

    Returns:
        ndarray: Saturation water vapor pressure (Pa).

    """
    def _saturation_vapor_pressure_accurate(T):
        """Accurate method for water vapour saturation pressure.

        References:
            The IAPWS Formulation 1995 for the Thermodynamic
            Properties of Ordinary Water Substance for General and Scientific
            Use, Journal of Physical and Chemical Reference Data, June 2002 ,
            Volume 31, Issue 2, pp. 387535.

        """
        Tc = 647.096
        Pc = 220640  # hPa
        C = [-7.85951783, 1.84408259, -11.7866497,
             22.6807411, -15.9618719, 1.80122502]
        v = 1 - T/Tc
        X = (Tc/T) * (C[0]*v
                      + C[1]*v**1.5
                      + C[2]*v**3
                      + C[3]*v**3.5
                      + C[4]*v**4
                      + C[5]*v**7.5)
        return Pc * np.exp(X) * 100

    def _saturation_vapor_pressure_fast(T):
        """Fast method for water vapour saturation.

        Notes:
            Method from Vaisala's white paper: "Humidity conversion formulas".

        """
        A, m, Tn = _vaisala_params()
        Tc = k2c(T)
        return A * 10**((m*Tc) / (Tc+Tn)) * 100

    if kind == 'fast':
        Pws = _saturation_vapor_pressure_fast(T)
    elif kind == 'accurate':
        Pws = _saturation_vapor_pressure_accurate(T)
    return Pws


def dew_point(Pw):
    """ Returns dew point temperature.

    Args:
        Pw (ndarray): Water wapor pressure (Pa).

    Returns:
        ndarray: Dew point temperature (K).

    Notes:
        Method from Vaisala's white paper: "Humidity conversion formulas".

    """
    A, m, Tn = _vaisala_params()
    Td = Tn / ((m/np.log10(Pw/100/A))-1)
    return c2k(Td)


def wet_bulb(Tdry, p, rh):
    """Returns wet bulb temperature.

    Returns wet bulb temperature for given temperature,
    pressure and relative humidity. Algorithm is based on a Taylor
    expansion of a simple expression for the saturated vapour pressure.

    Args:
        Tdry (ndarray): Temperature (K).
        p (ndarray): Pressure (Pa).
        rh (ndarray): Relative humidity (0-1).

    Returns:
        ndarray: Wet bulb temperature (K).

    """
    def _derivatives(Pw, Tdew, m=17.269, Tn=35.86):
        a = m*(Tn - con.T0)
        b = Tdew - Tn
        Pw_d = -Pw*a/b**2
        Pw_dd = Pw*((a/b**2)**2 + 2*a/b**3)
        return Pw_d, Pw_dd

    rh[rh < 1e-5] = 1e-5  # rh cant be 0
    Pws = saturation_vapor_pressure(Tdry, kind='fast')
    Pw = Pws * rh
    Tdew = dew_point(Pw)
    Pw_d, Pw_dd = _derivatives(Pw, Tdew)
    F = p*1004 / (con.latent_heat*con.mw_ratio)
    A = Pw_dd/2
    B = Pw_d + F - Tdew*Pw_dd
    C = -Tdry*F - Tdew*Pw_d + 0.5*Tdew**2*Pw_dd
    return (-B + np.sqrt(B*B - 4*A*C)) / (2*A)


def gas_atten(model, cat_bits, height):
    """Returns gas attenuation (assumes saturation inside liquid droplets).

    Args:
        model (dict): Interpolated 2-D model fields {'gas_atten',
            'specific_gas_atten', 'specific_saturated_gas_atten'}.
        cat_bits (ndarray): 2-D array of integers containing
            categorize flag bits.
        height (ndarray): 1-D altitude grid (m).

    Returns:
        CloudnetArray: Attenuation due to atmospheric gases.

    """
    dheight = utils.mdiff(height)
    is_liquid = utils.isbit(cat_bits, 0)
    spec = np.copy(model['specific_gas_atten'][:])
    spec[is_liquid] = model['specific_saturated_gas_atten'][:][is_liquid]
    layer1_att = model['gas_atten'][:][:, 0]
    gas_att = 2*np.cumsum(spec.T, axis=0)*dheight*1e-3 + layer1_att
    gas_att = np.insert(gas_att.T, 0, layer1_att, axis=1)[:, :-1]
    return {'radar_gas_atten': CloudnetArray(gas_att, 'radar_gas_atten')}


def liquid_atten(mwr, model, bits, liquid_bases, is_rain, height):
    """Calculates attenuation due to liquid water.

    Args:
        lwp (dict): Interpolated liquid water
            path and its error {'value', 'err'}.
        model (dict): Interpolated 2-D model fields {'temperature', 'pressure',
            'specific_liquid_atten'}
        bits (ndarray): Classification bits.
        height (ndarray): 1-D altitude grid (m).

    Returns:
        Dict containing

        - **value** (*MaskedArray*): Amount of liquid attenuation.
        - **err** (*MaskedArray*): Error in the liquid attenuation.
        - **is_corr** (*ndarray*): Boolean array denoting where liquid
          attenuation is present and we can compute its value.
        - **is_not_corr** (*ndarray*): Boolean array denoting where liquid
          attenuation is present but we can not compute its value.
 
    """
    spec_liq = model['specific_liquid_atten'][:]
    is_liq = utils.isbit(bits, 0)
    lwc_dz, lwc_dz_err, liq_att, liq_att_err, lwp_norm, lwp_norm_err = utils.init(6, is_liq.shape)
    ind = np.where(liquid_bases)
    lwc_dz[ind] = lwc.adiabatic_lwc(model['temperature'][:][ind],
                                    model['pressure'][:][ind])
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
    liq_att, cbit, ucbit = _screen_liq_atten(liq_att, bits, is_rain)
    return ({'radar_liquid_atten': CloudnetArray(liq_att, 'radar_liquid_atten')},
            {'liq_att_err': liq_att_err, 'is_corr': cbit, 'is_not_corr': ucbit})


def _screen_liq_atten(liq_atten, bits, is_rain):
    """Removes corrupted data from liquid attenuation.

    Args:
        liq_atten (ndarray): Liquid attenuation.
        bits: Dict containing classification bits and rain
            {'cat', 'rain'}.

    Returns:
        tuple: 3-element tuple containing:
        
        - MaskedArray: Screened liquid attenuation.
        - ndarray: Boolean array denoting where liquid attenuation was corrected.
        - ndarray: Boolean array denoting where liquid attenuation was present but not corrected.

    """
    melting_layer = utils.isbit(bits, 3)
    uncorr_atten = np.cumsum(melting_layer, axis=1) >= 1
    uncorr_atten[is_rain, :] = True
    corr_atten = (liq_atten > 0).filled(False) & ~uncorr_atten
    liq_atten[uncorr_atten] = ma.masked
    return liq_atten, corr_atten, uncorr_atten
