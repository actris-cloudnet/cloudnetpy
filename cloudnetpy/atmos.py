""" This module contains functions to calculate
atmospheric parameters.
"""

import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy import lwc
from cloudnetpy import constants as con


def c2k(temp):
    """Converts Celsius to Kelvins."""
    return np.array(temp) + 273.15


def k2c(temp):
    """Converts Kelvins to Celsius."""
    return np.array(temp) - 273.15


def saturation_vapor_pressure(T, kind='accurate'):
    """Returns water vapour saturation pressure (over liquid).

    Args:
        T (ndarray): Temperature (K).
        kind (str, optional): Specifies the used method as a string
           ('accurate', 'fast'), where 'accurate' is the
           IAPWS-95 formulation and 'fast' is Vaisala's
           simpler approximation. Default is 'accurate'.

    Returns:
        Saturation water vapor pressure (Pa).

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
        X = (Tc/T) * (C[0]*v + C[1]*v**1.5 + C[2]*v**3 + C[3]*v**3.5 +
                      C[4]*v**4 + C[5]*v**7.5)
        return Pc * np.exp(X) * 100

    def _saturation_vapor_pressure_fast(T, A=6.116441, m=7.591386,
                                        Tn=240.7263):
        """Fast method for water vapour saturation.

        References:
            Vaisala's white paper: "Humidity conversion formulas".

        """
        Tc = k2c(T)
        return A * 10**((m*Tc) / (Tc+Tn)) * 100

    if kind == 'fast':
        Pws = _saturation_vapor_pressure_fast(T)
    elif kind == 'accurate':
        Pws = _saturation_vapor_pressure_accurate(T)
    return Pws


def dew_point(Pw, A=6.116441, m=7.591386, Tn=240.7263):
    """ Return dew point temperature.

    Args:
        Pw (ndarray): Water wapor pressure (Pa).
        A (float, optional): Parameter for Vaisala's empirical formula.
        m (float, optional): Parameter for Vaisala's empirical formula.
        Tn (float, optional): Parameter for Vaisala's empirical formula.

    Returns:
        Dew point temperature (K).

    Notes:
        Method taken from Vaisala white paper: "Humidity conversion formulas".

    """
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
        (ndarray): Wet bulb temperature (K).

    """
    def _get_derivatives(Pw, Tdew, m=17.269, Tn=35.86):
        a = m*(Tn - con.T0)
        b = Tdew - Tn
        Pw_d = -Pw*a/b**2
        Pw_dd = Pw*((a/b**2)**2 + 2*a/b**3)
        return Pw_d, Pw_dd

    rh[rh < 1e-5] = 1e-5  # rh cant be 0
    Pws = saturation_vapor_pressure(Tdry, kind='fast')
    Pw = Pws * rh
    Tdew = dew_point(Pw)
    Pw_d, Pw_dd = _get_derivatives(Pw, Tdew)
    F = p*1004 / (con.latent_heat*con.mw_ratio)
    A = Pw_dd/2
    B = Pw_d + F - Tdew*Pw_dd
    C = -Tdry*F - Tdew*Pw_d + 0.5*Tdew**2*Pw_dd
    return (-B + np.sqrt(B*B - 4*A*C)) / (2*A)


def get_gas_atten(model_i, cat_bits, height):
    """Returns gas attenuation (assumes saturation inside liquid droplets).

    Args:
        model_i: Dict containing interpolated model fields.
        cat_bits (ndarray): 2D array of integers containing categorize
            flag bits.

    Returns:
        Attenuation due to atmospheric gases.

    """
    dheight = utils.med_diff(height)
    droplet_bit = utils.bit_test(cat_bits, 1)
    ind = np.where(droplet_bit)
    gas_atten = np.zeros_like(droplet_bit, float)
    spec_gas_atten = np.copy(model_i['specific_gas_atten'])
    spec_gas_atten[ind] = model_i['specific_saturated_gas_atten'][ind]
    gas_atten[:, :] = model_i['gas_atten'][0, :]
    gas_atten[:, 1:] = gas_atten[:, 1:] + 2.0*np.cumsum(spec_gas_atten[:, :-1],
                                                        axis=1)*dheight*0.001
    return gas_atten


def get_liquid_atten(lwp, model, bits, height):
    """Calculates attenuation due to liquid water.

    Args:
        lwp: Dict containing interpolated liquid water
            path and its error {'value', 'err'}.
        model: Dict containing interpolated model fields.
        bits: Dict containing classification bits {'cat', 'rain'}.
        height (ndarray): Altitude vector.

    Returns:
        Dict containing liquid attenuation, its error
        and ..

    """
    spec_liq_att = model['specific_liquid_atten']
    droplet_bit = utils.bit_test(bits['cat'], 1)
    msize = droplet_bit.shape
    lwc_adiabatic, lwc_err = np.zeros(msize), np.zeros(msize)
    lwp_boxes, lwp_boxes_err = np.zeros(msize), np.zeros(msize)
    liq_atten, liq_atten_err = np.zeros(msize), np.zeros(msize)
    dheight = utils.med_diff(height) * 1000
    is_liquid = np.any(droplet_bit, axis=1)
    is_lwp = np.isfinite(lwp['value'])
    for ii in np.where(is_lwp & is_liquid)[0]:
        bases, tops = utils.bases_and_tops(droplet_bit[ii, :])
        for base, top in zip(bases, tops):
            npoints = top - base + 1
            ran = np.arange(npoints)
            idx = ran + base
            dlwc_dz = lwc.theory_adiabatic_lwc(model['temperature'][ii, base],
                                               model['pressure'][ii, base])
            lwc_adiabatic[ii, idx] = dlwc_dz * dheight * (ran+1)
            lwc_err[ii, idx] = dlwc_dz  # unnormalised
        lwp_boxes[ii, :] = (lwp['value'][ii] *
                            lwc_adiabatic[ii, :]/np.sum(lwc_adiabatic[ii, :]))
        lwp_boxes_err[ii, :] = (lwp['err'][ii] *
                                lwc_err[ii, :]/np.sum(lwc_err[ii, :]))
    for ii in np.where(~is_lwp)[0]:
        lwp_boxes[ii, droplet_bit[ii, :] == 1] = None
    liq_atten[:, 1:] = 0.002 * np.cumsum(lwp_boxes[:, :-1] *
                                         spec_liq_att[:, :-1], axis=1)
    liq_atten_err[:, 1:] = 0.002 * np.cumsum(lwp_boxes_err[:, :-1] *
                                             spec_liq_att[:, :-1], axis=1)
    liq_atten, cbit, ucbit = _screen_liq_atten(liq_atten, bits)
    return {'value': liq_atten, 'err': liq_atten_err,
            'corr_bit': cbit, 'ucorr_bit': ucbit}


def _screen_liq_atten(liq_atten, bits):
    """Remove corrupted data from liquid attenuation.

    Args:
        liq_atten (ndarray): Liquid attenuation.
        bits: Dict containing classification bits and rain
        {'cat', 'rain'}.

    Returns:
        3-element tuple containing screened liquid
        attenuation (MaskedArray), and bitfields
        showing where it was corrected and where it
        was not.

    """
    def _get_bits(liq_atten, rain_ind, melt_ind):
        corr_atten_bit = np.zeros_like(melt_ind, dtype=bool)
        uncorr_atten_bit = np.zeros_like(melt_ind, dtype=bool)
        corr_atten_bit[~liq_atten.mask] = True
        uncorr_atten_bit[rain_ind, :] = True
        uncorr_atten_bit[melt_ind] = True
        return corr_atten_bit, uncorr_atten_bit

    rain_ind = bits['rain'] == 1
    melt_bit = utils.bit_test(bits['cat'], 4)
    above_melt = np.cumsum(melt_bit == 1, axis=1)
    melt_ind = above_melt >= 1
    liq_atten[melt_ind] = None
    liq_atten[rain_ind, :] = None
    liq_atten = ma.masked_invalid(liq_atten)
    liq_atten = ma.masked_equal(liq_atten, 0)
    cbit, ucbit = _get_bits(liq_atten, rain_ind, melt_ind)
    return liq_atten, cbit, ucbit
