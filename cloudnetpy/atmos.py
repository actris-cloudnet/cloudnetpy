""" This module contains functions to calculate
atmospheric parameters.
"""

import sys
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    """ approximation of a liquid attenuation in a profile """
    #msize = temperature.shape
    #lwc_adiabatic = ma.zeros(msize)
    #lwc_error = ma.zeros(msize)
    #lwp_boxes = ma.zeros(msize)
    #lwp_boxes_error = ma.zeros(msize)
    #liquid_attenuation = ma.zeros(msize)
    #liquid_attenuation_error = ma.zeros(msize)
    #corr_atten_bit = np.zeros_like(droplet_bit)
    #uncorr_atten_bit = np.zeros_like(droplet_bit)
    droplet_bit = utils.bit_test(bits['cat_bits'], 1)
    is_liquid = np.any(droplet_bit, axis=1)
    is_lwp = np.isfinite(lwp['lwp'])
    for ii in np.where(is_lwp & is_liquid)[0]:
        bases, tops = utils.bases_and_tops(droplet_bit[ii, :])
        for base, top in zip(bases, tops):
            npoints = top - base + 1
            idx = np.arange(npoints) + base
            dlwc_dz = lwc.theory_adiabatic_lwc(model['model_i']['temperature'][ii, base],
                                               model['model_i']['pressure'][ii, base])

            #lwc_adiabatic[ii,idx] = dlwc_dz * dheight * 1000 * (np.arange(npoints)+1)
            #lwc_error[ii,idx] = dlwc_dz # unnormalised

        #lwp_boxes[ii,:] = lwp[ii] * lwc_adiabatic[ii,:] / np.sum(lwc_adiabatic[ii,:])
        #lwp_boxes_error[ii,:] = lwp_error[ii] * lwc_error[ii,:] / np.sum(lwc_error[ii,:])

    #for ii in np.where(~is_lwp)[0]:
    #    lwp_boxes[ii,droplet_bit[ii,:]==1] = None

    #liquid_attenuation[:,1:] = 0.002 * ma.cumsum(lwp_boxes[:,0:-1] * specific_liquid_atten[:,0:-1], axis=1)
    #liquid_attenuation_error[:,1:] = 0.002 * np.cumsum(lwp_boxes_error[:,0:-1] * specific_liquid_atten[:,0:-1], axis=1)

    #liquid_attenuation[rain_bit==1,:] = None
    #above_melting = np.cumsum(melting_bit>0,axis=1)
    #liquid_attenuation[above_melting>=1] = None    
    #liquid_attenuation = ma.masked_invalid(liquid_attenuation)
    #liquid_attenuation = ma.masked_equal(liquid_attenuation,0)
    # bit indicating attenuation that was corrected
    #corr_atten_bit[~liquid_attenuation.mask] = 1
    # bit indicating attenuation that was NOT corrected
    #uncorr_atten_bit[rain_bit==1,:] = 1
    #uncorr_atten_bit[above_melting>=1] = 1
    return 0
