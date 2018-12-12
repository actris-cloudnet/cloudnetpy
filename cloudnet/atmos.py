""" This module contains functions to calculate
atmospheric parameters.
"""

import numpy as np
import constants as con

T0 = 273.16  # triple point of water


def c2k(temp):
    """ Celsius to Kelvins conversion. """
    return np.array(temp) + 273.15


def k2c(temp):
    """ Kelvin to Celsius conversion. """
    return np.array(temp) - 273.15


def saturation_vapor_pressure(T):
    """ Calculates water vapour saturation pressure (over liquid).

    Args:
        T (array_like): Temperature (K).

    Returns:
        Saturation water vapor pressure (Pa).

    Notes:
        Method taken from Vaisala white paper: "Humidity conversion formulas".
        This is the simpler of the two methods described in Vaisala's paper.

    """
    Tc = k2c(T)
    return con.A * 10**((con.m*Tc) / (Tc+con.Tn)) * 100


def dew_point(P_w):
    """ Calculates dew point temperature.

    Args:
        P_w (array_like): Water wapor pressure (Pa).

    Returns:
        Dew point temperature (K)

    Notes:
        Method taken from Vaisala white paper: "Humidity conversion formulas".

    """
    T_d = con.Tn / ((con.m/np.log10(P_w/100/con.A))-1)
    return c2k(T_d)


def wet_bulb(T_dry, p, rh):
    """ Calculates wet bulb temperature.

    This function returns wet bulb temperature for given temperature,
    pressure and relative humidity. Algorithm is ased on a Taylor
    expansion of a simple expression for the saturated vapour pressure.

    Args:
        T_dry (array_like): Temperature (K).
        p (array_like): Pressure (Pa).
        rh (array_like): Relative humidity (0-1).

    Returns:
        (array_like): Wet bulb temperature (K).

    """
    rh[rh < 1e-12] = 1e-12  # rh = 0 causes trouble
    P_ws = saturation_vapor_pressure(T_dry)
    P_w = P_ws * rh
    T_dew = dew_point(P_w)
    a, c = 17.269, 35.86
    Lv, Cp, epsilon = 2.5e6, 1004, 0.622
    numerator = a*(c-T0)
    denominator = T_dew-c
    e_dash = P_w*(-numerator/denominator**2)
    e_dash_dash = P_w*((numerator/denominator**2)**2 +
                       2*numerator/denominator**3)
    f = p*Cp / (Lv*epsilon)
    A = e_dash_dash/2
    B = e_dash + f - T_dew*e_dash_dash
    C = -T_dry*f - T_dew*e_dash + 0.5*T_dew**2*e_dash_dash
    return (-B + np.sqrt(B*B - 4*A*C)) / (2*A)
