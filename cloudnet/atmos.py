""" This module contains functions to calculate
atmospheric parameters.
"""

import numpy as np

# triple point of water
T0 = 273.16


def c2k(temp):
    """ Celsius to Kelvins conversion. """
    return np.array(temp) + 273.15


def k2c(temp):
    """ Kelvin to Celsius conversion. """
    return np.array(temp) - 273.15


def saturation_vapor_pressure(T, A=6.116441, m=7.591386, Tn=240.7263):
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
    return A * 10**((m*Tc) / (Tc+Tn)) * 100


def dew_point(Pw, A=6.116441, m=7.591386, Tn=240.7263):
    """ Calculates dew point temperature.

    Args:
        Pw (array_like): Water wapor pressure (Pa).

    Returns:
        Dew point temperature (K)

    Notes:
        Method taken from Vaisala white paper: "Humidity conversion formulas".

    """
    Td = Tn / ((m/np.log10(Pw/100/A))-1)
    return c2k(Td)


def wet_bulb(Tdry, p, rh):
    """ Calculates wet bulb temperature.

    This function returns wet bulb temperature for given temperature,
    pressure and relative humidity. Algorithm is based on a Taylor
    expansion of a simple expression for the saturated vapour pressure.

    Args:
        Tdry (array_like): Temperature (K).
        p (array_like): Pressure (Pa).
        rh (array_like): Relative humidity (0-1).

    Returns:
        (array_like): Wet bulb temperature (K).

    """
    def _get_derivatives(Pw, Tdew, m=17.269, Tn=35.86):
        a = m*(Tn - T0)
        b = Tdew - Tn
        Pw_d = -Pw*a/b**2
        Pw_dd = Pw*((a/b**2)**2 + 2*a/b**3)
        return Pw_d, Pw_dd

    rh[rh < 1e-12] = 1e-12  # rh = 0 causes trouble
    Pws = saturation_vapor_pressure(Tdry)
    Pw = Pws * rh
    Tdew = dew_point(Pw)
    Pw_d, Pw_dd = _get_derivatives(Pw, Tdew)
    F = p*1004 / (2.5e6*0.622)
    A = Pw_dd/2
    B = Pw_d + F - Tdew*Pw_dd
    C = -Tdry*F - Tdew*Pw_d + 0.5*Tdew**2*Pw_dd
    return (-B + np.sqrt(B*B - 4*A*C)) / (2*A)
