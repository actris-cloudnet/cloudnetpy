""" This module contains functions to calculate
atmospheric parameters.
"""

import numpy as np

# triple point of water
T0 = 273.16


def c2k(temp):
    """ Convert Celsius to Kelvins. """
    return np.array(temp) + 273.15


def k2c(temp):
    """ Convert Kelvin to Celsius. """
    return np.array(temp) - 273.15


def saturation_vapor_pressure(T, kind='accurate'):
    """ Return water vapour saturation pressure (over liquid).

    Args:
        T (array_like): Temperature (K).
        kind (str, optional): Specifies the used method as a string
           ('accurate', 'fast'), where 'accurate' is the
           IAPWS-95 formulation and 'fast' is Vaisala's
           simpler approximation. Default is 'accurate'.
    
    Returns:
        Saturation water vapor pressure (Pa).

    """

    if kind == 'fast':
        Pws = _saturation_vapor_pressure_fast(T)
    elif kind == 'accurate':
        Pws = _saturation_vapor_pressure_accurate(T)
    return Pws
        
def _saturation_vapor_pressure_accurate(T):
    """ Return water vapour saturation pressure (over liquid).

    Args:
        T (array_like): Temperature (K).

    Returns:
        Saturation water vapor pressure (Pa).

    Notes:
        Reference: "The IAPWS Formulation 1995 for the Thermodynamic 
        Properties of Ordinary Water Substance for General and Scientific Use ",
        Journal of Physical and Chemical Reference Data, June 2002 ,Volume 31, 
        Issue 2, pp. 387535".
    """
    Tc = 647.096
    Pc = 220640  # hPa
    C = [-7.85951783, 1.84408259, -11.7866497,
         22.6807411, -15.9618719, 1.80122502]
    v = 1 - T/Tc
    X = (Tc/T) * (C[0]*v + C[1]*v**1.5 + C[2]*v**3 + C[3]*v**3.5 +
                  C[4]*v**4 + C[5]*v**7.5)
    return Pc * np.exp(X) * 100


def _saturation_vapor_pressure_fast(T, A=6.116441, m=7.591386, Tn=240.7263):
    """ Return water vapour saturation pressure (over liquid).

    Args:
        T (array_like): Temperature (K).
        A (float, optional): Parameter for Vaisala's empirical formula.
        m (float, optional): Parameter for Vaisala's empirical formula.
        Tn (float, optional): Parameter for Vaisala's empirical formula.

    Returns:
        Saturation water vapor pressure (Pa).

    Notes:
        Method taken from Vaisala's white paper: "Humidity conversion formulas".

    """
    Tc = k2c(T)
    return A * 10**((m*Tc) / (Tc+Tn)) * 100


def dew_point(Pw, A=6.116441, m=7.591386, Tn=240.7263):
    """ Return dew point temperature.

    Args:
        Pw (array_like): Water wapor pressure (Pa).
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
    """ Return wet bulb temperature.

    Return wet bulb temperature for given temperature,
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
    Pws = saturation_vapor_pressure(Tdry, kind='fast')
    Pw = Pws * rh
    Tdew = dew_point(Pw)
    Pw_d, Pw_dd = _get_derivatives(Pw, Tdew)
    F = p*1004 / (2.5e6*0.622)
    A = Pw_dd/2
    B = Pw_d + F - Tdew*Pw_dd
    C = -Tdry*F - Tdew*Pw_d + 0.5*Tdew**2*Pw_dd
    return (-B + np.sqrt(B*B - 4*A*C)) / (2*A)
