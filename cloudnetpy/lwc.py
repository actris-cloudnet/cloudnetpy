"""Module for liquid water content calculations."""

import numpy as np
from cloudnetpy.constants import T0

def theory_adiabatic_lwc(T, P):
    """Returns adiabatic LWC change rate.

    Calculates the theoretical adiabatic rate of increase of LWC with
    height, or pressure, given the cloud base temperature and pressure,
    From Brenguier (1991).

    Args:
        T (ndarray): Temperature of cloud base (K).
        P (ndarray): Pressure of cloud base (Pa).

    Returns:
        dlwc/dz in kg m-3 m-1

    """
    e  = 0.62198  # ratio of the molecular weight of water vapor to dry air
    g  = -9.81  # acceleration due to gravity (m s-1)
    cp = 1005  # heat capacity of air at const pressure (J kg-1 K-1)
    L  = 2.5e6  # latent heat of evaporation (J kg-1)
    R  = 461.5 * e  # specific gas constant for dry air (J kg-1 K-1)
    drylapse = -g / cp  # dry lapse rate (K m-1)
    qs, es = temp2mixingratio(T, P)
    rhoa = P / (R * (1 + 0.6*qs) * T)
    dqldz = -(1 - (cp*T / (L*e))) * (1/((cp*T/(L*e)) + (L*qs*rhoa/(P-es))))*(rhoa*g*e*es)*((P-es)**(-2))
    dlwcdz = rhoa * dqldz
    return dlwcdz


def temp2mixingratio(T, P):
    """Converts temperature and pressure to mixing ratio."""
    t1 = T/T0
    t2 = 1 - (T0/T)
    svp = 10**(10.79574*(t2)-5.028*np.log10(t1) +
               1.50475e-4*(1-(10**(-8.2969*(t1-1)))) +
               0.42873e-3*(10**(4.76955*(t2))) +
               2.78614)
    mixing_ratio = 0.62198*svp/(P-svp)
    return mixing_ratio, svp
