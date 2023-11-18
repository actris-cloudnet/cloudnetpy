from typing import Final

import numpy as np
from numpy import ma

import cloudnetpy.constants as con

HPA_TO_P: Final = 100
P_TO_HPA: Final = 0.01


def calc_wet_bulb_temperature(model_data: dict) -> np.ndarray:
    """Returns wet bulb temperature.

    Returns wet bulb temperature for given temperature,
    pressure and relative humidity. Algorithm is based on a Taylor
    expansion of a simple expression for the saturated vapour pressure.

    Args:
    ----
        model_data: Model variables `temperature`, `pressure`, `rh`.

    Returns:
    -------
        Wet bulb temperature (K).

    References:
    ----------
        J. Sullivan and L. D. Sanders: Method for obtaining wet-bulb
        temperatures by modifying the psychrometric formula.

    """

    def _screen_rh() -> np.ndarray:
        rh = model_data["rh"]
        rh_min = 1e-5
        rh[rh < rh_min] = rh_min
        return rh

    def _vapor_derivatives() -> tuple:
        m = 17.269
        tn = 35.86
        f1 = m * (tn - con.T0)
        f2 = dew_point - tn
        first = -vapor_pressure * f1 / (f2**2)
        second = vapor_pressure * ((f1 / (f2**2)) ** 2 + 2 * f1 / (f2**3))
        return first, second

    relative_humidity = _screen_rh()
    saturation_pressure = calc_saturation_vapor_pressure(model_data["temperature"])
    vapor_pressure = saturation_pressure * relative_humidity
    dew_point = calc_dew_point_temperature(vapor_pressure)
    psychrometric_constant = calc_psychrometric_constant(model_data["pressure"])
    first_der, second_der = _vapor_derivatives()
    a = 0.5 * second_der
    b = first_der + psychrometric_constant - dew_point * second_der
    c = (
        -model_data["temperature"] * psychrometric_constant
        - dew_point * first_der
        + 0.5 * dew_point**2 * second_der
    )
    return (-b + ma.sqrt(b * b - 4 * a * c)) / (2 * a)


def calc_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Goff-Gratch formula for saturation vapor pressure over water adopted by WMO.

    Args:
    ----
        temperature: Temperature (K).

    Returns:
    -------
        Saturation vapor pressure (Pa).

    """
    ratio = con.T0 / temperature
    inv_ratio = ratio**-1
    return (
        10
        ** (
            10.79574 * (1 - ratio)
            - 5.028 * np.log10(inv_ratio)
            + 1.50475e-4 * (1 - (10 ** (-8.2969 * (inv_ratio - 1))))
            + 0.42873e-3 * (10 ** (4.76955 * (1 - ratio)) - 1)
            + 0.78614
        )
    ) * HPA_TO_P


def calc_psychrometric_constant(pressure: np.ndarray) -> np.ndarray:
    """Returns psychrometric constant.

    Psychrometric constant relates the partial pressure
    of water in air to the air temperature.

    Args:
    ----
        pressure: Atmospheric pressure (Pa).

    Returns:
    -------
        Psychrometric constant value (Pa K-1)

    """
    return pressure * con.SPECIFIC_HEAT / (con.LATENT_HEAT * con.MW_RATIO)


def calc_dew_point_temperature(vapor_pressure: np.ndarray) -> np.ndarray:
    """Returns dew point temperature.

    Args:
    ----
        vapor_pressure: Water vapor pressure (Pa).

    Returns:
    -------
        Dew point temperature (K).

    Notes:
    -----
        Method from Vaisala's white paper: "Humidity conversion formulas".

    """
    vaisala_parameters_over_water = (6.116441, 7.591386, 240.7263)
    a, m, tn = vaisala_parameters_over_water
    dew_point_celsius = tn / ((m / np.log10(vapor_pressure * P_TO_HPA / a)) - 1)
    return c2k(dew_point_celsius)


def c2k(temp: np.ndarray) -> np.ndarray:
    """Converts Celsius to Kelvins."""
    return ma.array(temp) + 273.15


def k2c(temp: np.ndarray) -> np.ndarray:
    """Converts Kelvins to Celsius."""
    return ma.array(temp) - 273.15


def mmh2ms(data: np.ndarray) -> np.ndarray:
    """Converts mm h-1 to m s-1"""
    return data / con.SEC_IN_HOUR * con.MM_TO_M
