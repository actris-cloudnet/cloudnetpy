""" This module contains functions to calculate
various atmospheric parameters.
"""
import numpy as np
import numpy.ma as ma
from cloudnetpy import constants as con
from cloudnetpy import utils


def c2k(temp):
    """Converts Celsius to Kelvins."""
    return ma.array(temp) + 273.15


def k2c(temp):
    """Converts Kelvins to Celsius."""
    return ma.array(temp) - 273.15


VAISALA_PARAMS_OVER_WATER = (6.116441, 7.591386, 240.7263)
HPA_TO_P = 100
P_TO_HPA = 0.01


def calc_lwc_change_rate(temperature, pressure):
    """Returns adiabatic LWC change rate.

    Calculates the theoretical adiabatic rate of increase of LWC with
    height given the cloud base temperature and pressure.

    Args:
        temperature (ndarray): Temperature of cloud base (K).
        pressure (ndarray): Pressure of cloud base (Pa).

    Returns:
        ndarray: dlwc/dz (kg m-3 m-1)

    References:
        Brenguier, 1991, https://bit.ly/2QCSJtb

    """

    svp = calc_saturation_vapor_pressure(temperature)
    svp_mixing_ratio = calc_mixing_ratio(svp, pressure)

    air_density = pressure / (con.Rs*temperature*(0.6*svp_mixing_ratio + 1))
    a = con.specific_heat*temperature / (con.latent_heat*con.mw_ratio)
    b = pressure - svp
    f1 = a - 1
    f2 = 1 / (a + (con.latent_heat*svp_mixing_ratio*air_density/b))
    f3 = air_density*con.g*con.mw_ratio*svp*b**-2
    return air_density*f1*f2*f3


def calc_mixing_ratio(svp, pressure):
    """Calculates mixing ratio from saturation vapor pressure and pressure.

    Args:
        svp (ndarray): Saturation vapor pressure (Pa).
        pressure (ndarray): Atmospheric pressure (Pa).

    Returns:
        ndarray: Mixing ratio.

    """
    return con.mw_ratio*svp/(pressure-svp)


def calc_saturation_vapor_pressure(temp_kelvin):
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
    return a * 10**((m*temp_celsius)/(temp_celsius+tn)) * HPA_TO_P


def calc_dew_point_temperature(vapor_pressure):
    """ Returns dew point temperature.

    Args:
        vapor_pressure (ndarray): Water vapor pressure (Pa).

    Returns:
        ndarray: Dew point temperature (K).

    Notes:
        Method from Vaisala's white paper: "Humidity conversion formulas".

    """
    a, m, tn = VAISALA_PARAMS_OVER_WATER
    dew_point_celsius = tn / ((m/np.log10(vapor_pressure*P_TO_HPA/a))-1)
    return c2k(dew_point_celsius)


def calc_psychrometric_constant(pressure):
    """Returns psychrometric constant.

    Psychrometric constant relates the partial pressure
    of water in air to the air temperature.

    Args:
        pressure (ndarray): Atmospheric pressure (Pa).

    Returns:
        ndarray: Psychrometric constant value (Pa C-1)

    """
    return pressure*con.specific_heat / (con.latent_heat * con.mw_ratio)


def calc_wet_bulb_temperature(model_data):
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

    relative_humidity = _screen_rh()
    saturation_pressure = calc_saturation_vapor_pressure(model_data['temperature'])
    vapor_pressure = saturation_pressure * relative_humidity
    dew_point = calc_dew_point_temperature(vapor_pressure)
    psychrometric_constant = calc_psychrometric_constant(model_data['pressure'])
    first_der, second_der = _vapor_derivatives()
    a = 0.5*second_der
    b = first_der + psychrometric_constant - dew_point*second_der
    c = (-model_data['temperature']*psychrometric_constant
         - dew_point*first_der
         + 0.5*dew_point**2*second_der)
    return (-b+np.sqrt(b*b-4*a*c))/(2*a)


def get_attenuations(model, mwr, classification):
    """Wrapper for attenuations."""
    gas = GasAttenuation(model, classification)
    liquid = LiquidAttenuation(model, classification, mwr)
    return {'radar_gas_atten': gas.atten,
            'radar_liquid_atten': liquid.atten,
            'liquid_atten_err': liquid.atten_err,
            'liquid_corrected': liquid.corrected,
            'liquid_uncorrected': liquid.uncorrected
            }


class Attenuation:
    """Base class for gas and liquid attenuations."""
    def __init__(self, model, classification):
        self._dheight = utils.mdiff(model.height)
        self._model = model.data_dense
        self._liquid_in_pixel = utils.isbit(classification.category_bits, 0)
        self.classification = classification


class GasAttenuation(Attenuation):
    """Radar gas attenuation class."""
    def __init__(self, model, classification):
        super().__init__(model, classification)
        self.atten = self._calc_gas_atten()

    def _calc_gas_atten(self):
        atten = self._init_gas_atten()
        self._fix_atten_in_liquid(atten)
        return self._specific_to_gas_atten(atten)

    def _init_gas_atten(self):
        return np.copy(self._model['specific_gas_atten'][:])

    def _fix_atten_in_liquid(self, atten):
        saturated_atten = self._model['specific_saturated_gas_atten'][:]
        atten[self._liquid_in_pixel] = saturated_atten[self._liquid_in_pixel]

    def _specific_to_gas_atten(self, atten):
        layer1_atten = self._model['gas_atten'][:][:, 0]
        atten_cumsum = np.cumsum(atten.T, axis=0)
        atten = 2 * atten_cumsum * self._dheight * 1e-3 + layer1_atten
        atten = np.insert(atten.T, 0, layer1_atten, axis=1)[:, :-1]
        return atten


class LiquidAttenuation(Attenuation):
    """Radar liquid attenuation class."""
    def __init__(self, model, classification, mwr):
        super().__init__(model, classification)
        self._mwr = mwr.data
        self._lwc_dz_err = self._get_lwc_change_rate_error()
        self.atten = self.get_liquid_atten()
        self.atten_err = self._get_liquid_atten_err()
        self.corrected, self.uncorrected = self._screen_attenuations()

    def _get_lwc_change_rate_error(self):
        """Fills cloud pixels with the LWC change rate of base."""
        lwc_dz = self._get_lwc_change_rate()
        lwc_dz_err = ma.zeros(lwc_dz.shape)
        lwc_dz_err[self._liquid_in_pixel] = utils.ffill(lwc_dz[self._liquid_in_pixel])
        return lwc_dz_err

    def _get_lwc_change_rate(self):
        """Finds LWC change rate in liquid cloud bases."""
        liquid_bases = self.classification.liquid_bases
        lwc_dz = ma.zeros(self._liquid_in_pixel.shape)
        temperature = self._model['temperature'][liquid_bases]
        pressure = self._model['pressure'][liquid_bases]
        lwc_dz[liquid_bases] = calc_lwc_change_rate(temperature, pressure)
        return lwc_dz

    def get_liquid_atten(self):
        """Finds radar liquid attenuation."""
        def _get_lwc_adiabatic():
            ind_from_base = utils.cumsumr(self._liquid_in_pixel, axis=1)
            return ind_from_base * self._lwc_dz_err * self._dheight * 1e3

        def _get_lwp_normalized():
            lwc = _get_lwc_adiabatic()
            mwr = self._mwr['lwp']
            return self._normalize_lwp(lwc, mwr)

        lwp_norm = _get_lwp_normalized()
        return self._calc_attenuation(lwp_norm)

    def _get_liquid_atten_err(self):
        """Finds radar liquid attenuation error."""
        def _get_lwp_normalized():
            lwc = self._lwc_dz_err
            mwr = self._mwr['lwp_error']
            return self._normalize_lwp(lwc, mwr)

        lwp_norm = _get_lwp_normalized()
        return self._calc_attenuation(lwp_norm)

    def _normalize_lwp(self, lwc_var, mwr_var):
        """Normalizes measured LWP with model LWC."""
        def _get_lwp_ind():
            mwr_lwp = self._mwr['lwp'][:]
            return np.isfinite(mwr_lwp) & np.any(self._liquid_in_pixel, axis=1)

        lwp_norm = ma.zeros(self._liquid_in_pixel.shape, dtype=float)
        lwp_and_liquid = _get_lwp_ind()
        mwr = mwr_var[lwp_and_liquid]
        lwc = lwc_var[lwp_and_liquid, :]
        lwc_sum = np.sum(lwc, axis=1)
        lwp_norm[lwp_and_liquid] = (lwc.T*mwr/lwc_sum).T
        return lwp_norm

    def _calc_attenuation(self, lwp_norm):
        liq_att = ma.zeros(self._liquid_in_pixel.shape, dtype=float)
        spec_liq = self._model['specific_liquid_atten']
        lwp_cumsum = np.cumsum(lwp_norm[:, :-1]*spec_liq[:, :-1], axis=1)
        liq_att[:, 1:] = 2e-3 * lwp_cumsum
        return liq_att

    def _screen_attenuations(self):
        melting_layer = utils.isbit(self.classification.category_bits, 3)
        uncorrected = np.cumsum(melting_layer, axis=1) >= 1
        uncorrected[self.classification.is_rain, :] = True
        corrected = (self.atten > 0).filled(False) & ~uncorrected
        self.atten[uncorrected] = ma.masked
        return corrected, uncorrected

