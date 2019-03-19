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
KM_TO_M = 0.001
KG_TO_G = 1000
TWO_WAY = 2


def calc_lwc_change_rate(temperature, pressure):
    """Returns rate of change of condensable water (LWC).

    Calculates the theoretical adiabatic rate of increase of LWC
    with height, given the cloud base temperature and pressure.

    Args:
        temperature (ndarray): Temperature of cloud base (K).
        pressure (ndarray): Pressure of cloud base (Pa).

    Returns:
        ndarray: dlwc/dz (g m-3 m-1)

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
    return air_density*f1*f2*f3 * KG_TO_G


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
    return pressure*con.specific_heat / (con.latent_heat*con.mw_ratio)


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
            'liquid_uncorrected': liquid.uncorrected}


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
        return np.copy(self._model['specific_gas_atten'])

    def _fix_atten_in_liquid(self, atten):
        saturated_atten = self._model['specific_saturated_gas_atten']
        atten[self._liquid_in_pixel] = saturated_atten[self._liquid_in_pixel]

    def _specific_to_gas_atten(self, atten):
        layer1_atten = self._model['gas_atten'][:, 0]
        atten_cumsum = np.cumsum(atten.T, axis=0)
        atten = TWO_WAY * atten_cumsum * self._dheight * 1e-3 + layer1_atten
        atten = np.insert(atten.T, 0, layer1_atten, axis=1)[:, :-1]
        return atten


class LiquidAttenuation(Attenuation):
    """Radar liquid attenuation class."""
    def __init__(self, model, classification, mwr):
        super().__init__(model, classification)
        self._mwr = mwr.data
        self._lwc_dz_err = self._get_lwc_change_rate_error()
        self.atten = self._get_liquid_atten()
        self.atten_err = self._get_liquid_atten_err()
        self.corrected, self.uncorrected = self._screen_attenuations()

    def _get_lwc_change_rate_error(self):
        atmosphere = (self._model['temperature'], self._model['pressure'])
        return fill_clouds_with_lwc_dz(atmosphere, self._liquid_in_pixel)

    def _get_liquid_atten(self):
        """Finds radar liquid attenuation."""
        lwc = calc_adiabatic_lwc(self._lwc_dz_err,
                                 self._liquid_in_pixel,
                                 self._dheight)
        lwc_scaled = scale_lwc(lwc, self._mwr['lwp'][:])
        return self._calc_attenuation(lwc_scaled)

    def _get_liquid_atten_err(self):
        """Finds radar liquid attenuation error."""
        lwc_err_scaled = scale_lwc(self._lwc_dz_err, self._mwr['lwp_error'][:])
        return self._calc_attenuation(lwc_err_scaled)

    def _calc_attenuation(self, lwc_norm):
        """Finds liquid attenuation (dB)."""
        liq_att = ma.zeros(self._liquid_in_pixel.shape, dtype=float)
        spec_liq = self._model['specific_liquid_atten']
        lwp_cumsum = np.cumsum(lwc_norm[:, :-1] * spec_liq[:, :-1], axis=1)
        liq_att[:, 1:] = TWO_WAY * KM_TO_M * lwp_cumsum
        return liq_att

    def _screen_attenuations(self):
        melting_layer = utils.isbit(self.classification.category_bits, 3)
        uncorrected = np.cumsum(melting_layer, axis=1) >= 1
        uncorrected[self.classification.is_rain, :] = True
        corrected = (self.atten > 0).filled(False) & ~uncorrected
        self.atten[uncorrected] = ma.masked
        return corrected, uncorrected


def scale_lwc(lwc, lwp):
    """Scales theoretical liquid water content to match the measured LWP.

    Args:
        lwc (ndarray): 2D liquid water content (g/m3).
        lwp (ndarray): 1D liquid water path (g/m2).

    Returns:
        ndarray: Scaled liquid water content.

    """
    lwc_sum = np.sum(lwc, axis=1)
    lwc_scaled = (lwc.T/lwc_sum*lwp).T
    return lwc_scaled


def get_lwc_change_rate_at_bases(atmosphere, is_liquid):
    """Finds LWC change rate in liquid cloud bases.

    Args:
        atmosphere (tuple): 2-element tuple containing temperature (K) and
            pressure (Pa).
        is_liquid (ndarray): Boolean array indicating presence of liquid clouds.

    Returns:
        liquid water content change rate at cloud bases (kg/m3/m).

    """
    liquid_bases = utils.find_cloud_bases(is_liquid)
    lwc_dz = ma.zeros(liquid_bases.shape)
    lwc_dz[liquid_bases] = calc_lwc_change_rate(atmosphere[0][liquid_bases],
                                                atmosphere[1][liquid_bases])
    return lwc_dz


def fill_clouds_with_lwc_dz(atmosphere, is_liquid):
    """Fills liquid clouds with lwc change rate at the cloud bases.

    Args:
        atmosphere (tuple): 2-element tuple containing temperature (K) and pressure (Pa).
        is_liquid (ndarray): Boolean array indicating presence of liquid clouds.

    Returns:
        liquid water content change rate (g/m3/m), so that for each cloud
            the base value is filled for the whole cloud.

    """
    lwc_dz = get_lwc_change_rate_at_bases(atmosphere, is_liquid)
    lwc_dz_filled = ma.zeros(lwc_dz.shape)
    lwc_dz_filled[is_liquid] = utils.ffill(lwc_dz[is_liquid])
    return lwc_dz_filled


def calc_adiabatic_lwc(lwc_change_rate, is_liquid, dheight):
    """Calculates adiabatic liquid water content.

    Args:
        lwc_change_rate (ndarray): Liquid water content change rate (g/m3/m)
            calculated at the base of each cloud and then filled to that cloud.
        is_liquid (ndarray): Boolean array indicating presense of liquid clouds.
        dheight: Median difference of the height vector (m).

    Returns:
        Liquid water content (g/m3).

    """
    ind_from_base = utils.cumsumr(is_liquid, axis=1)
    lwc = ind_from_base * dheight * lwc_change_rate
    return lwc
