""" This module contains functions to calculate
various atmospheric parameters.
"""
import numpy as np
import numpy.ma as ma
import scipy.constants
from cloudnetpy import constants as con
from cloudnetpy import utils


HPA_TO_P = 100
P_TO_HPA = 0.01
M_TO_KM = 0.001
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
    air_density = calc_air_density(pressure, temperature, svp_mixing_ratio)
    kelvin_per_kg = calc_psychrometric_constant(temperature)
    pressure_difference = pressure - svp
    f1 = kelvin_per_kg - 1
    f2 = 1 / (kelvin_per_kg + (con.LATENT_HEAT * svp_mixing_ratio
                               * air_density / pressure_difference))
    f3 = con.MW_RATIO * svp * pressure_difference ** -2
    dqs_dp = f1 * f2 * f3
    dqs_dz = dqs_dp * air_density**2 * -scipy.constants.g
    return dqs_dz * KG_TO_G


def calc_saturation_vapor_pressure(temperature):
    """Goff-Gratch formula for saturation vapor pressure over water adopted by WMO.

    Args:
        temperature (ndarray): Temperature (K).

    Returns:
        ndarray: Saturation vapor pressure (Pa).

    """
    ratio = con.T0 / temperature
    inv_ratio = ratio**-1
    return (10 ** (10.79574 * (1-ratio)
                   - 5.028 * np.log10(inv_ratio)
                   + 1.50475e-4 * (1 - (10 ** (-8.2969 * (inv_ratio-1))))
                   + 0.42873e-3 * (10 ** (4.76955 * (1-ratio)) - 1)
                   + 0.78614)) * HPA_TO_P


def calc_mixing_ratio(svp, pressure):
    """Calculates mixing ratio from saturation vapor pressure and pressure.

    Args:
        svp (ndarray): Saturation vapor pressure (Pa).
        pressure (ndarray): Atmospheric pressure (Pa).

    Returns:
        ndarray: Mixing ratio (kg kg-1).

    """
    return con.MW_RATIO * svp / (pressure - svp)


def calc_air_density(pressure, temperature, svp_mixing_ratio):
    """Calculates air density (kg m-3).

    Args:
        pressure (ndarray): Pressure (Pa).
        temperature (ndarray): Temperature (K).
        svp_mixing_ratio (ndarray): Saturation vapor pressure mixing ratio (kg/kg).

    Returns:
        ndarray: Air density (kg m-3).

    """
    return pressure / (con.RS * temperature * (0.6 * svp_mixing_ratio + 1))


def calc_psychrometric_constant(pressure):
    """Returns psychrometric constant.

    Psychrometric constant relates the partial pressure
    of water in air to the air temperature.

    Args:
        pressure (ndarray): Atmospheric pressure (Pa).

    Returns:
        ndarray: Psychrometric constant value (Pa K-1)

    """
    return pressure * con.SPECIFIC_HEAT / (con.LATENT_HEAT * con.MW_RATIO)


def calc_wet_bulb_temperature(model_data):
    """Returns wet bulb temperature.

    Returns wet bulb temperature for given temperature,
    pressure and relative humidity. Algorithm is based on a Taylor
    expansion of a simple expression for the saturated vapour pressure.

    Args:
        model_data (dict): Model variables `temperature`, `pressure`, `rh`.

    Returns:
        ndarray: Wet bulb temperature (K).

    References:
        J. Sullivan and L. D. Sanders: Method for obtaining wet-bulb
        temperatures by modifying the psychrometric formula.

    """
    def _screen_rh():
        rh = model_data['rh']
        rh_min = 1e-5
        rh[rh < rh_min] = rh_min
        return rh

    def _vapor_derivatives():
        m = 17.269
        tn = 35.86
        f1 = m*(tn - con.T0)
        f2 = dew_point - tn
        first = -vapor_pressure*f1/(f2**2)
        second = vapor_pressure*((f1/(f2**2))**2 + 2*f1/(f2**3))
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


def calc_dew_point_temperature(vapor_pressure):
    """ Returns dew point temperature.

    Args:
        vapor_pressure (ndarray): Water vapor pressure (Pa).

    Returns:
        ndarray: Dew point temperature (K).

    Notes:
        Method from Vaisala's white paper: "Humidity conversion formulas".

    """
    vaisala_parameters_over_water = (6.116441, 7.591386, 240.7263)
    a, m, tn = vaisala_parameters_over_water
    dew_point_celsius = tn / ((m/np.log10(vapor_pressure*P_TO_HPA/a))-1)
    return c2k(dew_point_celsius)


def get_attenuations(model, mwr, classification):
    """Calculates attenuations due to atmospheric gases and liquid water.

    Args:
        model (Model): The :class:`Model` instance.
        mwr (Mwr): The :class:`Mwr` instance.
        classification (ClassificationResult): The
            :class:`ClassificationResult` instance.

    Returns:
        dict: Dictionary containing `radar_gas_atten`, `radar_liquid_atten`,
        `liquid_atten_err`, `liquid_corrected` and `liquid_uncorrected` fields.

    """
    gas = GasAttenuation(model, classification)
    liquid = LiquidAttenuation(model, classification, mwr)
    return {
        'radar_gas_atten': gas.atten,
        'radar_liquid_atten': liquid.atten,
        'liquid_atten_err': liquid.atten_err,
        'liquid_corrected': liquid.corrected,
        'liquid_uncorrected': liquid.uncorrected
        }


class Attenuation:
    """Base class for gas and liquid attenuations.

    Args:
        model (Model): The :class:`Model` instance.
        classification (ClassificationResult): The :class:`ClassificationResult` instance.

    Attributes:
        classification (ClassificationResult): The :class:`ClassificationResult` instance.

    """
    def __init__(self, model, classification):
        self._dheight = utils.mdiff(model.height)
        self._model = model.data_dense
        self._liquid_in_pixel = utils.isbit(classification.category_bits, 0)
        self.classification = classification


class GasAttenuation(Attenuation):
    """Radar gas attenuation class. Child of Attenuation.

    Args:
        model (Model): The :class:`Model` instance.
        classification (ClassificationResult): The :class:`ClassificationResult` instance.

    Attributes:
        atten (ndarray): Gas attenuation (dB).

    """
    def __init__(self, model, classification):
        super().__init__(model, classification)
        self.atten = self._calc_gas_atten()

    def _calc_gas_atten(self):
        atten = np.copy(self._model['specific_gas_atten'])
        self._fix_atten_in_liquid(atten)
        return self._specific_to_gas_atten(atten)

    def _fix_atten_in_liquid(self, atten):
        saturated_atten = self._model['specific_saturated_gas_atten']
        atten[self._liquid_in_pixel] = saturated_atten[self._liquid_in_pixel]

    def _specific_to_gas_atten(self, specific_atten):
        layer1_atten = self._model['gas_atten'][:, 0]
        atten_cumsum = np.cumsum(specific_atten, axis=1)
        atten = TWO_WAY * atten_cumsum * self._dheight * M_TO_KM
        atten += utils.transpose(layer1_atten)
        atten = np.insert(atten, 0, layer1_atten, axis=1)[:, :-1]
        return atten


class LiquidAttenuation(Attenuation):
    """Radar liquid attenuation class. Child of Attenuation.

    Args:
        model (Model): The :class:`Model` instance.
        classification (ClassificationResult): The :class:`ClassificationResult` instance.
        mwr (Mwr): The :class:`Mwr` instance.

    Attributes:
        atten (ndarray): Radar liquid attenuation (dB).
        atten_err (ndarray): Error of radar liquid attenuation (dB).
        uncorrected (ndarray): Boolean array denoting uncorrected pixels.
        corrected (ndarray): Boolean array denoting corrected pixels.

    """
    def __init__(self, model, classification, mwr):
        super().__init__(model, classification)
        self._mwr = mwr.data
        self._lwc_dz_err = self._get_lwc_change_rate_error()
        self.atten = self._get_liquid_atten()
        self.atten_err = self._get_liquid_atten_err()
        self.uncorrected = self._find_pixels_hard_to_correct()
        self.corrected = self._find_corrected_pixels()
        self._mask_uncorrected_attenuation()

    def _get_lwc_change_rate_error(self):
        atmosphere = (self._model['temperature'], self._model['pressure'])
        return fill_clouds_with_lwc_dz(atmosphere, self._liquid_in_pixel)

    def _get_liquid_atten(self):
        """Finds radar liquid attenuation."""
        lwc = calc_adiabatic_lwc(self._lwc_dz_err, self._dheight)
        lwc_scaled = distribute_lwp_to_liquid_clouds(lwc, self._mwr['lwp'][:])
        return self._calc_attenuation(lwc_scaled)

    def _get_liquid_atten_err(self):
        """Finds radar liquid attenuation error."""
        lwc_err_scaled = distribute_lwp_to_liquid_clouds(self._lwc_dz_err,
                                                         self._mwr['lwp_error'][:])
        return self._calc_attenuation(lwc_err_scaled)

    def _calc_attenuation(self, lwc_scaled):
        """Calculates liquid attenuation (dB)."""
        liquid_attenuation = ma.zeros(lwc_scaled.shape)
        spec_liq = self._model['specific_liquid_atten']
        lwp_cumsum = ma.cumsum(lwc_scaled[:, :-1] * spec_liq[:, :-1], axis=1)
        liquid_attenuation[:, 1:] = TWO_WAY * lwp_cumsum * M_TO_KM
        return liquid_attenuation

    def _find_pixels_hard_to_correct(self):
        melting_layer = utils.isbit(self.classification.category_bits, 3)
        hard_to_correct = np.cumsum(melting_layer, axis=1) >= 1
        hard_to_correct[self.classification.is_rain, :] = True
        attenuated = self._find_attenuated_part_of_atmosphere()
        hard_to_correct[attenuated & self.atten.mask] = True
        return hard_to_correct

    def _find_corrected_pixels(self):
        return (self.atten > 0).filled(False) & ~self.uncorrected

    def _mask_uncorrected_attenuation(self):
        self.atten[self.uncorrected] = ma.masked

    def _find_attenuated_part_of_atmosphere(self):
        return np.cumsum(self._lwc_dz_err, axis=1) > 0


def fill_clouds_with_lwc_dz(atmosphere, is_liquid):
    """Fills liquid clouds with lwc change rate at the cloud bases.

    Args:
        atmosphere (tuple): 2-element tuple containing temperature (K) and pressure (Pa).
        is_liquid (ndarray): Boolean array indicating presence of liquid clouds.

    Returns:
        ndarray: liquid water content change rate (g/m3/m), so that for each
        cloud the base value is filled for the whole cloud.

    """
    lwc_dz = get_lwc_change_rate_at_bases(atmosphere, is_liquid)
    lwc_dz_filled = ma.zeros(lwc_dz.shape)
    lwc_dz_filled[is_liquid] = utils.ffill(lwc_dz[is_liquid])
    return lwc_dz_filled


def get_lwc_change_rate_at_bases(atmosphere, is_liquid):
    """Finds LWC change rate in liquid cloud bases.

    Args:
        atmosphere (tuple): 2-element tuple containing temperature (K) and
            pressure (Pa).
        is_liquid (ndarray): Boolean array indicating presence of liquid clouds.

    Returns:
        ndarray: liquid water content change rate at cloud bases (kg/m3/m).

    """
    liquid_bases = find_cloud_bases(is_liquid)
    lwc_dz = ma.zeros(liquid_bases.shape)
    lwc_dz[liquid_bases] = calc_lwc_change_rate(atmosphere[0][liquid_bases],
                                                atmosphere[1][liquid_bases])
    return lwc_dz


def find_cloud_bases(array):
    """Finds bases of clouds.

    Args:
        array (ndarray): 2D boolean array denoting clouds or some other
            similar field.

    Returns:
        ndarray: Boolean array indicating bases of the individual clouds.

    """
    zeros = np.zeros(array.shape[0])
    array_padded = np.insert(array, 0, zeros, axis=1).astype(int)
    return np.diff(array_padded, axis=1) == 1


def find_cloud_tops(array):
    """Finds tops of clouds.

    Args:
        array (ndarray): 2D boolean array denoting clouds or some other
            similar field.

    Returns:
        ndarray: Boolean array indicating tops of the individual clouds.

    """
    array_flipped = np.fliplr(array)
    bases_of_flipped = find_cloud_bases(array_flipped)
    return np.fliplr(bases_of_flipped)


def calc_adiabatic_lwc(lwc_change_rate, dheight):
    """Calculates adiabatic liquid water content (g/m3).

    Args:
        lwc_change_rate (ndarray): Liquid water content change rate (g/m3/m)
            calculated at the base of each cloud and filled to that cloud.
        dheight (float): Median difference of the height vector (m).

    Returns:
        Liquid water content (g/m3).

    """
    is_liquid = lwc_change_rate != 0
    ind_from_base = utils.cumsumr(is_liquid, axis=1)
    return ind_from_base * dheight * lwc_change_rate


def distribute_lwp_to_liquid_clouds(lwc, lwp):
    """Finds LWC that would produce measured LWP.

    Calculates LWP-weighted, normalized LWC. This is the measured
    LWP distributed to liquid cloud pixels according to their
    theoretical proportion, i.e., sum(scaled LWC) = measured LWP.

    Args:
        lwc (ndarray): 2D liquid water content (g/m3).
        lwp (ndarray): 1D liquid water path (g/m2).

    Returns:
        ndarray: 2D LWP-weighted, normalized LWC (g/m2).

    """
    lwc_sum = ma.sum(lwc, axis=1)
    return (lwc.T / lwc_sum * lwp).T


def c2k(temp):
    """Converts Celsius to Kelvins."""
    return ma.array(temp) + 273.15


def k2c(temp):
    """Converts Kelvins to Celsius."""
    return ma.array(temp) - 273.15
