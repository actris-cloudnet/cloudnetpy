"""This module contains functions to calculate
various atmospheric parameters.
"""
from typing import Final

import numpy as np
import scipy.constants
from numpy import ma

from cloudnetpy import constants as con
from cloudnetpy import utils
from cloudnetpy.categorize.atmos_utils import (
    calc_psychrometric_constant,
    calc_saturation_vapor_pressure,
)
from cloudnetpy.categorize.containers import ClassificationResult
from cloudnetpy.categorize.model import Model

M_TO_KM: Final = 0.001
TWO_WAY: Final = 2


def calc_lwc_change_rate(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Returns rate of change of condensable water (LWC).

    Calculates the theoretical adiabatic rate of increase of LWC
    with height, given the cloud base temperature and pressure.

    Args:
    ----
        temperature: Temperature of cloud base (K).
        pressure: Pressure of cloud base (Pa).

    Returns:
    -------
        dlwc/dz (kg m-3 m-1)

    References:
    ----------
        Brenguier, 1991, https://bit.ly/2QCSJtb

    """
    svp = calc_saturation_vapor_pressure(temperature)
    svp_mixing_ratio = calc_mixing_ratio(svp, pressure)
    air_density = calc_air_density(pressure, temperature, svp_mixing_ratio)
    kelvin_per_kg = calc_psychrometric_constant(temperature)
    pressure_difference = pressure - svp
    f1 = kelvin_per_kg - 1
    f2 = 1 / (
        kelvin_per_kg
        + (con.LATENT_HEAT * svp_mixing_ratio * air_density / pressure_difference)
    )
    f3 = con.MW_RATIO * svp * pressure_difference**-2
    dqs_dp = f1 * f2 * f3
    return dqs_dp * air_density**2 * -scipy.constants.g


def calc_mixing_ratio(svp: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Calculates mixing ratio from saturation vapor pressure and pressure.

    Args:
    ----
        svp: Saturation vapor pressure (Pa).
        pressure: Atmospheric pressure (Pa).

    Returns:
    -------
        Mixing ratio (kg kg-1).

    """
    return con.MW_RATIO * svp / (pressure - svp)


def calc_air_density(
    pressure: np.ndarray,
    temperature: np.ndarray,
    svp_mixing_ratio: np.ndarray,
) -> np.ndarray:
    """Calculates air density (kg m-3).

    Args:
    ----
        pressure: Pressure (Pa).
        temperature: Temperature (K).
        svp_mixing_ratio: Saturation vapor pressure mixing ratio (kg/kg).

    Returns:
    -------
        Air density (kg m-3).

    """
    return pressure / (con.RS * temperature * (0.6 * svp_mixing_ratio + 1))


def get_attenuations(data: dict, classification: ClassificationResult) -> dict:
    """Calculates attenuations due to atmospheric gases and liquid water.

    Args:
    ----
        data: Containing :class:`Model` and :class:`Mwr` instances.
        classification: A :class:`ClassificationResult` instance.

    Returns:
    -------
        Dictionary containing `radar_gas_atten`, `radar_liquid_atten`,
            `liquid_atten_err`, `liquid_corrected` and `liquid_uncorrected` fields.

    """
    gas = GasAttenuation(data, classification)
    liquid = LiquidAttenuation(data, classification)
    return {
        "radar_gas_atten": gas.atten,
        "radar_liquid_atten": liquid.atten,
        "liquid_atten_err": liquid.atten_err,
        "liquid_corrected": liquid.corrected,
        "liquid_uncorrected": liquid.uncorrected,
    }


class Attenuation:
    """Base class for gas and liquid attenuations.

    Args:
    ----
        model: The :class:`Model` instance.
        classification: The :class:`ClassificationResult` instance.

    Attributes:
    ----------
        classification (ClassificationResult): The :class:`ClassificationResult`
        instance.

    """

    def __init__(self, model: Model, classification: ClassificationResult):
        self._dheight = utils.mdiff(model.height)
        self._model = model.data_dense
        self._liquid_in_pixel = utils.isbit(classification.category_bits, 0)
        self.classification = classification


class GasAttenuation(Attenuation):
    """Radar gas attenuation class. Child of Attenuation.

    Args:
    ----
        data: Containing :class:`Model` instance.
        classification: The :class:`ClassificationResult` instance.

    Attributes:
    ----------
        atten (ndarray): Gas attenuation (dB).

    """

    def __init__(self, data: dict, classification: ClassificationResult):
        super().__init__(data["model"], classification)
        self.atten = self._calc_gas_atten()

    def _calc_gas_atten(self) -> np.ndarray:
        specific_atten = ma.copy(self._model["specific_gas_atten"])
        specific_atten_corrected = self._fix_atten_in_liquid(specific_atten)
        return self._specific_to_gas_atten(specific_atten_corrected)

    def _fix_atten_in_liquid(self, atten: np.ndarray) -> np.ndarray:
        saturated_atten = self._model["specific_saturated_gas_atten"]
        atten[self._liquid_in_pixel] = saturated_atten[self._liquid_in_pixel]
        return atten

    def _specific_to_gas_atten(self, specific_atten: np.ndarray) -> np.ndarray:
        layer1_atten = self._model["gas_atten"][:, 0]
        atten_cumsum = ma.cumsum(specific_atten, axis=1)
        atten = TWO_WAY * atten_cumsum * self._dheight * M_TO_KM
        atten += utils.transpose(layer1_atten)
        atten = np.insert(atten, 0, layer1_atten, axis=1)[:, :-1]
        return ma.array(atten, mask=atten_cumsum.mask)


class LiquidAttenuation(Attenuation):
    """Radar liquid attenuation class. Child of Attenuation.

    Args:
    ----
        data: Containing :class:`Model` and :class:`Mwr` instances.
        classification: The :class:`ClassificationResult` instance.

    Attributes:
    ----------
        atten (ndarray): Radar liquid attenuation (dB).
        atten_err (ndarray): Error of radar liquid attenuation (dB).
        uncorrected (ndarray): Boolean array denoting uncorrected pixels.
        corrected (ndarray): Boolean array denoting corrected pixels.

    """

    def __init__(self, data: dict, classification: ClassificationResult):
        super().__init__(data["model"], classification)
        self._mwr = data["mwr"].data
        self._lwc_dz_err = self._get_lwc_change_rate_error()
        self.atten = self._get_liquid_atten()
        self.atten_err = self._get_liquid_atten_err()
        self.uncorrected = self._find_pixels_hard_to_correct()
        self.corrected = self._find_corrected_pixels()
        self._mask_uncorrected_attenuation()

    def _get_lwc_change_rate_error(self) -> np.ndarray:
        atmosphere = (self._model["temperature"], self._model["pressure"])
        return fill_clouds_with_lwc_dz(atmosphere, self._liquid_in_pixel)

    def _get_liquid_atten(self) -> ma.MaskedArray:
        """Finds radar liquid attenuation."""
        lwp = ma.copy(self._mwr["lwp"][:])
        lwp[lwp < 0] = 0
        lwc = calc_adiabatic_lwc(self._lwc_dz_err, self._dheight)
        lwc_scaled = distribute_lwp_to_liquid_clouds(lwc, lwp)
        return self._calc_attenuation(lwc_scaled)

    def _get_liquid_atten_err(self) -> ma.MaskedArray:
        """Finds radar liquid attenuation error."""
        lwc_err_scaled = distribute_lwp_to_liquid_clouds(
            self._lwc_dz_err,
            self._mwr["lwp_error"][:],
        )
        return self._calc_attenuation(lwc_err_scaled)

    def _calc_attenuation(self, lwc_scaled: np.ndarray) -> ma.MaskedArray:
        """Calculates liquid attenuation (dB)."""
        liquid_attenuation = ma.zeros(lwc_scaled.shape)
        spec_liq = self._model["specific_liquid_atten"]
        lwp_cumsum = ma.cumsum(lwc_scaled[:, :-1] * spec_liq[:, :-1], axis=1)
        liquid_attenuation[:, 1:] = TWO_WAY * lwp_cumsum * M_TO_KM
        return liquid_attenuation

    def _find_pixels_hard_to_correct(self) -> np.ndarray:
        melting_layer = utils.isbit(self.classification.category_bits, 3)
        hard_to_correct = np.cumsum(melting_layer, axis=1) >= 1
        hard_to_correct[self.classification.is_rain, :] = True
        attenuated = self._find_attenuated_part_of_atmosphere()
        hard_to_correct[attenuated & self.atten.mask] = True
        return hard_to_correct

    def _find_corrected_pixels(self) -> np.ndarray:
        proper_values = ma.array(self.atten > 0)
        filled = False
        return proper_values.filled(filled) & ~self.uncorrected

    def _mask_uncorrected_attenuation(self) -> None:
        self.atten[self.uncorrected] = ma.masked

    def _find_attenuated_part_of_atmosphere(self) -> np.ndarray:
        return np.cumsum(self._lwc_dz_err, axis=1) > 0


def fill_clouds_with_lwc_dz(atmosphere: tuple, is_liquid: np.ndarray) -> np.ndarray:
    """Fills liquid clouds with lwc change rate at the cloud bases.

    Args:
    ----
        atmosphere: 2-element tuple containing temperature (K) and pressure (Pa).
        is_liquid: Boolean array indicating presence of liquid clouds.

    Returns:
    -------
        Liquid water content change rate (kg/m3/m), so that for each cloud the base
        value is filled for the whole cloud.

    """
    lwc_dz = get_lwc_change_rate_at_bases(atmosphere, is_liquid)
    lwc_dz_filled = ma.zeros(lwc_dz.shape)
    lwc_dz_filled[is_liquid] = utils.ffill(lwc_dz[is_liquid])
    return lwc_dz_filled


def get_lwc_change_rate_at_bases(
    atmosphere: tuple,
    is_liquid: np.ndarray,
) -> np.ndarray:
    """Finds LWC change rate in liquid cloud bases.

    Args:
    ----
        atmosphere: 2-element tuple containing temperature (K) and pressure (Pa).
        is_liquid: Boolean array indicating presence of liquid clouds.

    Returns:
    -------
        Liquid water content change rate at cloud bases (kg/m3/m).

    """
    liquid_bases = find_cloud_bases(is_liquid)
    lwc_dz = ma.zeros(liquid_bases.shape)
    lwc_dz[liquid_bases] = calc_lwc_change_rate(
        atmosphere[0][liquid_bases],
        atmosphere[1][liquid_bases],
    )
    return lwc_dz


def find_cloud_bases(array: np.ndarray) -> np.ndarray:
    """Finds bases of clouds.

    Args:
    ----
        array: 2D boolean array denoting clouds or some other similar field.

    Returns:
    -------
        Boolean array indicating bases of the individual clouds.

    """
    zeros = np.zeros(array.shape[0])
    array_padded = np.insert(array, 0, zeros, axis=1).astype(int)
    return np.diff(array_padded, axis=1) == 1


def find_cloud_tops(array: np.ndarray) -> np.ndarray:
    """Finds tops of clouds.

    Args:
    ----
        array: 2D boolean array denoting clouds or some other similar field.

    Returns:
    -------
        Boolean array indicating tops of the individual clouds.

    """
    array_flipped = np.fliplr(array)
    bases_of_flipped = find_cloud_bases(array_flipped)
    return np.fliplr(bases_of_flipped)


def find_lowest_cloud_bases(
    cloud_mask: np.ndarray,
    height: np.ndarray,
) -> ma.MaskedArray:
    """Finds altitudes of cloud bases."""
    cloud_heights = cloud_mask * height
    return _find_lowest_heights(cloud_heights)


def find_highest_cloud_tops(
    cloud_mask: np.ndarray,
    height: np.ndarray,
) -> ma.MaskedArray:
    """Finds altitudes of cloud tops."""
    cloud_heights = cloud_mask * height
    cloud_heights_flipped = np.fliplr(cloud_heights)
    return _find_lowest_heights(cloud_heights_flipped)


def _find_lowest_heights(cloud_heights: np.ndarray) -> ma.MaskedArray:
    inds = (cloud_heights != 0).argmax(axis=1)
    heights = np.array([cloud_heights[i, ind] for i, ind in enumerate(inds)])
    return ma.masked_equal(heights, 0.0)


def calc_adiabatic_lwc(lwc_change_rate: np.ndarray, dheight: float) -> np.ndarray:
    """Calculates adiabatic liquid water content (kg/m3).

    Args:
    ----
        lwc_change_rate: Liquid water content change rate (kg/m3/m) calculated at the
            base of each cloud and filled to that cloud.
        dheight: Median difference of the height vector (m).

    Returns:
    -------
        Liquid water content (kg/m3).

    """
    is_liquid = lwc_change_rate != 0
    ind_from_base = utils.cumsumr(is_liquid, axis=1)
    return ind_from_base * dheight * lwc_change_rate


def distribute_lwp_to_liquid_clouds(lwc: np.ndarray, lwp: np.ndarray) -> np.ndarray:
    """Finds LWC that would produce measured LWP.

    Calculates LWP-weighted, normalized LWC. This is the measured
    LWP distributed to liquid cloud pixels according to their
    theoretical proportion, i.e., sum(scaled LWC) = measured LWP.

    Args:
    ----
        lwc: 2D liquid water content (kg/m3).
        lwp: 1D liquid water path (kg/m2).

    Returns:
    -------
        2D LWP-weighted, normalized LWC (kg/m2).

    """
    lwc_sum = ma.sum(lwc, axis=1)
    return (lwc.T / lwc_sum * lwp).T
