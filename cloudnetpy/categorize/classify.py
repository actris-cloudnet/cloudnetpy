"""Module containing low-level functions to classify gridded
radar / lidar measurements.
"""
from typing import List

import numpy as np
import skimage
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.categorize import droplet, falling, freezing, insects, melting
from cloudnetpy.categorize.containers import ClassData, ClassificationResult


def classify_measurements(data: dict) -> ClassificationResult:
    """Classifies radar/lidar observations.

    This function classifies atmospheric scatterers from the input data.
    The input data needs to be averaged or interpolated to the common
    time / height grid before calling this function.

    Args:
        data: Containing :class:`Radar`, :class:`Lidar`, :class:`Model` and :class:`Mwr` instances.

    Returns:
        A :class:`ClassificationResult` instance.

    References:
        The Cloudnet classification scheme is based on methodology proposed by
        Hogan R. and O'Connor E., 2004, https://bit.ly/2Yjz9DZ and its
        proprietary Matlab implementation.

    Notes:
        Some individual classification methods are changed in this Python
        implementation compared to the original Cloudnet methodology.
        Especially methods classifying insects, melting layer and liquid droplets.

    """
    obs = ClassData(data)
    bits: List[np.ndarray] = [np.array([])] * 6
    liquid = droplet.find_liquid(obs)
    bits[3] = melting.find_melting_layer(obs)
    bits[2] = freezing.find_freezing_region(obs, bits[3])
    bits[0] = droplet.correct_liquid_top(obs, liquid, bits[2], limit=500)
    bits[5], insect_prob = insects.find_insects(obs, bits[3], bits[0])
    bits[1] = falling.find_falling_hydrometeors(obs, bits[0], bits[5])
    bits, filtered_ice = _filter_falling(bits)
    for _ in range(5):
        bits[3] = _fix_undetected_melting_layer(bits)
        bits = _filter_insects(bits)
    bits[4] = _find_aerosols(obs, bits[1], bits[0])
    bits[4][filtered_ice] = False
    return ClassificationResult(
        _bits_to_integer(bits),
        obs.is_rain,
        obs.is_clutter,
        liquid["bases"],
        obs.rain_rate,
        insect_prob,
    )


def fetch_quality(data: dict, classification: ClassificationResult, attenuations: dict) -> dict:
    """Returns Cloudnet quality bits.

    Args:
        data: Containing :class:`Radar` and :class:`Lidar` instances.
        classification: A :class:`ClassificationResult` instance.
        attenuations: Dictionary containing keys `liquid_corrected`, `liquid_uncorrected`.

    Returns:
        Dictionary containing `quality_bits`, an integer array with the bits:

            - bit 0: Pixel contains radar data
            - bit 1: Pixel contains lidar data
            - bit 2: Pixel contaminated by radar clutter
            - bit 3: Molecular scattering present (currently not implemented!)
            - bit 4: Pixel was affected by liquid attenuation
            - bit 5: Liquid attenuation was corrected
            - bit 6: Data gap in radar or lidar data

    """
    bits: List[np.ndarray] = [np.ndarray([])] * 7
    radar_echo = data["radar"].data["Z"][:]
    bits[0] = ~radar_echo.mask
    bits[1] = ~data["lidar"].data["beta"][:].mask
    bits[2] = classification.is_clutter
    bits[4] = attenuations["liquid_corrected"] | attenuations["liquid_uncorrected"]
    bits[5] = attenuations["liquid_corrected"]
    qbits = _bits_to_integer(bits)
    return {"quality_bits": qbits}


def _find_aerosols(obs: ClassData, is_falling: np.ndarray, is_liquid: np.ndarray) -> np.ndarray:
    """Estimates aerosols from lidar backscattering.

    Aerosols are lidar signals that are: a) not falling, b) not liquid droplets.

    Args:
        obs: A :class:`ClassData` instance.
        is_falling: 2-D boolean array of falling hydrometeors.
        is_liquid: 2-D boolean array of liquid droplets.

    Returns:
        2-D boolean array containing aerosols.

    """
    is_beta = ~obs.beta.mask
    return is_beta & ~is_falling & ~is_liquid


def _fix_undetected_melting_layer(bits: list) -> np.ndarray:
    melting_layer = bits[3]
    drizzle_and_falling = _find_drizzle_and_falling(*bits[:3])
    transition = ma.diff(drizzle_and_falling, axis=1) == -1
    melting_layer[:, 1:][transition] = True
    return melting_layer


def _find_drizzle_and_falling(
    is_liquid: np.ndarray, is_falling: np.ndarray, is_freezing: np.ndarray
) -> np.ndarray:
    """Classifies pixels as falling, drizzle and others.

    Args:
        is_liquid: 2D boolean array denoting liquid layers.
        is_falling: 2D boolean array denoting falling pixels.
        is_freezing: 2D boolean array denoting subzero temperatures.

    Returns:
        2D array where values are 1 (falling, drizzle, supercooled liquids),
        2 (drizzle), and masked (all others).

    """
    falling_dry = is_falling & ~is_liquid
    supercooled_liquids = is_liquid & is_freezing
    drizzle = falling_dry & ~is_freezing
    drizzle_and_falling = falling_dry.astype(int) + drizzle.astype(int)
    drizzle_and_falling = ma.copy(drizzle_and_falling)
    drizzle_and_falling[supercooled_liquids] = 1
    drizzle_and_falling[drizzle_and_falling == 0] = ma.masked
    return drizzle_and_falling


def _bits_to_integer(bits: list) -> np.ndarray:
    """Creates array of integers from individual boolean arrays.

    Args:
        bits: List of bit fields (of similar sizes) to be saved in the resulting array of integers.
            bits[0] is saved as bit 0, bits[1] as bit 1, etc.

    Returns:
        Array of integers containing the information of the individual boolean arrays.

    """
    int_array = np.zeros_like(bits[0], dtype=int)
    for n, bit in enumerate(bits):
        ind = np.where(bit)  # works also if bit is None
        int_array[ind] = utils.setbit(int_array[ind].astype(int), n)
    return int_array


def _filter_insects(bits: list) -> list:
    is_melting_layer = bits[3]
    is_insects = bits[5]
    is_falling = bits[1]

    # Remove above melting layer
    above_melting = utils.ffill(is_melting_layer)
    ind = np.where(is_insects & above_melting)
    is_falling[ind] = True
    is_insects[ind] = False

    # remove around melting layer:
    original_insects = np.copy(is_insects)
    n_gates = 5
    for x, y in zip(*np.where(is_melting_layer)):
        try:
            # change insects to drizzle below melting layer pixel
            ind1 = np.arange(y - n_gates, y)
            ind11 = np.where(original_insects[x, ind1])[0]
            n_drizzle = sum(is_falling[x, :y])
            if n_drizzle > 5:
                is_falling[x, ind1[ind11]] = True
                is_insects[x, ind1[ind11]] = False
            else:
                continue
            # change insects on the right and left of melting layer pixel to drizzle
            ind1 = np.arange(x - n_gates, x + n_gates + 1)
            ind11 = np.where(original_insects[ind1, y])[0]
            is_falling[ind1[ind11], y - 1 : y + 2] = True
            is_insects[ind1[ind11], y - 1 : y + 2] = False
        except IndexError:
            continue
    bits[1] = is_falling
    bits[5] = is_insects
    return bits


def _filter_falling(bits: list) -> tuple:
    # filter falling ice speckle noise
    is_freezing = bits[2]
    is_falling = bits[1]
    is_falling_filtered = skimage.morphology.remove_small_objects(is_falling, 10, connectivity=1)
    is_filtered = is_falling & ~np.array(is_falling_filtered)
    ice_ind = np.where(is_freezing & is_filtered)
    is_falling[ice_ind] = False
    # in warm these are (probably) insects
    insect_ind = np.where(~is_freezing & is_filtered)
    is_falling[insect_ind] = False
    bits[1] = is_falling
    bits[5][insect_ind] = True
    return bits, ice_ind
