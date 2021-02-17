"""Module containing low-level functions to classify gridded
radar / lidar measurements.
"""
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy.categorize import droplet
from cloudnetpy.categorize import melting, insects, falling, freezing
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
        Some of the individual classification methods are changed in this Python
        implementation compared to the original Cloudnet methodology.
        Especially methods classifying insects, melting layer and liquid droplets.

    """
    obs = ClassData(data)
    bits = [None] * 6
    liquid = droplet.find_liquid(obs)
    bits[3] = melting.find_melting_layer(obs)
    bits[2] = freezing.find_freezing_region(obs, bits[3])
    bits[0] = droplet.correct_liquid_top(obs, liquid, bits[2], limit=500)
    bits[5], insect_prob = insects.find_insects(obs, bits[3], bits[0])
    bits[1] = falling.find_falling_hydrometeors(obs, bits[0], bits[5])
    bits[4] = _find_aerosols(obs, bits[1], bits[0])
    return ClassificationResult(_bits_to_integer(bits),
                                obs.is_rain,
                                obs.is_clutter,
                                insect_prob,
                                liquid['bases'],
                                _find_profiles_with_undetected_melting(bits))


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

    """
    bits = [None]*6
    bits[0] = ~data['radar'].data['Z'][:].mask
    bits[1] = ~data['lidar'].data['beta'][:].mask
    bits[2] = classification.is_clutter
    bits[4] = attenuations['liquid_corrected'] | attenuations['liquid_uncorrected']
    bits[5] = attenuations['liquid_corrected']
    qbits = _bits_to_integer(bits)
    return {'quality_bits': qbits}


def _find_aerosols(obs: ClassData,
                   is_falling: np.ndarray,
                   is_liquid: np.ndarray) -> np.ndarray:
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


def _find_profiles_with_undetected_melting(bits: list) -> np.ndarray:
    drizzle_and_falling = _find_drizzle_and_falling(*bits[:3])
    transition = ma.diff(drizzle_and_falling, axis=1)
    is_transition = ma.any(transition, axis=1)
    is_melting_layer = ma.any(bits[3], axis=1)
    is_undetected_melting = is_transition & ~is_melting_layer
    is_undetected_melting[is_undetected_melting == 0] = ma.masked
    return is_undetected_melting.astype(int)


def _find_drizzle_and_falling(is_liquid: np.ndarray,
                              is_falling: np.ndarray,
                              is_freezing: np.ndarray) -> np.ndarray:
    """Classifies pixels as falling, drizzle and others.

    Args:
        is_liquid: 2D boolean array denoting liquid layers.
        is_falling: 2D boolean array denoting falling pixels.
        is_freezing: 2D boolean array denoting subzero temperatures.

    Returns:
        2D array where values are 1 (falling), 2 (drizzle), and masked (all others).

    """
    falling_dry = is_falling & ~is_liquid
    drizzle = falling_dry & ~is_freezing
    drizzle_and_falling = falling_dry.astype(int) + drizzle.astype(int)
    drizzle_and_falling = ma.copy(drizzle_and_falling)
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
