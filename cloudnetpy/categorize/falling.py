"""Module to find falling hydrometeors from data."""
import numpy as np
from numpy import ma

from cloudnetpy.categorize import atmos
from cloudnetpy.categorize.containers import ClassData
from cloudnetpy.constants import T0


def find_falling_hydrometeors(
    obs: ClassData, is_liquid: np.ndarray, is_insects: np.ndarray
) -> np.ndarray:
    """Finds falling hydrometeors.

    Falling hydrometeors are radar signals that are
    a) not insects b) not clutter. Furthermore, falling hydrometeors
    are strong lidar pixels excluding liquid layers (thus these pixels
    are ice or rain). They are also weak radar signals in very cold
    temperatures.

    Args:
        obs: The :class:`ClassData` instance.
        is_liquid: 2-D boolean array of liquid droplets.
        is_insects: 2-D boolean array of insects.

    Returns:
        2-D boolean array containing falling hydrometeors.

    References:
        Hogan R. and O'Connor E., 2004, https://bit.ly/2Yjz9DZ.

    """

    falling_from_radar = _find_falling_from_radar(obs, is_insects)
    falling_from_radar_fixed = _fix_liquid_dominated_radar(obs, falling_from_radar, is_liquid)
    cold_aerosols = _find_cold_aerosols(obs, is_liquid)
    return falling_from_radar_fixed | cold_aerosols


def _find_falling_from_radar(obs: ClassData, is_insects: np.ndarray) -> np.ndarray:
    is_z = ~obs.z.mask
    no_clutter = ~obs.is_clutter
    no_insects = ~is_insects
    return is_z & no_clutter & no_insects


def _find_cold_aerosols(obs: ClassData, is_liquid: np.ndarray) -> np.ndarray:
    """Lidar signals which are in colder than the threshold temperature are assumed ice.
    This method should be improved in the future if possible.
    """
    cold_aerosols = np.zeros(is_liquid.shape, dtype=bool)
    temperature_limit = T0 - 15
    is_beta = ~obs.beta.mask
    ind = np.where((obs.tw.data < temperature_limit) & is_beta & ~is_liquid)
    cold_aerosols[ind] = True
    return cold_aerosols


def _fix_liquid_dominated_radar(
    obs: ClassData, falling_from_radar: np.ndarray, is_liquid: np.ndarray
) -> np.ndarray:
    """Radar signals inside liquid clouds are NOT ice if Z is
    increasing in height inside the cloud.
    """
    liquid_bases = atmos.find_cloud_bases(is_liquid)
    liquid_tops = atmos.find_cloud_tops(is_liquid)
    base_indices = np.where(liquid_bases)
    top_indices = np.where(liquid_tops)

    for n, base, _, top in zip(*base_indices, *top_indices):
        z_prof = obs.z[n, :]
        if _is_z_missing_above_liquid(z_prof, top) and _is_z_increasing(z_prof, base, top):
            falling_from_radar[n, base : top + 1] = False

    return falling_from_radar


def _is_z_missing_above_liquid(z: ma.MaskedArray, ind_top: int) -> bool:
    """Checks is z is masked right above the liquid layer top."""
    if ind_top == len(z) - 1:
        return False
    return z.mask[ind_top + 1]


def _is_z_increasing(z: ma.MaskedArray, ind_base: int, ind_top: int) -> bool:
    """Checks is z is increasing inside the liquid cloud."""
    z = z[ind_base : ind_top + 1].compressed()
    if len(z) > 1:
        return z[-1] > z[0]
    return False
