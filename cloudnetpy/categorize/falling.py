"""Module to find falling hydrometeors from data."""
import numpy as np
from cloudnetpy import utils
from cloudnetpy.categorize import atmos
from cloudnetpy.constants import T0


def find_falling_hydrometeors(obs, is_liquid, is_insects):
    """Finds falling hydrometeors.

    Falling hydrometeors are radar signals that are
    a) not insects b) not clutter. Furthermore, falling hydrometeors
    are strong lidar pixels excluding liquid layers (thus these pixels
    are ice or rain). They are also weak radar signals in very cold
    temperatures.

    Args:
        obs (ClassData): The :class:`ClassData` instance.
        is_liquid (ndarray): 2-D boolean array of liquid droplets.
        is_insects (ndarray): 2-D boolean array of insects.

    Returns:
        ndarray: 2-D boolean array containing falling hydrometeors.

    """

    falling_from_radar = _find_falling_from_radar(obs, is_insects)
    falling_from_radar_fixed = _fix_liquid_dominated_radar(obs, falling_from_radar, is_liquid)
    falling_from_lidar = _find_falling_from_lidar(obs, is_liquid)
    cold_aerosols = _find_cold_aerosols(obs, is_liquid)
    return falling_from_radar_fixed | falling_from_lidar | cold_aerosols


def _find_falling_from_radar(obs, is_insects):
    is_z = ~obs.z.mask
    no_clutter = ~obs.is_clutter
    no_insects = ~is_insects
    return is_z & no_clutter & no_insects


def _find_falling_from_lidar(obs, is_liquid):
    is_beta = ~obs.beta.mask
    strong_beta_limit = 2e-6
    return is_beta & (obs.beta.data > strong_beta_limit) & ~is_liquid


def _find_cold_aerosols(obs, is_liquid):
    """Lidar signals which are in colder than the threshold temperature and
    have gap below in the profile are probably ice.
    """
    temperature_limit = T0 - 15
    is_beta = ~obs.beta.mask
    region = utils.ffill(is_beta, 1) == 0
    return is_beta & (obs.tw.data < temperature_limit) & ~is_liquid & region


def _fix_liquid_dominated_radar(obs, falling_from_radar, is_liquid):
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
            falling_from_radar[n, base:top+1] = False

    return falling_from_radar


def _is_z_missing_above_liquid(z, ind_top):
    """Checks is z is masked right above the liquid layer top."""
    if ind_top == len(z) - 1:
        return False
    return z.mask[ind_top+1]


def _is_z_increasing(z, ind_base, ind_top):
    """Checks is z is increasing inside the liquid cloud."""
    z = z[ind_base:ind_top+1].compressed()
    if len(z) > 1:
        return z[-1] > z[0]
    return False
