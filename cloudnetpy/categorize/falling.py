"""Module to find falling hydrometeors from data."""

import numpy as np
from numpy import ma

from cloudnetpy.categorize import atmos_utils
from cloudnetpy.categorize.containers import ClassData
from cloudnetpy.constants import T0


def find_falling_hydrometeors(
    obs: ClassData,
    is_liquid: np.ndarray,
    is_insects: np.ndarray,
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
    falling_from_radar_fixed = _fix_liquid_dominated_radar(
        obs,
        falling_from_radar,
        is_liquid,
    )
    cold_aerosols = _find_cold_aerosols(obs, is_liquid)
    return falling_from_radar_fixed | cold_aerosols


def _find_falling_from_radar(obs: ClassData, is_insects: np.ndarray) -> np.ndarray:
    is_z = ~obs.z.mask
    no_clutter = ~obs.is_clutter
    no_insects = ~is_insects
    return is_z & no_clutter & no_insects


def _find_cold_aerosols(obs: ClassData, is_liquid: np.ndarray) -> np.ndarray:
    """Lidar signals which are in colder than the threshold temperature
    and threshold altitude from the ground are assumed ice.

    These pixels are easily mixed with aerosols at lower altitudes,
    and at higher altitudes they could be supercooled liquid, actually.
    This should be investigated and fixed in the future.
    """
    cold_aerosols = np.zeros(is_liquid.shape, dtype=bool)
    lidar_range = obs.height - obs.altitude
    cold_aerosol_temperature_limit = T0 - 15
    cold_aerosol_min_altitude = 2000
    is_beta = ~obs.beta.mask
    lidar_ice_indices = np.where(
        (obs.tw.data < cold_aerosol_temperature_limit) & is_beta & ~is_liquid,
    )
    cold_aerosols[lidar_ice_indices] = True
    low_range_indices = np.where(lidar_range < cold_aerosol_min_altitude)
    if low_range_indices:
        cold_aerosols[:, low_range_indices] = False

    # Further investigate range gates between 2000 and 4000 m
    # to avoid abrupt transitions from aerosol to ice.
    altitude_limit = 4000
    window_size = 6
    n_beta_in_window = 2
    for time_ind, profile in enumerate(cold_aerosols):
        for alt_ind, is_cold_aerosol in enumerate(profile):
            if is_cold_aerosol and lidar_range[alt_ind] < altitude_limit:
                start_ind = max(0, alt_ind - window_size + 1)
                end_ind = alt_ind + 1
                n_beta_below = np.sum(is_beta[time_ind, start_ind:end_ind])
                if n_beta_below > n_beta_in_window:
                    cold_aerosols[time_ind, alt_ind] = False

    return cold_aerosols


def _fix_liquid_dominated_radar(
    obs: ClassData,
    falling_from_radar: np.ndarray,
    is_liquid: np.ndarray,
) -> np.ndarray:
    """Radar signals inside liquid clouds are NOT ice if Z is
    increasing in height inside the cloud.
    """
    liquid_bases = atmos_utils.find_cloud_bases(is_liquid)
    liquid_tops = atmos_utils.find_cloud_tops(is_liquid)
    base_indices = np.where(liquid_bases)
    top_indices = np.where(liquid_tops)

    for n, base, _, top in zip(*base_indices, *top_indices, strict=True):
        z_prof = obs.z[n, :]
        if _is_z_missing_above_liquid(z_prof, top) and _is_z_increasing(
            z_prof,
            base,
            top,
        ):
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
