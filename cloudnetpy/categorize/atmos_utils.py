import atmoslib
import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy import utils


def find_cloud_bases(array: npt.NDArray) -> npt.NDArray:
    """Finds bases of clouds.

    Args:
        array: 2D boolean array denoting clouds or some other similar field.

    Returns:
        Boolean array indicating bases of the individual clouds.

    """
    zeros = np.zeros(array.shape[0])
    array_padded = np.insert(array, 0, zeros, axis=1).astype(int)
    return np.diff(array_padded, axis=1) == 1


def find_cloud_tops(array: npt.NDArray) -> npt.NDArray:
    """Finds tops of clouds.

    Args:
        array: 2D boolean array denoting clouds or some other similar field.

    Returns:
        Boolean array indicating tops of the individual clouds.

    """
    array_flipped = np.fliplr(array)
    bases_of_flipped = find_cloud_bases(array_flipped)
    return np.fliplr(bases_of_flipped)


def find_lowest_cloud_bases(
    cloud_mask: npt.NDArray,
    height: npt.NDArray,
) -> ma.MaskedArray:
    """Finds altitudes of cloud bases."""
    cloud_heights = cloud_mask * height
    return _find_lowest_heights(cloud_heights)


def find_highest_cloud_tops(
    cloud_mask: npt.NDArray,
    height: npt.NDArray,
) -> ma.MaskedArray:
    """Finds altitudes of cloud tops."""
    cloud_heights = cloud_mask * height
    cloud_heights_flipped = np.fliplr(cloud_heights)
    return _find_lowest_heights(cloud_heights_flipped)


def _find_lowest_heights(cloud_heights: npt.NDArray) -> ma.MaskedArray:
    inds = (cloud_heights != 0).argmax(axis=1)
    heights = np.array([cloud_heights[i, ind] for i, ind in enumerate(inds)])
    return ma.masked_equal(heights, 0.0)


def fill_clouds_with_lwc_dz(
    temperature: npt.NDArray, pressure: npt.NDArray, is_liquid: npt.NDArray
) -> npt.NDArray:
    """Fills liquid clouds with lwc change rate at the cloud bases.

    Args:
        temperature: 2D temperature array (K).
        pressure: 2D pressure array (Pa).
        is_liquid: Boolean array indicating presence of liquid clouds.

    Returns:
        Liquid water content change rate (kg m-3 m-1), so that for each cloud the base
        value is filled for the whole cloud.

    """
    lwc_dz = get_lwc_change_rate_at_bases(temperature, pressure, is_liquid)
    lwc_dz_filled = ma.zeros(lwc_dz.shape)
    lwc_dz_filled[is_liquid] = utils.ffill(lwc_dz[is_liquid])
    return lwc_dz_filled


def get_lwc_change_rate_at_bases(
    temperature: npt.NDArray,
    pressure: npt.NDArray,
    is_liquid: npt.NDArray,
) -> npt.NDArray:
    """Finds LWC change rate in liquid cloud bases.

    Args:
        temperature: 2D temperature array (K).
        pressure: 2D pressure array (Pa).
        is_liquid: Boolean array indicating presence of liquid clouds.

    Returns:
        Liquid water content change rate at cloud bases (kg m-3 m-1).

    """
    liquid_bases = find_cloud_bases(is_liquid)
    lwc_dz = ma.zeros(liquid_bases.shape)
    lwc_dz[liquid_bases] = atmoslib.adiabatic_dlwc_dz(
        temperature[liquid_bases],
        pressure[liquid_bases],
    )

    return lwc_dz


def calc_adiabatic_lwc(lwc_dz: npt.NDArray, height_agl: npt.NDArray) -> npt.NDArray:
    """Calculates adiabatic liquid water content (kg m-3).

    Args:
        lwc_dz: Liquid water content change rate (kg m-3 m-1) calculated at the
            base of each cloud and filled to that cloud.
        height_agl: Height above ground level vector (m).

    Returns:
        Liquid water content (kg m-3).

    """
    is_cloud = lwc_dz != 0
    cloud_indices = utils.cumsumr(is_cloud, axis=1)
    dz = utils.path_lengths_from_ground(height_agl) * np.ones_like(lwc_dz)
    dz[cloud_indices < 1] = 0
    return utils.cumsumr(dz, axis=1) * lwc_dz


def normalize_lwc_by_lwp(
    lwc_adiabatic: npt.NDArray, lwp: npt.NDArray, height_agl: npt.NDArray
) -> npt.NDArray:
    """Finds LWC that would produce measured LWP.

    Calculates LWP-weighted, normalized LWC. This is the measured
    LWP distributed to liquid cloud pixels according to their
    theoretical proportion.

    Args:
        lwc_adiabatic: Theoretical 2D liquid water content (kg m-3).
        lwp: 1D liquid water path (kg m-2).
        height_agl: Height above ground level vector (m).

    Returns:
        2D LWP-weighted, scaled LWC (kg m-3) that would produce the observed LWP.

    """
    path_lengths = utils.path_lengths_from_ground(height_agl)
    theoretical_lwp = ma.sum(lwc_adiabatic * path_lengths, axis=1)
    scaling_factors = lwp / theoretical_lwp
    return lwc_adiabatic * utils.transpose(scaling_factors)
