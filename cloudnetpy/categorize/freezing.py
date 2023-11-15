"""Module to find freezing region from data."""
import logging

import numpy as np
from numpy import ma
from scipy.interpolate import interp1d

from cloudnetpy import utils
from cloudnetpy.categorize.containers import ClassData
from cloudnetpy.constants import T0


def find_freezing_region(obs: ClassData, melting_layer: np.ndarray) -> np.ndarray:
    """Finds freezing region using the model temperature and melting layer.

    Every profile that contains melting layer, subzero region starts from
    the mean melting layer height. If there are (long) time windows where
    no melting layer is present, model temperature is used in the
    middle of the time window. Finally, the subzero altitudes are linearly
    interpolated for all profiles.

    Args:
    ----
        obs: The :class:`ClassData` instance.
        melting_layer: 2-D boolean array denoting melting layer.

    Returns:
    -------
        2-D boolean array denoting the sub-zero region.

    Notes:
    -----
        It is not clear how model temperature and melting layer should be
        ideally combined to determine the sub-zero region. This current
        method differs slightly from the original Matlab code and should
        be validated more carefully later.

    """
    is_freezing = np.zeros(obs.tw.shape, dtype=bool)
    t0_alt = _find_t0_alt(obs.tw, obs.height)
    mean_melting_alt = _find_mean_melting_alt(obs, melting_layer)

    if _is_all_freezing(mean_melting_alt, t0_alt, obs.height):
        logging.info("All temperatures below freezing and no detected melting layer")
        return np.ones(obs.tw.shape, dtype=bool)

    freezing_alt = ma.copy(mean_melting_alt)

    for ind in (0, -1):
        freezing_alt[ind] = mean_melting_alt[ind] or t0_alt[ind]
    win = utils.n_elements(obs.time, 240, "time")  # 4h window
    mid_win = int(win / 2)
    for n in range(len(obs.time) - win):
        if mean_melting_alt[n : n + win].mask.all():
            freezing_alt[n + mid_win] = t0_alt[n + mid_win]
    ind = ~freezing_alt.mask
    f = interp1d(obs.time[ind], freezing_alt[ind])
    freezing_alt_interpolated = f(obs.time) - 1
    for ii, alt in enumerate(freezing_alt_interpolated):
        is_freezing[ii, obs.height > alt] = True
    return is_freezing


def _is_all_freezing(
    mean_melting_alt: np.ndarray,
    t0_alt: np.ndarray,
    height: np.ndarray,
) -> bool:
    no_detected_melting = mean_melting_alt.all() is ma.masked
    all_temperatures_below_freezing = (t0_alt <= height[0]).all()
    return no_detected_melting and all_temperatures_below_freezing


def _find_mean_melting_alt(obs: ClassData, melting_layer: np.ndarray) -> ma.MaskedArray:
    if melting_layer.dtype != bool:
        msg = "melting_layer data type should be boolean"
        raise ValueError(msg)
    alt_array = np.tile(obs.height, (len(obs.time), 1))
    melting_alts = ma.array(alt_array, mask=~melting_layer)
    return ma.median(melting_alts, axis=1)


def _find_t0_alt(temperature: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Interpolates altitudes where temperature goes below freezing.

    Args:
    ----
        temperature: 2-D temperature (K).
        height: 1-D altitude grid (m).

    Returns:
    -------
        1-D array denoting altitudes where the temperature drops below 0 deg C.

    """
    alt: np.ndarray = np.array([])
    for prof in temperature:
        ind = np.where(prof < T0)[0][0]
        if ind == 0:
            alt = np.append(alt, height[0])
        else:
            x, y = zip(
                *sorted(
                    zip(
                        prof[ind - 1 : ind + 1], height[ind - 1 : ind + 1], strict=True
                    ),
                ),
                strict=True,
            )
            alt = np.append(alt, np.interp(T0, x, y))
    return alt
