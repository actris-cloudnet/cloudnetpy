import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from cloudnetpy.constants import T0
from cloudnetpy import utils


def find_freezing_region(obs, melting_layer):
    """Finds freezing region using the model temperature and melting layer.

    Every profile that contains melting layer, subzero region starts from
    the mean melting layer height. If there are (long) time windows where
    no melting layer is present, model temperature is used in the
    middle of the time window. Finally, the subzero altitudes are linearly
    interpolated for all profiles.

    Args:
        obs (_ClassData): Input data container.
        melting_layer (ndarray): 2-D boolean array denoting melting layer.

    Returns:
        ndarray: 2-D boolean array denoting the sub-zero region.

    Notes:
        It is not clear how model temperature and melting layer should be
        ideally combined to determine the sub-zero region.

    """
    is_freezing = np.zeros(obs.tw.shape, dtype=bool)
    n_time = obs.time.shape[0]
    t0_alt = find_t0_alt(obs.tw, obs.height)
    alt_array = np.tile(obs.height, (n_time, 1))
    melting_alts = ma.array(alt_array, mask=~melting_layer)
    mean_melting_alt = ma.median(melting_alts, axis=1)
    freezing_alt = ma.copy(mean_melting_alt)
    for ind in (0, -1):
        freezing_alt[ind] = mean_melting_alt[ind] or t0_alt[ind]
    win = utils.n_elements(obs.time, 240, 'time')  # 4h window
    mid_win = int(win/2)
    for n in range(n_time-win):
        if mean_melting_alt[n:n+win].mask.all():
            freezing_alt[n+mid_win] = t0_alt[n+mid_win]
    ind = ~freezing_alt.mask
    f = interp1d(obs.time[ind], freezing_alt[ind])
    for ii, alt in enumerate(f(obs.time)):
        is_freezing[ii, obs.height > alt] = True
    return is_freezing


def find_t0_alt(temperature, height):
    """ Interpolates altitudes where temperature goes below freezing.

    Args:
        temperature (ndarray): 2-D temperature (K).
        height (ndarray): 1-D altitude grid (m).

    Returns:
        ndarray: 1-D array denoting altitudes where the
            temperature drops below 0 deg C.

    """
    alt = np.array([])
    for prof in temperature:
        ind = np.where(prof < T0)[0][0]
        if ind == 0:
            alt = np.append(alt, height[0])
        else:
            x = prof[ind-1:ind+1]
            y = height[ind-1:ind+1]
            x, y = zip(*sorted(zip(x, y)))
            alt = np.append(alt, np.interp(T0, x, y))
    return alt
