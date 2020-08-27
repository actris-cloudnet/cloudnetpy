"""Functions to find melting layer from data."""

import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import gaussian_filter
from cloudnetpy.constants import T0
from cloudnetpy.categorize import droplet


def find_melting_layer(obs, smooth=True):
    """Finds melting layer from model temperature, ldr, and velocity.

    Melting layer is detected using linear depolarization ratio, *ldr*,
    Doppler velocity, *v*, and wet-bulb temperature, *Tw*.

    The algorithm is based on *ldr* having a clear Gaussian peak around
    the melting layer. This signature is caused by the growth of ice
    crystals into snowflakes that are much larger. In addition, when snow and
    ice melt, emerging heavy water droplets start to drop rapidly towards
    ground. Thus, there is also a similar positive peak in the
    first difference of *v*.

    The peak in *ldr* is the primary parameter we analyze. If
    *ldr* has a proper peak, and *v* < -1 m/s in the base, melting layer
    has been found. If *ldr* is missing we only analyze the behaviour
    of *v*, which is always present, to detect the melting layer.

    Model temperature is used to limit the melting layer search to a certain
    temperature range around 0 C. For ECMWF the range is -4..+3, and for
    the rest -8..+6.

    Notes:
        This melting layer detection method is novel and needs to be validated.
        Also note that there might be some detection problems with strong
        updrafts of air. In these cases the absolute values for speed do not
        make sense (rain drops can even move upwards instead of down).

    Args:
        obs (ClassData): The :class:`ClassData` instance.
        smooth (bool, optional): If True, apply a small
            Gaussian smoother to the melting layer. Default is True.

    Returns:
        ndarray: 2-D boolean array denoting the melting layer.

    """

    melting_layer = np.zeros(obs.tw.shape, dtype=bool)
    ldr_diff = np.diff(obs.ldr, axis=1).filled(0)
    v_diff = np.diff(obs.v, axis=1).filled(0)
    t_range = _find_model_temperature_range(obs.model_type)

    for ind, t_prof in enumerate(obs.tw):

        temp_indices = _get_temp_indices(t_prof, t_range)
        ldr_prof = obs.ldr[ind, temp_indices]
        ldr_dprof = ldr_diff[ind, temp_indices]
        v_prof = obs.v[ind, temp_indices]
        v_dprof = v_diff[ind, temp_indices]

        if ma.count(ldr_prof) > 3 or ma.count(v_prof) > 3:
            ldr_peak = np.argmax(ldr_prof)
            v_peak = np.argmax(v_dprof)
            try:
                top, base = _basetop(ldr_dprof, ldr_peak)
                if _is_good_ldr_peak(ldr_prof, v_prof, (base, ldr_peak, top)):
                    melting_layer[ind, temp_indices[ldr_peak-1]:temp_indices[top]+1] = True
            except:
                try:
                    top, base = _basetop(v_dprof, v_peak)
                    if _is_good_v_peak(v_prof, (base, top)):
                        melting_layer[ind, temp_indices[v_peak-1:v_peak+2]] = True
                except:
                    continue
    if smooth:
        smoothed_layer = gaussian_filter(np.array(melting_layer, dtype=float), (2, 0.1))
        melting_layer = (smoothed_layer > 0.2).astype(bool)
    return melting_layer


def _is_good_v_peak(v, indices):
    base, top = indices
    diff = v[top] - v[base]
    return diff > 0.5 and v[base] < -2


def _is_good_ldr_peak(ldr, v, indices):
    base, peak, top = indices
    conditions = (ldr[peak] - ldr[top] > 4,
                  ldr[peak] - ldr[base] > 4,
                  ldr[peak] > -30,
                  v[base] < -1)
    return all(conditions)


def _basetop(dprof, pind):
    """Finds the base and top of peak in ldr or v profile."""
    top = droplet.ind_top(dprof, pind, len(dprof), 10, 2)
    base = droplet.ind_base(dprof, pind, 10, 2)
    return top, base


def _get_temp_indices(t_prof, t_range):
    """Finds indices of temperature profile covering the given range."""
    bottom_point = np.where(t_prof < (T0 - t_range[0]))[0][0]
    top_point = np.where(t_prof > (T0 + t_range[0]))[0]
    top_point = top_point[-1] if top_point.size > 0 else 0
    return np.arange(bottom_point, top_point + 1)


def _find_model_temperature_range(model_type):
    """Returns temperature range around 0C for given model type."""
    if 'ecmwf' in model_type.lower():
        return -4, 3
    return -8, 6
