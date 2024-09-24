"""Functions to find melting layer from data."""

import numpy as np
from numpy import ma
from scipy.ndimage import gaussian_filter

from cloudnetpy import utils
from cloudnetpy.categorize import droplet
from cloudnetpy.categorize.containers import ClassData
from cloudnetpy.constants import T0


def find_melting_layer(obs: ClassData, *, smooth: bool = True) -> np.ndarray:
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
        obs: The :class:`ClassData` instance.
        smooth: If True, apply a small Gaussian smoother to the
            melting layer. Default is True.

    Returns:
        2-D boolean array denoting the melting layer.

    """
    melting_layer = np.zeros(obs.tw.shape, dtype=bool)

    ldr_prof: np.ndarray | None = None
    ldr_dprof: np.ndarray | None = None
    ldr_diff: np.ndarray | None = None
    width_prof = None

    if hasattr(obs, "ldr"):
        # Required for peak detection
        diffu = ma.array(np.diff(obs.ldr, axis=1))
        ldr_diff = diffu.filled(0)

    t_range = _find_model_temperature_range(obs.model_type)

    for ind, t_prof in enumerate(obs.tw):
        temp_indices = _get_temp_indices(t_prof, t_range)
        if len(temp_indices) <= 1:
            continue
        z_prof = obs.z[ind, temp_indices]
        v_prof = obs.v[ind, temp_indices]

        if ldr_diff is not None:
            if not hasattr(obs, "ldr"):
                msg = "ldr_diff is not None but obs.ldr does not exist"
                raise RuntimeError(msg)
            ldr_prof = obs.ldr[ind, temp_indices]
            ldr_dprof = ldr_diff[ind, temp_indices]

        if ma.count(ldr_prof) > 3 or ma.count(v_prof) > 3:
            try:
                if ldr_prof is None or ldr_dprof is None:
                    msg = "ldr_prof or ldr_dprof is None"
                    raise AssertionError(msg)  # noqa: TRY301
                indices = _find_melting_layer_from_ldr(
                    ldr_prof,
                    ldr_dprof,
                    v_prof,
                    z_prof,
                )
            except (ValueError, IndexError, AssertionError):
                height = obs.height[temp_indices]
                if hasattr(obs, "width"):
                    width_prof = obs.width[ind, temp_indices]
                indices = _find_melting_layer_from_v(v_prof, width_prof, height)
            if indices is not None:
                melting_layer[ind, temp_indices[indices]] = True

    if smooth:
        smoothed_layer = gaussian_filter(np.array(melting_layer, dtype=float), (2, 0.1))
        melting_layer = (smoothed_layer > 0.2).astype(bool)

    return melting_layer


def _find_melting_layer_from_ldr(
    ldr_prof: np.ndarray,
    ldr_dprof: np.ndarray,
    v_prof: np.ndarray,
    z_prof: np.ndarray,
) -> np.ndarray | None:
    peak = int(np.argmax(ldr_prof))
    base, top = _basetop(ldr_dprof, peak)
    conditions = (
        ldr_prof[peak] - ldr_prof[base] > 4,
        ldr_prof[peak] > -30,
        z_prof[base] > -25,
        v_prof[base] < -1.5,
    )

    if all(conditions):
        base = int(np.floor(base + (peak - base) / 2))
        return np.arange(base, top)
    return None


def _find_melting_layer_from_v(
    v_prof: np.ndarray,
    width_prof: np.ndarray | None,
    height: np.ndarray,
) -> np.ndarray | None:
    v = np.copy(v_prof[:-1])
    v_diff = np.diff(v_prof)
    v[v_diff < 0] = 0
    v[v_diff > 0] = 1
    n_increasing = utils.cumsumr(v)
    try:
        top = int(np.argmax(n_increasing))
        base = np.where(n_increasing[:top] == 0)[0][-1]
    except IndexError:
        return None
    if width_prof is not None:
        conditions = [
            width_prof[base] - width_prof[top] > 0.2,
            v_prof[top] - v_prof[base] > 0.5,
            50 < (height[top] - height[base]) < 1000,
            v_prof[base] < -2,
        ]
    else:
        conditions = [
            v_prof[top] - v_prof[base] > 2,
            50 < (height[top] - height[base]) < 1000,
            v_prof[base] < -2,
        ]
    if all(conditions):
        base = int(round(top - (top - base) / 2))
        return np.arange(base, top)
    return None


def _basetop(dprof: np.ndarray, pind: int) -> tuple[int, int]:
    """Finds the base and top of ldr peak."""
    top = droplet.ind_top(dprof, pind, len(dprof), 10, 2)
    base = droplet.ind_base(dprof, pind, 10, 2)
    return base, top


def _get_temp_indices(t_prof: np.ndarray, t_range: tuple) -> np.ndarray:
    """Finds indices of temperature profile covering the given range."""
    ind = np.where((t_prof > min(t_range) + T0) & (t_prof < max(t_range) + T0))[0]
    return np.array([]) if len(ind) == 0 else np.arange(min(ind), max(ind) + 1)


def _find_model_temperature_range(model_type: str) -> tuple[float, float]:
    """Returns temperature range around 0C for given model type."""
    if "gdas1" in model_type.lower():
        return -8, 6
    return -4, 3
