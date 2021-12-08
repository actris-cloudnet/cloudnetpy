"""Module to find insects from data."""
from typing import Optional, Tuple
import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import gaussian_filter
from cloudnetpy import utils
from cloudnetpy.categorize import droplet
from cloudnetpy.categorize.containers import ClassData


def find_insects(obs: ClassData,
                 melting_layer: np.ndarray,
                 liquid_layers: np.ndarray,
                 prob_lim: Optional[float] = 0.8) -> np.ndarray:
    """Returns insect probability and boolean array of insect presence.

    Insects are classified by estimating heuristic probability
    of insects from various individual radar parameters and combining
    these probabilities. Insects typically yield small echo and spectral width
    but high linear depolarization ratio (ldr), and they are present in warm
    temperatures.

    The combination of echo, ldr and temperature is generally the best proxy
    for insects. If ldr is not available, we use other radar parameters.

    Insects are finally screened from liquid layers and melting layer - and
    above melting layer.

    Args:
        obs: The :class:`ClassData` instance.
        melting_layer: 2D array denoting melting layer.
        liquid_layers: 2D array denoting liquid layers.
        prob_lim: Probability higher than this will lead to positive detection. Default is 0.8.

    Returns:
        tuple: 2-element tuple containing

        - 2-D boolean flag of insects presence.

    Notes:
        This insect detection method is novel and needs to be validated.

    """
    probabilities = _insect_probability(obs)
    insect_prob = _screen_insects(*probabilities, melting_layer, liquid_layers, obs)
    is_insects = insect_prob > prob_lim
    return is_insects


def _insect_probability(obs: ClassData) -> Tuple[np.ndarray, np.ndarray]:
    prob = _get_probabilities(obs)
    prob_from_ldr = _calc_prob_from_ldr(prob)
    prob_from_others = _calc_prob_from_all(prob)
    prob_from_others = _adjust_for_radar(obs, prob, prob_from_others)
    prob_combined = _fill_missing_pixels(prob_from_ldr, prob_from_others)
    return prob_combined, prob_from_others


def _get_probabilities(obs: ClassData) -> dict:
    smooth_v = _get_smoothed_v(obs)
    lwp_interp = droplet.interpolate_lwp(obs)
    fun = utils.array_to_probability
    return {
        'width': fun(obs.width, 1, 0.3, True) if hasattr(obs, 'width') else 1,
        'z_strong': fun(obs.z, 0, 8, True),
        'z_weak': fun(obs.z, -20, 8, True),
        'ldr': fun(obs.ldr, -25, 5) if hasattr(obs, 'ldr') else None,
        'temp_loose': fun(obs.tw, 268, 2),
        'temp_strict': fun(obs.tw, 274, 1),
        'v': fun(smooth_v, -3.5, 2),
        'lwp': utils.transpose(fun(lwp_interp, 150, 50, invert=True)),
        'v_sigma': fun(obs.v_sigma, 0.01, 0.1)}


def _get_smoothed_v(obs: ClassData,
                    sigma: Optional[Tuple[float, float]] = (5, 5)) -> np.ndarray:
    smoothed_v = gaussian_filter(obs.v, sigma)
    smoothed_v = ma.masked_where(obs.v.mask, smoothed_v)
    return smoothed_v


def _calc_prob_from_ldr(prob: dict) -> np.ndarray:
    """This is the most reliable proxy for insects."""
    if prob['ldr'] is None:
        return np.zeros(prob['z_strong'].shape)
    return prob['ldr'] * prob['temp_loose']


def _calc_prob_from_all(prob: dict) -> np.ndarray:
    """This can be tried when LDR is not available. To detect insects without LDR unambiguously is
    difficult and might result in many false positives and/or false negatives."""
    return prob['z_weak'] * prob['temp_strict'] * prob['width'] * prob['v']


def _adjust_for_radar(obs: ClassData,
                      prob: dict,
                      prob_from_others: np.ndarray) -> np.ndarray:
    """Adds radar-specific weighting to insect probabilities."""
    if 'mira' in obs.radar_type.lower():
        prob_from_others *= prob['lwp']
    return prob_from_others


def _fill_missing_pixels(prob_from_ldr: np.ndarray,
                         prob_from_others: np.ndarray) -> np.ndarray:
    prob_combined = np.copy(prob_from_ldr)
    no_ldr = np.where(prob_from_ldr == 0)
    prob_combined[no_ldr] = prob_from_others[no_ldr]
    return prob_combined


def _screen_insects(insect_prob, insect_prob_no_ldr, melting_layer,
                    liquid_layers, obs):

    def _screen_liquid_layers():
        prob[liquid_layers == 1] = 0

    def _screen_above_melting():
        above_melting = utils.ffill(melting_layer)
        prob[above_melting == 1] = 0

    def _screen_above_liquid():
        above_liquid = utils.ffill(liquid_layers)
        prob[(above_liquid == 1) & (insect_prob_no_ldr > 0)] = 0

    def _screen_rainy_profiles():
        prob[obs.is_rain == 1, :] = 0

    prob = np.copy(insect_prob)
    _screen_liquid_layers()
    _screen_above_melting()
    _screen_above_liquid()
    _screen_rainy_profiles()
    return prob
