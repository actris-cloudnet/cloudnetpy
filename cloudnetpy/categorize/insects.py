"""Module to find insects from data."""
import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import gaussian_filter
from cloudnetpy import utils
from cloudnetpy.categorize import droplet


def find_insects(obs, melting_layer, liquid_layers, prob_lim=0.8):
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
        obs (ClassData): The :class:`ClassData` instance.
        melting_layer (ndarray): 2D array denoting melting layer.
        liquid_layers (ndarray): 2D array denoting liquid layers.
        prob_lim (float, optional): Probability higher than
            this will lead to positive detection. Default is 0.8.

    Returns:
        tuple: 2-element tuple containing

        - ndarray: 2-D probability of pixel containing insects.
        - ndarray: 2-D boolean flag of insects presence.

    """
    probabilities = _insect_probability(obs)
    insect_prob = _screen_insects(*probabilities, melting_layer, liquid_layers, obs)
    is_insects = insect_prob > prob_lim
    return is_insects, ma.masked_where(insect_prob == 0, insect_prob)


def _insect_probability(obs):
    prob = _get_probabilities(obs)
    prob_from_ldr = _calc_prob_from_ldr(prob)
    prob_from_others = _calc_prob_from_all(prob)
    prob_from_others = _adjust_for_radar(obs, prob, prob_from_others)
    prob_combined = _fill_missing_pixels(prob_from_ldr, prob_from_others)
    return prob_combined, prob_from_others


def _get_probabilities(obs):
    smooth_v = _get_smoothed_v(obs)
    lwp_interp = droplet.interpolate_lwp(obs)
    fun = utils.array_to_probability
    return {
        'width': fun(obs.width, 1, 0.3, True),
        'z': fun(obs.z, -15, 8, True),
        'ldr': fun(obs.ldr, -20, 5),
        'temp_loose': fun(obs.tw, 268, 2),
        'temp_strict': fun(obs.tw, 274, 1),
        'v': fun(smooth_v, -2.5, 2),
        'lwp': utils.transpose(fun(lwp_interp, 150, 50, invert=True)),
        'v_sigma': fun(obs.v_sigma, 0.01, 0.1)}


def _get_smoothed_v(obs, sigma=(5, 5)):
    smoothed_v = gaussian_filter(obs.v, sigma)
    smoothed_v = ma.masked_where(obs.v.mask, smoothed_v)
    return smoothed_v


def _calc_prob_from_ldr(prob):
    """This is the most reliable proxy for insects."""
    return prob['z'] * prob['temp_loose'] * prob['ldr']


def _calc_prob_from_all(prob):
    """This can be used to detect insects when ldr is not available."""
    return prob['z'] * prob['temp_strict'] * prob['v'] * prob['width']


def _adjust_for_radar(obs, prob, prob_from_others):
    """Adds radar-specific weighting to insect probabilities."""
    if 'mira' in obs.radar_type.lower():
        prob_from_others *= prob['lwp']
    return prob_from_others


def _fill_missing_pixels(prob_from_ldr, prob_from_others):
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
