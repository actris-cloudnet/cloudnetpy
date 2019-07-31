"""Module containing low-level functions to classify gridded
radar / lidar measurements.
"""
from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from cloudnetpy import droplet, utils
from cloudnetpy.constants import T0


def fetch_quality(radar, lidar, classification, attenuations):
    """Returns Cloudnet quality bits.

    Args:
        radar (Radar): Radar data container.
        lidar (Lidar): Lidar data container.
        classification (_ClassificationResult): Container for classification
            results.
        attenuations (dict):

    Returns:
        ndarray: Integer array containing the following bits:
            - bit 0: Pixel contains radar data
            - bit 1: Pixel contains lidar data
            - bit 2: Pixel contaminated by radar clutter
            - bit 3: Molecular scattering present (currently not implemented!)
            - bit 4: Pixel was affected by liquid attenuation
            - bit 5: Liquid attenuation was corrected

    See also:
        classify.fetch_cat_bits()

    """
    bits = [None]*6
    bits[0] = ~radar.data['Z'][:].mask
    bits[1] = ~lidar.data['beta'][:].mask
    bits[2] = classification.is_clutter
    bits[4] = attenuations['liquid_corrected'] | attenuations['liquid_uncorrected']
    bits[5] = attenuations['liquid_corrected']
    qbits = _bits_to_integer(bits)
    return {'quality_bits': qbits}


def classify_measurements(radar, lidar, model, mwr):
    """Classifies radar/lidar observations.

    This function classifies atmospheric scatterer from the input data.
    The input data needs to be averaged or interpolated to the common
    time / height grid before calling this function.

    Args:
        radar (Radar): A Radar object.
        lidar (Lidar): A Lidar object.
        model (Model): A Model object.
        mwr (Mwr): A Mwr object.

    Returns:
        _ClassificationResult: Object containing the result
            of classification.

    See also:
        classify.fetch_qual_bits()

    """
    obs = _ClassData(radar, lidar, model, mwr)
    bits = [None] * 6
    liquid = droplet.find_liquid(obs)
    bits[3] = find_melting_layer(obs)
    bits[2] = find_freezing_region(obs, bits[3])
    bits[0] = droplet.correct_liquid_top(obs, liquid, bits[2], limit=500)
    bits[5], insect_prob = find_insects(obs, bits[3], bits[0])
    bits[1] = find_falling_hydrometeors(obs, bits[0], bits[5])
    bits[4], extra_ice = find_aerosols(obs, bits[1], bits[0])
    bits[1][extra_ice] = True
    cat_bits = _bits_to_integer(bits)
    return _ClassificationResult(cat_bits, obs.is_rain, obs.is_clutter,
                                 insect_prob, liquid['bases'])


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
        There might be some detection problems with strong updrafts of air.
        In these cases the absolute values for speed do not make sense (rain
        drops can even move upwards instead of down).

    Args:
        obs (_ClassData): Input data container.
        smooth (bool, optional): If True, apply a small
            Gaussian smoother to the melting layer. Default is True.

    Returns:
        ndarray: 2-D boolean array denoting the melting layer.

    """

    def _slice(arg1, arg2):
        out1, out2 = arg1[ind, temp_indices], arg2[ind, temp_indices]
        return out1, out2, ma.count(out1)

    def _basetop(dprof, pind):
        top1 = droplet.ind_top(dprof, pind, len(temp_indices), 10, 2)
        base1 = droplet.ind_base(dprof, pind, 10, 2)
        return top1, base1

    if 'ecmwf' in obs.model_type.lower():
        t_range = (-4, 3)
    else:
        t_range = (-8, 6)

    melting_layer = np.zeros(obs.tw.shape, dtype=bool)
    ldr_diff = np.diff(obs.ldr, axis=1).filled(0)
    v_diff = np.diff(obs.v, axis=1).filled(0)

    for ind, t_prof in enumerate(obs.tw):
        temp_indices = np.where((t_prof > T0+t_range[0]) &
                                (t_prof < T0+t_range[1]))[0]
        ldr_prof, ldr_dprof, nldr = _slice(obs.ldr, ldr_diff)
        v_prof, v_dprof, nv = _slice(obs.v, v_diff)

        if nldr > 3 or nv > 3:
            ldr_p = np.argmax(ldr_prof)
            v_p = np.argmax(v_dprof)
            try:
                top, base = _basetop(ldr_dprof, ldr_p)
                conds = (ldr_prof[ldr_p] - ldr_prof[top] > 4,
                         ldr_prof[ldr_p] - ldr_prof[base] > 4,
                         ldr_prof[ldr_p] > -30,
                         v_prof[base] < -1)
                if all(conds):
                    melting_layer[ind, temp_indices[ldr_p]:temp_indices[top]+1] = True
            except:
                try:
                    top, base = _basetop(v_dprof, v_p)
                    diff = v_prof[top] - v_prof[base]
                    if diff > 0.5 and v_prof[base] < -2:
                        melting_layer[ind, temp_indices[v_p-1:v_p+2]] = True
                except:
                    continue
    if smooth:
        smoothed_layer = gaussian_filter(np.array(melting_layer, dtype=float), (2, 0.1))
        melting_layer = (smoothed_layer > 0.2).astype(bool)
    return melting_layer


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
        obs (_ClassData): Input data container.
        melting_layer (ndarray): 2D array denoting melting layer.
        liquid_layers (ndarray): 2D array denoting liquid layers.
        prob_lim (float, optional): Probability higher than
            this will lead to positive detection. Default is 0.8.

    Returns:
        2-element tuple containing

        - ndarray: 2-D probability of pixel containing insects.
        - ndarray: 2-D boolean flag of insects presence.

    """
    i_prob, i_prob_no_ldr = _insect_probability(obs)
    i_prob = _screen_insects(i_prob, i_prob_no_ldr, melting_layer, liquid_layers)
    is_insects = i_prob > prob_lim
    return is_insects, ma.masked_where(i_prob == 0, i_prob)


def _insect_probability(obs):
    def _interpolate_lwp():
        ind = ma.where(obs.lwp)
        return np.interp(obs.time, obs.time[ind], obs.lwp[ind])

    def _get_smoothed_v():
        smoothed_v = gaussian_filter(obs.v, (5, 5))
        smoothed_v = ma.array(smoothed_v)
        smoothed_v[obs.v.mask] = ma.masked
        return smoothed_v

    def _get_probabilities():
        smooth_v = _get_smoothed_v()
        lwp_interp = _interpolate_lwp()
        fun = utils.array_to_probability
        return {
            'width': fun(obs.width, 1, 0.3, True),
            'z': fun(obs.z, -15, 8, True),
            'ldr': fun(obs.ldr, -20, 5),
            'temp_loose': fun(obs.tw, 268, 2),
            'temp_strict': fun(obs.tw, 274, 1),
            'v': fun(smooth_v, -2, 1),
            'lwp': utils.transpose(fun(lwp_interp, 100, 50, invert=True))
        }
    prob = _get_probabilities()
    prob_combined = prob['z'] * prob['temp_loose'] * prob['ldr']
    prob_no_ldr = prob['z'] * prob['temp_strict'] * prob['v'] * prob['lwp'] * prob['width']
    no_ldr = np.where(prob_combined == 0)
    prob_combined[no_ldr] = prob_no_ldr[no_ldr]
    return prob_combined, prob_no_ldr


def _screen_insects(insect_prob, insect_prob_no_ldr, melting_layer, liquid_layers):
    def _screen_liquid_layers():
        prob[liquid_layers == 1] = 0

    def _screen_above_melting():
        above_melting = utils.ffill(melting_layer)
        prob[above_melting == 1] = 0

    def _screen_above_liquid():
        above_liquid = utils.ffill(liquid_layers)
        prob[np.logical_and(above_liquid == 1, insect_prob_no_ldr > 0)] = 0

    prob = np.copy(insect_prob)
    _screen_liquid_layers()
    _screen_above_melting()
    _screen_above_liquid()
    return prob


def find_falling_hydrometeors(obs, is_liquid, is_insects):
    """Finds falling hydrometeors.

    Falling hydrometeors are radar signals that are
    a) not insects b) not clutter. Furthermore, falling hydrometeors
    are strong lidar pixels excluding liquid layers (thus these pixels
    are ice or rain).

    Args:
        obs (_ClassData): Container for observations.
        is_liquid (ndarray): 2-D boolean array of liquid droplets.
        is_insects (ndarray): 2-D boolean array of insects.

    Returns:
        ndarray: 2-D boolean array containing falling hydrometeors.

    """
    def _find_falling_from_lidar():
        is_beta = ~obs.beta.mask
        return is_beta & ~is_liquid & (obs.beta.data > 2e-6)

    is_z = ~obs.z.mask
    no_clutter = ~obs.is_clutter
    no_insects = ~is_insects
    falling_from_lidar = _find_falling_from_lidar()
    is_falling = (is_z & no_clutter & no_insects) | falling_from_lidar
    return is_falling


def find_aerosols(obs, is_falling, is_liquid):
    """Estimates aerosols from lidar backscattering.

    Aerosols are lidar signals that are: a) not falling, b) not liquid droplets,
    and are present in warmer than some threshold temperature.

    Args:
        obs (_ClassData): Container for observations.
        is_falling (ndarray): 2-D boolean array of falling hydrometeors.
        is_liquid (ndarray): 2-D boolean array of liquid droplets.

    Returns:
        ndarray: 2-D boolean array containing aerosols.

    """
    temperature_limit = T0 - 15
    is_beta = ~obs.beta.mask
    potential_aerosols = is_beta & ~is_falling & ~is_liquid
    aerosols = np.logical_and(potential_aerosols, obs.tw > temperature_limit)
    ice = np.logical_and(potential_aerosols, obs.tw < temperature_limit)
    return aerosols, ice


def _bits_to_integer(bits):
    """Creates array of integers from individual boolean arrays.

    Args:
        bits (list): List of bit fields (of similar sizes)
        to be saved in the resulting array of integers. bits[0]
        is saved as bit 0, bits[1] as bit 1, etc.

    Returns:
        ndarray: Array of integers containing the information
            of the individual boolean arrays.

    """
    int_array = np.zeros_like(bits[0], dtype=int)
    for n, bit in enumerate(bits):
        ind = np.where(bit)  # works also if bit is None
        int_array[ind] = utils.setbit(int_array[ind].astype(int), n)
    return int_array


class _ClassData:
    def __init__(self, radar, lidar, model, mwr):
        self.z = radar.data['Z'][:]
        self.ldr = radar.data['ldr'][:]
        self.v = radar.data['v'][:]
        self.width = radar.data['width'][:]
        self.v_sigma = radar.data['v_sigma'][:]
        self.tw = model.data['Tw'][:]
        self.beta = lidar.data['beta'][:]
        self.lwp = mwr.data['lwp'][:]
        self.time = radar.time
        self.height = radar.height
        self.model_type = model.type
        self.is_rain = self._find_rain()
        self.is_clutter = self._find_clutter()

    def _find_rain(self, time_buffer=5):
        """Find profiles affected by rain.

        Rain is present in such profiles where the radar echo in
        the third range gate is > 0 dB. To make sure we do not include any
        rainy profiles, we also flag a few profiles before and after
        detections as raining.

        Args:
            time_buffer (int): Time in minutes.

        """
        is_rain = ma.array(self.z[:, 3] > 0, dtype=bool).filled(False)
        n_profiles = len(self.time)
        n_steps = utils.n_elements(self.time, time_buffer, 'time')
        for ind in np.where(is_rain)[0]:
            ind1 = max(0, ind - n_steps)
            ind2 = min(ind + n_steps, n_profiles)
            is_rain[ind1:ind2 + 1] = True
        return is_rain

    def _find_clutter(self, n_gates=10, v_lim=0.05):
        """Estimates clutter from doppler velocity.

        Args:
            n_gates (int, optional): Number of range gates from the ground
                where clutter is expected to be found. Default is 10.
            v_lim (float, optional): Velocity threshold. Smaller values are
                classified as clutter. Default is 0.05 (m/s).

        Returns:
            ndarray: 2-D boolean array denoting pixels contaminated by clutter.

        """
        is_clutter = np.zeros(self.v.shape, dtype=bool)
        tiny_velocity = (np.abs(self.v[:, :n_gates]) < v_lim).filled(False)
        is_clutter[:, :n_gates] = tiny_velocity * utils.transpose(~self.is_rain)
        return is_clutter


@dataclass
class _ClassificationResult:
    category_bits: np.ndarray
    is_rain: np.ndarray
    is_clutter: np.ndarray
    insect_prob: np.ndarray
    liquid_bases: np.ndarray
