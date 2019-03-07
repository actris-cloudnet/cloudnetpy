"""Module containing low-level functions to classify gridded
radar / lidar measurements.
"""
from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import scipy.ndimage
from scipy import stats
from scipy.interpolate import interp1d
from . import droplet, utils
from .constants import T0


class _ClassData:
    def __init__(self, radar, lidar, model):
        self.z = radar.data['Z'][:]
        self.ldr = radar.data['ldr'][:]
        self.v = radar.data['v'][:]
        self.width = radar.data['width'][:]
        self.tw = model.data['Tw'][:]
        self.beta = lidar.data['beta'][:]
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
            i1 = max(0, ind - n_steps)
            i2 = min(ind + n_steps, n_profiles)
            is_rain[i1:i2 + 1] = True
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
        is_clutter[:, :n_gates] = (tiny_velocity.T * (~self.is_rain)).T
        return is_clutter


@dataclass
class _ClassificationResult:
    category_bits: np.ndarray
    is_rain: np.ndarray
    is_clutter: np.ndarray
    insect_prob: np.ndarray
    liquid_bases: np.ndarray


def classify_measurements(radar, lidar, model):
    """Classifies radar/lidar observations.

    This function classifies atmospheric scatterer from the input data. 
    The input data needs to be averaged or interpolated to the common
    time / height grid before calling this function.

    See also:
        classify.fetch_qual_bits()

    """
    obs = _ClassData(radar, lidar, model)
    bits = [None] * 6
    liquid = droplet.find_liquid(obs)
    bits[3] = find_melting_layer(obs)
    bits[2] = find_freezing_region(obs, bits[3])
    bits[0] = droplet.correct_liquid_top(obs, liquid, bits[2])
    bits[5], insect_prob = find_insects(obs, bits[3], bits[0])
    bits[1] = find_falling_hydrometeors(obs, bits[0], bits[5])
    bits[4] = find_aerosols(obs.beta, bits[1], bits[0])
    cat_bits = _bits_to_integer(bits)
    return _ClassificationResult(cat_bits, obs.is_rain, obs.is_clutter,
                                 insect_prob, liquid['bases'])


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
        out1, out2 = arg1[ii, temp_indices], arg2[ii, temp_indices]
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

    for ii, t_prof in enumerate(obs.tw):
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
                    melting_layer[ii, temp_indices[ldr_p]:temp_indices[top]+1] = True
            except:
                try:
                    top, base = _basetop(v_dprof, v_p)
                    diff = v_prof[top] - v_prof[base]
                    if diff > 0.5 and v_prof[base] < -2:
                        melting_layer[ii, temp_indices[v_p-1:v_p+2]] = True
                except:
                    continue
    if smooth:
        ml = scipy.ndimage.filters.gaussian_filter(np.array(melting_layer,
                                                            dtype=float), (2, 0.1))
        melting_layer = (ml > 0.2).astype(bool)
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
    t0_alt = _t0_alt(obs.tw, obs.height)
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


def _t0_alt(tw, height):
    """ Interpolates altitudes where model temperature goes below freezing.

    Args:
        tw (ndarray): 2-D wet bulb temperature.
        height (ndarray): 1-D altitude grid (m).

    Returns:
        ndarray: 1-D array denoting altitudes where the
        temperature drops below 0 deg C.

    """
    alt = np.array([])
    for prof in tw:
        ind = np.where(prof < T0)[0][0]
        if ind == 0:
            alt = np.append(alt, height[0])
        else:
            x = prof[ind-1:ind+1]
            y = height[ind-1:ind+1]
            x, y = zip(*sorted(zip(x, y)))
            alt = np.append(alt, np.interp(T0, x, y))
    return alt


def find_insects(obs, *args, prob_lim=0.8):
    """Returns insect probability and boolean array of insect presence.

    Insects are detected by calculating heuristic probability of insects using
    radar echo, *Zh*, linear depolarization ratio, *ldr* and Doppler
    width, *w*. Generally insects have small *Zh*, high *ldr* and small *w*.

    If a small echo from temperatures above zero has high *ldr*,
    it most probably contains insects. A probability value between 0 and 1
    is assigned using a 2-D probability distribution in *ldr*-*Zh* space.

    If *ldr* is not available, we must use *w* which is not as
    good indicator but still usable. Here a fixed value, 0.06, is used
    (smaller *w* values than this are insects).

    The approach above generally does not give many false positives but instead
    misses a few insect cases. If hordes of insects are present, they can
    yield a relatively strong radar signal. Because this is not a typical
    insect signature, a too low probability will appear.

    Finally, positive insect detections are canceled from profiles with rain,
    liquid droplets pixels, melting layer pixels and too cold temperatures.
    
    Args:
        obs (_ClassData): Input data container.
        *args: Binary fields that are used to screen the
            insect probability. E.g. rain, clutter,
            melting_layer, etc.
        prob_lim (float, optional): Probability higher than
            this will lead to positive detection. Default is 0.8.

    Returns:
        2-element tuple containing

        - ndarray: 2-D probability of pixel containing insects.
        - ndarray: 2-D boolean flag of insects presense.

    """
    i_prob = _insect_probability(obs.z, obs.ldr, obs.width)
    i_prob = _screen_insects(i_prob, obs.tw, *args)
    is_insects = i_prob > prob_lim
    return is_insects, ma.masked_where(i_prob == 0, i_prob)


def _insect_probability(z, ldr, width):
    """Estimates insect probability from radar parameters.

    Args:
        z (MaskedArray): 2-D radar echo.
        ldr (MaskedArray): 2-D radar linear depolarization ratio.
        width (MaskedArray): 2-D radar spectral width.

    Returns:
        ndarray: 2-D insect probability between 0-1.

    """
    def _insect_prob_ldr(z_loc=15, ldr_loc=-20):
        """Finds probability of insects, based on echo and ldr."""
        zp, ldr_prob = np.zeros(z.shape), np.zeros(z.shape)
        ind = ~z.mask
        zp[ind] = stats.norm.cdf(z[ind]*-1, loc=z_loc, scale=8)
        ind = ~ldr.mask
        ldr_prob[ind] = stats.norm.cdf(ldr[ind], loc=ldr_loc, scale=5)
        return zp * ldr_prob

    def _insect_prob_width(w_limit=0.06):
        """Finds (0, 1) probability of insects, based on spectral width."""
        temp_w = np.ones(z.shape)
        ind = ldr.mask & ~z.mask  # pixels that have Z but no LDR
        temp_w[ind] = width[ind]
        return (temp_w < w_limit).astype(int)

    p1 = _insect_prob_ldr()
    p2 = _insect_prob_width()
    return p1 + p2


def _screen_insects(insect_prob, tw, *args):
    """Screens insects by temperature and other misc. conditions.

    Args:
        insect_prob (ndarray): Insect probability with the shape (m, n).
        tw (ndarray): Wet bulb temperature with the shape (m, n).
        *args (ndrray): Variable number of boolean arrays where True
            means that insect probablity should be 0. Shape of these
            fields can be (m, n), or (m,) when the whole profile
            is flagged.

    """
    def _screen_insects_misc():
        """Sets insect probability to 0, indicated by *args."""
        for arg in args:
            if arg.size == prob.shape[0]:
                prob[arg, :] = 0
            else:
                prob[arg] = 0

    def _screen_insects_temp(t_lim=-5):
        prob[tw < (T0+t_lim)] = 0

    prob = np.copy(insect_prob)
    _screen_insects_misc()
    _screen_insects_temp()
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
    is_z = ~obs.z.mask
    no_clutter = ~obs.is_clutter
    no_insects = ~is_insects
    falling_from_lidar = ~obs.beta.mask & (obs.beta.data > 1e-6) & ~is_liquid
    is_falling = (is_z & no_clutter & no_insects) | falling_from_lidar
    return is_falling


def find_aerosols(beta, is_falling, is_liquid):
    """Estimates aerosols from lidar backscattering.

    Aerosols are lidar signals that are: a) not falling, b) not liquid droplets.

    Args:
        beta (MaskedArray): 2-D attenuated backscattering coefficient.
        is_falling (ndarray): 2-D boolean array of falling hydrometeors.
        is_liquid (ndarray): 2-D boolean array of liquid droplets.

    Returns:
        ndarray: 2-D boolean array containing aerosols.

    """
    return ~beta.mask & ~is_falling & ~is_liquid


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
