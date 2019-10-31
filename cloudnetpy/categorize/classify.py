"""Module containing low-level functions to classify gridded
radar / lidar measurements.
"""
from collections import namedtuple
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from cloudnetpy import utils
from cloudnetpy.categorize import droplet, atmos
from cloudnetpy.constants import T0
from cloudnetpy.categorize import melting, insects


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
    bits[3] = melting.find_melting_layer(obs)
    bits[2] = find_freezing_region(obs, bits[3])
    bits[0] = droplet.correct_liquid_top(obs, liquid, bits[2], limit=500)
    bits[5], insect_prob = insects.find_insects(obs, bits[3], bits[0])
    bits[1] = find_falling_hydrometeors(obs, bits[0], bits[5])
    bits[4] = find_aerosols(obs, bits[1], bits[0])
    return ClassificationResult(_bits_to_integer(bits),
                                obs.is_rain,
                                obs.is_clutter,
                                insect_prob,
                                liquid['bases'],
                                find_profiles_with_undetected_melting(bits))


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


def find_falling_hydrometeors(obs, is_liquid, is_insects):
    """Finds falling hydrometeors.

    Falling hydrometeors are radar signals that are
    a) not insects b) not clutter. Furthermore, falling hydrometeors
    are strong lidar pixels excluding liquid layers (thus these pixels
    are ice or rain). They are also weak radar signals in very cold
    temperatures.

    Args:
        obs (_ClassData): Container for observations.
        is_liquid (ndarray): 2-D boolean array of liquid droplets.
        is_insects (ndarray): 2-D boolean array of insects.

    Returns:
        ndarray: 2-D boolean array containing falling hydrometeors.

    """
    def _find_falling_from_radar():
        is_z = ~obs.z.mask
        no_clutter = ~obs.is_clutter
        no_insects = ~is_insects
        return is_z & no_clutter & no_insects

    def _find_falling_from_lidar():
        strong_beta_limit = 2e-6
        return (obs.beta.data > strong_beta_limit) & ~is_liquid

    def _find_cold_aerosols():
        """Lidar signals which are in colder than the
        threshold temperature and have gap below in the profile
        are probably ice."""
        temperature_limit = T0 - 15
        is_beta = ~obs.beta.mask
        region = utils.ffill(is_beta, 1) == 0
        return is_beta & (obs.tw < temperature_limit) & ~is_liquid & region

    def _fix_liquid_dominated_radar():
        """Radar signals inside liquid clouds are NOT ice if Z in cloud is
        increasing in height."""

        def _is_z_missing_above_liquid():
            if top == obs.z.shape[1] - 1:
                return False
            return obs.z.mask[n, top+1]

        def _is_z_increasing():
            z = obs.z[n, base+1:top].compressed()
            if len(z) > 1:
                return z[-1] > z[0]
            return False

        liquid_bases = atmos.find_cloud_bases(is_liquid)
        liquid_tops = atmos.find_cloud_tops(is_liquid)
        base_indices = np.where(liquid_bases)
        top_indices = np.where(liquid_tops)

        for n, base, _, top in zip(*base_indices, *top_indices):
            if _is_z_missing_above_liquid() and _is_z_increasing():
                falling_from_radar[n, base:top+1] = False

    falling_from_radar = _find_falling_from_radar()
    _fix_liquid_dominated_radar()
    falling_from_lidar = _find_falling_from_lidar()
    cold_aerosols = _find_cold_aerosols()
    return falling_from_radar | falling_from_lidar | cold_aerosols


def find_aerosols(obs, is_falling, is_liquid):
    """Estimates aerosols from lidar backscattering.

    Aerosols are lidar signals that are: a) not falling, b) not liquid droplets.

    Args:
        obs (_ClassData): Container for observations.
        is_falling (ndarray): 2-D boolean array of falling hydrometeors.
        is_liquid (ndarray): 2-D boolean array of liquid droplets.

    Returns:
        ndarray: 2-D boolean array containing aerosols.

    """
    is_beta = ~obs.beta.mask
    return is_beta & ~is_falling & ~is_liquid


def find_profiles_with_undetected_melting(bits):
    is_falling = bits[1] & ~bits[0]
    is_drizzle = is_falling & ~bits[2]
    drizzle_and_falling = is_falling.astype(int) + is_drizzle.astype(int)
    drizzle_and_falling[drizzle_and_falling == 0] = ma.masked
    transition = ma.diff(drizzle_and_falling, axis=1)
    is_transition = ma.any(transition, axis=1)
    is_melting_layer = ma.any(bits[3], axis=1)
    is_undetected_melting = is_transition & ~is_melting_layer
    is_undetected_melting[is_undetected_melting == 0] = ma.masked
    return is_undetected_melting.astype(int)


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
        self.radar_type = radar.type
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


ClassificationResult = namedtuple('ClassificationResult', ['category_bits',
                                                           'is_rain',
                                                           'is_clutter',
                                                           'insect_prob',
                                                           'liquid_bases',
                                                           'is_undetected_melting'])
