"""Module containing low-level functions to classify gridded
radar / lidar measurements.
"""
from collections import namedtuple
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy.categorize import droplet
from cloudnetpy.categorize import melting, insects, falling, freezing


def classify_measurements(radar, lidar, model, mwr):
    """Classifies radar/lidar observations.

    This function classifies atmospheric scatterers from the input data.
    The input data needs to be averaged or interpolated to the common
    time / height grid before calling this function.

    Args:
        radar (Radar): The :class:`Radar` instance.
        lidar (Lidar): The :class:`Lidar` instance.
        model (Model): The :class:`Model` instance.
        mwr (Mwr): The :class:`Mwr` instance.

    Returns:
        ClassificationResult:
            The :class:`ClassificationResult` instance.

    References:
        The Cloudnet classification scheme is based on methodology proposed by
        Hogan R. and O'Connor E., 2004, https://bit.ly/2Yjz9DZ and its
        proprietary Matlab implementation.

    Notes:
        Some of the individual classification methods are changed in this Python
        implementation compared to the original Cloudnet methodology.
        Especially methods classifying insects, melting layer and liquid droplets.

    """
    obs = ClassData(radar, lidar, model, mwr)
    bits = [None] * 6
    liquid = droplet.find_liquid(obs)
    bits[3] = melting.find_melting_layer(obs)
    bits[2] = freezing.find_freezing_region(obs, bits[3])
    bits[0] = droplet.correct_liquid_top(obs, liquid, bits[2], limit=500)
    bits[5], insect_prob = insects.find_insects(obs, bits[3], bits[0])
    bits[1] = falling.find_falling_hydrometeors(obs, bits[0], bits[5])
    bits[4] = _find_aerosols(obs, bits[1], bits[0])
    return ClassificationResult(_bits_to_integer(bits),
                                obs.is_rain,
                                obs.is_clutter,
                                insect_prob,
                                liquid['bases'],
                                _find_profiles_with_undetected_melting(bits))


def fetch_quality(radar, lidar, classification, attenuations):
    """Returns Cloudnet quality bits.

    Args:
        radar (Radar): The :class:`Radar` instance.
        lidar (Lidar): The :class:`Lidar` instance.
        classification (ClassificationResult): The
            :class:`ClassificationResult` instance.
        attenuations (dict): Dictionary containing keys `liquid_corrected`,
            `liquid_uncorrected`.

    Returns:
        dict: Dictionary containing `quality_bits`, an integer array with the bits:

            - bit 0: Pixel contains radar data
            - bit 1: Pixel contains lidar data
            - bit 2: Pixel contaminated by radar clutter
            - bit 3: Molecular scattering present (currently not implemented!)
            - bit 4: Pixel was affected by liquid attenuation
            - bit 5: Liquid attenuation was corrected

    """
    bits = [None]*6
    bits[0] = ~radar.data['Z'][:].mask
    bits[1] = ~lidar.data['beta'][:].mask
    bits[2] = classification.is_clutter
    bits[4] = attenuations['liquid_corrected'] | attenuations['liquid_uncorrected']
    bits[5] = attenuations['liquid_corrected']
    qbits = _bits_to_integer(bits)
    return {'quality_bits': qbits}


def _find_aerosols(obs, is_falling, is_liquid):
    """Estimates aerosols from lidar backscattering.

    Aerosols are lidar signals that are: a) not falling, b) not liquid droplets.

    Args:
        obs (ClassData): The :class:`ClassData` instance.
        is_falling (ndarray): 2-D boolean array of falling hydrometeors.
        is_liquid (ndarray): 2-D boolean array of liquid droplets.

    Returns:
        ndarray: 2-D boolean array containing aerosols.

    """
    is_beta = ~obs.beta.mask
    return is_beta & ~is_falling & ~is_liquid


def _find_profiles_with_undetected_melting(bits):
    drizzle_and_falling = _find_drizzle_and_falling(*bits[:3])
    transition = ma.diff(drizzle_and_falling, axis=1)
    is_transition = ma.any(transition, axis=1)
    is_melting_layer = ma.any(bits[3], axis=1)
    is_undetected_melting = is_transition & ~is_melting_layer
    is_undetected_melting[is_undetected_melting == 0] = ma.masked
    return is_undetected_melting.astype(int)


def _find_drizzle_and_falling(is_liquid, is_falling, is_freezing):
    """Classifies pixels as falling, drizzle and others.

    Args:
        is_liquid (ndarray): 2D boolean array denoting liquid layers.
        is_falling (ndarray): 2D boolean array denoting falling pixels.
        is_freezing (ndarray): 2D boolean array denoting subzero temperatures.

    Returns:
        MaskedArray: 2D array where values are 1 (falling), 2 (drizzle), and
            masked (all others).

    """
    falling_dry = is_falling & ~is_liquid
    drizzle = falling_dry & ~is_freezing
    drizzle_and_falling = falling_dry.astype(int) + drizzle.astype(int)
    drizzle_and_falling = ma.copy(drizzle_and_falling)
    drizzle_and_falling[drizzle_and_falling == 0] = ma.masked
    return drizzle_and_falling


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


class ClassData:
    """ Container for observations that are used in the classification.

    Args:
        radar (Radar): The :class:`Radar` instance.
        lidar (Lidar): The :class:`Lidar` instance.
        model (Model): The :class:`Model` instance.
        mwr (Mwr): The :class:`Mwr` instance.

    Attributes:
        z (ndarray): 2D radar echo.
        ldr (ndarray): 2D linear depolarization ratio.
        v (ndarray): 2D radar velocity.
        width (ndarray): 2D radar width.
        v_sigma (ndarray): 2D standard deviation of the velocity.
        tw (ndarray): 2D wet bulb temperature.
        beta (ndarray): 2D lidar backscatter.
        lwp (ndarray): 1D liquid water path.
        time (ndarray): 1D fraction hour.
        height (ndarray): 1D height vector (m).
        model_type (str): Model identifier.
        radar_type (str): Radar identifier.
        is_rain (ndarray): 2D boolean array denoting rain.
        is_clutter (ndarray): 2D boolean array denoting clutter.

    """
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
        self.is_rain = _find_rain(self.z, self.time)
        self.is_clutter = _find_clutter(self.v, self.is_rain)


def _find_rain(z, time, time_buffer=5):
    """Find profiles affected by rain.

    Rain is present in such profiles where the radar echo in
    the third range gate is > 0 dB. To make sure we do not include any
    rainy profiles, we also flag a few profiles before and after
    detections as raining.

    Args:
        z (ndarray): Radar echo.
        time (ndarray): Time vector.
        time_buffer (int): Time in minutes.

    """
    is_rain = ma.array(z[:, 3] > 0, dtype=bool).filled(False)
    n_profiles = len(time)
    n_steps = utils.n_elements(time, time_buffer, 'time')
    for ind in np.where(is_rain)[0]:
        ind1 = max(0, ind - n_steps)
        ind2 = min(ind + n_steps, n_profiles)
        is_rain[ind1:ind2 + 1] = True
    return is_rain


def _find_clutter(v, is_rain, n_gates=10, v_lim=0.05):
    """Estimates clutter from doppler velocity.

        Args:
            n_gates (int, optional): Number of range gates from the ground
                where clutter is expected to be found. Default is 10.
            v_lim (float, optional): Velocity threshold. Smaller values are
                classified as clutter. Default is 0.05 (m/s).

        Returns:
            ndarray: 2-D boolean array denoting pixels contaminated by clutter.

        """
    is_clutter = np.zeros(v.shape, dtype=bool)
    tiny_velocity = (np.abs(v[:, :n_gates]) < v_lim).filled(False)
    is_clutter[:, :n_gates] = tiny_velocity * utils.transpose(~is_rain)
    return is_clutter


class ClassificationResult(namedtuple('ClassificationResult',
                                      ['category_bits',
                                       'is_rain',
                                       'is_clutter',
                                       'insect_prob',
                                       'liquid_bases',
                                       'is_undetected_melting'])):
    """ Result of classification

    Attributes:
        category_bits (ndarray): Array of integers concatenating all the
            individual boolean bit arrays.
        is_rain (ndarray): 1D array denoting presence of rain.
        is_clutter (ndarray): 2D array denoting presence of clutter.
        insect_prob (ndarray): 2D array denoting 0-1 probability of insects.
        liquid_bases (ndarray): 2D array denoting bases of liquid clouds.
        is_undetected_melting (ndarray): 1D array denoting profiles that should
            contain melting layer but was not detected from the data.

    """
