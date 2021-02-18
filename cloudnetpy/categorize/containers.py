from collections import namedtuple
from typing import Optional
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils


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


class ClassData:
    """ Container for observations that are used in the classification.

    Args:
        data: Containing :class:`Radar`, :class:`Lidar`, :class:`Model` and :class:`Mwr` instances.

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
    def __init__(self, data: dict):
        self.z = data['radar'].data['Z'][:]
        self.ldr = data['radar'].data['ldr'][:]
        self.v = data['radar'].data['v'][:]
        self.width = data['radar'].data['width'][:]
        self.v_sigma = data['radar'].data['v_sigma'][:]
        self.tw = data['model'].data['Tw'][:]
        self.beta = data['lidar'].data['beta'][:]
        self.lwp = data['mwr'].data['lwp'][:]
        self.time = data['radar'].time
        self.height = data['radar'].height
        self.model_type = data['model'].type
        self.radar_type = data['radar'].type
        self.is_rain = _find_rain(self.z, self.time)
        self.is_clutter = _find_clutter(self.v, self.is_rain)


def _find_rain(z: np.ndarray,
               time: np.ndarray,
               time_buffer: Optional[int] = 5) -> np.ndarray:
    """Find profiles affected by rain.

    Rain is present in such profiles where the radar echo in
    the third range gate is > 0 dB. To make sure we do not include any
    rainy profiles, we also flag a few profiles before and after
    detections as raining.

    Args:
        z: Radar echo.
        time: Time vector.
        time_buffer: Time in minutes.

    Returns:
        1D Boolean array denoting profiles with rain.

    """
    is_rain = ma.array(z[:, 3] > 0, dtype=bool).filled(False)
    n_profiles = len(time)
    n_steps = utils.n_elements(time, time_buffer, 'time')
    for ind in np.where(is_rain)[0]:
        ind1 = max(0, ind - n_steps)
        ind2 = min(ind + n_steps, n_profiles)
        is_rain[ind1:ind2 + 1] = True
    return is_rain


def _find_clutter(v: np.ndarray,
                  is_rain: np.ndarray,
                  n_gates: Optional[int] = 10,
                  v_lim: Optional[float] = 0.05) -> np.ndarray:
    """Estimates clutter from doppler velocity.

        Args:
            n_gates: Number of range gates from the ground where clutter is expected to be found.
                Default is 10.
            v_lim: Velocity threshold. Smaller values are classified as clutter.
                Default is 0.05 (m/s).

        Returns:
            2-D boolean array denoting pixels contaminated by clutter.

        """
    is_clutter = np.zeros(v.shape, dtype=bool)
    tiny_velocity = (np.abs(v[:, :n_gates]) < v_lim).filled(False)
    is_clutter[:, :n_gates] = tiny_velocity * utils.transpose(~is_rain)
    return is_clutter
