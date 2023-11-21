from dataclasses import dataclass

import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.constants import MM_H_TO_M_S


@dataclass
class ClassificationResult:
    """Result of classification"""

    category_bits: np.ndarray
    is_rain: np.ndarray
    is_clutter: np.ndarray
    insect_prob: np.ndarray
    liquid_prob: np.ndarray | None


class ClassData:
    """Container for observations that are used in the classification.

    Args:
        data: Containing :class:`Radar`, :class:`Lidar`, :class:`Model`
            and :class:`Mwr` instances.

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
        altitude: site altitude.

    """

    def __init__(self, data: dict):
        self.data = data
        self.z = data["radar"].data["Z"][:]
        self.v = data["radar"].data["v"][:]
        self.v_sigma = data["radar"].data["v_sigma"][:]
        for key in ("width", "ldr", "sldr"):
            if key in data["radar"].data:
                setattr(self, key, data["radar"].data[key][:])
        self.time = data["radar"].time
        self.height = data["radar"].height
        self.radar_type = data["radar"].source_type
        self.tw = data["model"].data["Tw"][:]
        self.model_type = data["model"].source_type
        self.beta = data["lidar"].data["beta"][:]
        self.lwp = data["mwr"].data["lwp"][:]
        self.is_rain = self._find_profiles_with_rain()
        self.is_clutter = _find_clutter(self.v, self.is_rain)
        self.altitude = data["radar"].altitude
        self.lv0_files = data["lv0_files"]
        self.date = data["radar"].get_date()

    def _find_profiles_with_rain(self) -> np.ndarray:
        is_rain = self._find_rain_from_radar_echo()
        rain_from_disdrometer = self._find_rain_from_disdrometer()
        ind = ~rain_from_disdrometer.mask
        is_rain[ind] = rain_from_disdrometer[ind]
        return is_rain

    def _find_rain_from_radar_echo(self) -> np.ndarray:
        gate_number = 3
        threshold = 0
        z = self.z[:, gate_number]
        return np.where((~ma.getmaskarray(z)) & (z > threshold), 1, 0)

    def _find_rain_from_disdrometer(self) -> ma.MaskedArray:
        threshold_mm_h = 0.25  # Standard threshold for drizzle -> rain
        threshold_particles = 10  # This is arbitrary and should be better tested
        threshold_rate = threshold_mm_h * MM_H_TO_M_S
        try:
            rainfall_rate = self.data["disdrometer"].data["rainfall_rate"].data
            n_particles = self.data["disdrometer"].data["n_particles"].data
            is_rain = ma.array(
                (rainfall_rate > threshold_rate) & (n_particles > threshold_particles),
                dtype=int,
            )
        except (AttributeError, KeyError):
            is_rain = ma.masked_all(self.time.shape, dtype=int)
        return is_rain


def _find_clutter(
    v: np.ma.MaskedArray,
    is_rain: np.ndarray,
    n_gates: int = 10,
    v_lim: float = 0.05,
) -> np.ndarray:
    """Estimates clutter from doppler velocity.

    Args:
        n_gates: Number of range gates from the ground where clutter is expected
            to be found. Default is 10.
        v_lim: Velocity threshold. Smaller values are classified as clutter.
            Default is 0.05 (m/s).

    Returns:
        2-D boolean array denoting pixels contaminated by clutter.

    """
    is_clutter = np.zeros(v.shape, dtype=bool)
    filled = False
    tiny_velocity = (np.abs(v[:, :n_gates]) < v_lim).filled(filled)
    is_clutter[:, :n_gates] = tiny_velocity * utils.transpose(~is_rain)
    return is_clutter
