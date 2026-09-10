import logging
from collections.abc import Container

import numpy as np
import numpy.typing as npt
from atmoslib import isa_air_density
from atmoslib.constants import RHO_STD
from numpy import ma


def convert_to_numpy(
    data: dict,
    fill_values: dict | None = None,
    int_keys: Container | None = None,
    float_keys: Container | None = None,
) -> dict:
    output = {}
    for key, value in data.items():
        arr = np.array(value)
        if fill_values is not None and key in fill_values:
            arr = ma.masked_where(arr == fill_values[key], arr)
        if int_keys is not None and key in int_keys:
            new_arr = arr.astype(np.int32)
            if not np.all(arr == new_arr):
                msg = "Cannot convert non-integer float to integer"
                raise ValueError(msg)
            arr = new_arr
        elif float_keys is not None and key in float_keys:
            arr = arr.astype(np.float32)
        output[key] = arr
    return output


def make_rain_mask(
    diameter: npt.NDArray[np.floating],
    velocity: npt.NDArray[np.floating],
    threshold: float = 0.5,
    altitude: float | None = None,
) -> npt.NDArray[np.bool]:
    """Make mask for rain bins based on theoretical diameter-velocity relation.

    Args:
        diameter: Diameter (mm)
        velocity: Velocity (m s-1)
        threshold: Maximum error to theoretical relation (1)
        altitude: Site altitude for optional altitude correction (m)

    References:
        Atlas, D., Srivastava, R. C., & Sekhon, R. S. (1973). Doppler Radar
        Characteristics of Precipitation at Vertical Incidence. Reviews of
        Geophysics and Space Physics, 11(1), 1-35.
        https://doi.org/10.1029/RG011i001p00001

        Beard, K. V. (1985). Simple Altitude Adjustments to Raindrop Velocities
        for Doppler Radar Analysis. Journal of Atmospheric and Oceanic
        Technology, 2(4), 468-471.
        https://doi.org/10.1175/1520-0426(1985)002%3C0468:SAATRV%3E2.0.CO;2
    """
    v = 9.65 - 10.3 * np.exp(-0.6 * diameter)  # Eq. 24, Atlas et al. (1973)
    if altitude is None:
        logging.warning(
            "No altitude given, no correction applied for non-sea-level conditions."
        )
    else:
        rho = isa_air_density(altitude)
        m = 0.375 + 0.025 * diameter  # Eq. 3, Beard (1985)
        v = (RHO_STD / rho) ** m * v  # Eq. 1, Beard (1985)
    e = np.abs((velocity - v[:, np.newaxis]) / v[:, np.newaxis])
    return e < threshold
