from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import ma


@dataclass
class DisdroL1:
    diameter: npt.NDArray
    diameter_spread: npt.NDArray
    velocity: npt.NDArray
    velocity_spread: npt.NDArray
    time: npt.NDArray
    interval: npt.NDArray
    area: float
    data_raw: npt.NDArray


@dataclass
class DisdroL2:
    diameter: npt.NDArray
    diameter_spread: npt.NDArray
    velocity: npt.NDArray
    velocity_spread: npt.NDArray
    time: npt.NDArray
    interval: npt.NDArray
    sampling_area: float
    data_raw: npt.NDArray
    n_particles: npt.NDArray
    number_concentration: npt.NDArray
    fall_velocity: npt.NDArray
    rainfall_rate: npt.NDArray


def process_l2(l1: DisdroL1) -> DisdroL2:
    n_particles = np.sum(l1.data_raw, axis=(1, 2))
    n_bins = np.sum(l1.data_raw > 0, axis=(1, 2))
    is_valid = (n_particles > 10) & (n_bins >= 3)  # Guyot et al. (2019)
    filtered = np.copy(l1.data_raw)
    filtered[~is_valid] = 0

    number_concentration = np.sum(
        filtered
        / (l1.area * l1.interval * l1.velocity * l1.diameter_spread[:, np.newaxis]),
        axis=2,
    )
    fall_velocity = ma.divide(
        np.sum(l1.velocity * filtered, axis=2), np.sum(filtered, axis=2)
    )
    rainfall_rate = (
        6e-4
        * np.pi
        * np.sum(filtered * l1.diameter[:, np.newaxis] ** 3, axis=(1, 2))
        / (l1.area * l1.interval)
    )

    return DisdroL2(
        diameter=l1.diameter,
        diameter_spread=l1.diameter_spread,
        velocity=l1.velocity,
        velocity_spread=l1.velocity_spread,
        time=l1.time,
        interval=l1.interval,
        sampling_area=l1.area,
        data_raw=l1.data_raw,
        n_particles=n_particles,
        number_concentration=number_concentration,
        fall_velocity=fall_velocity,
        rainfall_rate=rainfall_rate,
    )
