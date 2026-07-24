from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import ma


@dataclass
class DisdroL1:
    diameter: npt.NDArray
    diameter_spread: npt.NDArray
    velocity: npt.NDArray
    # velocity_spread: npt.NDArray
    time: npt.NDArray
    interval: npt.NDArray
    area: float
    data_raw: npt.NDArray


@dataclass
class DisdroL2:
    diameter: npt.NDArray
    diameter_spread: npt.NDArray
    velocity: npt.NDArray
    # velocity_spread: npt.NDArray
    time: npt.NDArray
    interval: npt.NDArray
    area: float
    data_raw: npt.NDArray
    n_particles: npt.NDArray
    number_concentration: npt.NDArray
    fall_velocity: npt.NDArray
    rain_rate: npt.NDArray
    rain_accum: npt.NDArray
    radar_refl: npt.NDArray
    energy_flux: npt.NDArray


def process_l2(l1: DisdroL1) -> DisdroL2:
    n_time = len(l1.time)
    n_particles = l1.data_raw.reshape(n_time, -1).sum(axis=1)
    n_bins = (l1.data_raw > 0).reshape(n_time, -1).sum(axis=1)
    is_valid = (n_particles > 10) & (n_bins >= 3)  # Guyot et al. (2019)
    filtered = np.copy(l1.data_raw)
    filtered[~is_valid] = 0

    area_mm2 = l1.area * 1e6
    interval_h = l1.interval / 3600
    rho_w = 1e-6  # kg mm-3

    if filtered.ndim == 2:
        number_concentration = (
            filtered
            / (l1.velocity * l1.diameter_spread)
            / (l1.area * l1.interval)[:, np.newaxis]
        )
        fall_velocity = ma.masked_where(
            filtered == 0, np.tile(l1.velocity, (n_time, 1))
        )
        rain_amount = np.pi / 6 * np.sum(filtered * l1.diameter**3, axis=1) / area_mm2
        radar_refl = np.sum(filtered * l1.diameter**6 / l1.velocity, axis=1) / (
            l1.area * l1.interval
        )
        energy_flux = (
            np.pi
            / 12
            * rho_w
            * np.sum(filtered * l1.diameter**3 * l1.velocity**2, axis=1)
            / (l1.area * interval_h)
        )
    else:
        number_concentration = (
            np.sum(
                filtered / (l1.velocity * l1.diameter_spread[:, np.newaxis]),
                axis=2,
            )
            / (l1.area * l1.interval)[:, np.newaxis]
        )
        fall_velocity = ma.divide(
            np.sum(l1.velocity * filtered, axis=2), np.sum(filtered, axis=2)
        )
        rain_amount = (
            np.pi
            / 6
            * np.sum(filtered * l1.diameter[:, np.newaxis] ** 3, axis=(1, 2))
            / area_mm2
        )
        radar_refl = np.sum(
            filtered * l1.diameter[:, np.newaxis] ** 6 / l1.velocity, axis=(1, 2)
        ) / (l1.area * l1.interval)
        energy_flux = (
            np.pi
            / 12
            * rho_w
            * np.sum(
                filtered * l1.diameter[:, np.newaxis] ** 3 * l1.velocity**2, axis=(1, 2)
            )
            / (l1.area * interval_h)
        )
    rain_rate = rain_amount / interval_h
    rain_accum = np.cumsum(rain_amount)
    radar_refl_db = 10 * ma.log10(radar_refl)
    radar_refl_db[radar_refl_db < -10] = ma.masked

    return DisdroL2(
        diameter=l1.diameter,
        diameter_spread=l1.diameter_spread,
        velocity=l1.velocity,
        # velocity_spread=l1.velocity_spread,
        time=l1.time,
        interval=l1.interval,
        area=l1.area,
        data_raw=l1.data_raw,
        n_particles=n_particles,
        number_concentration=number_concentration,
        fall_velocity=fall_velocity,
        rain_rate=rain_rate,
        rain_accum=rain_accum,
        radar_refl=radar_refl_db,
        energy_flux=energy_flux,
    )
