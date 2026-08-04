from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy.disdronator.utils import make_rain_mask


@dataclass
class DisdroL1:
    diameter: npt.NDArray
    diameter_bins: npt.NDArray
    diameter_spread: npt.NDArray
    velocity: npt.NDArray
    velocity_bins: npt.NDArray
    velocity_spread: npt.NDArray
    time: npt.NDArray
    interval: npt.NDArray
    area_nom: float
    area_eff: npt.NDArray | None
    data_raw: npt.NDArray


@dataclass
class DisdroL2:
    diameter: npt.NDArray
    diameter_bins: npt.NDArray
    diameter_spread: npt.NDArray
    velocity: npt.NDArray
    velocity_bins: npt.NDArray
    velocity_spread: npt.NDArray
    time: npt.NDArray
    interval: npt.NDArray
    area_nom: float
    area_eff: npt.NDArray | None
    data_raw: npt.NDArray
    n_particles: npt.NDArray
    number_concentration: npt.NDArray
    fall_velocity: npt.NDArray
    rain_rate: npt.NDArray
    rain_accum: npt.NDArray
    radar_refl: npt.NDArray
    energy_flux: npt.NDArray
    visibility: npt.NDArray


def process_l2(l1: DisdroL1) -> DisdroL2:
    time, time_ind = np.unique(l1.time, return_index=True)
    data_raw = l1.data_raw[time_ind]
    interval = l1.interval[time_ind]

    n_time = len(time)
    n_diameter = len(l1.diameter)
    n_particles = data_raw.reshape(n_time, -1).sum(axis=1)
    n_bins = (data_raw > 0).reshape(n_time, -1).sum(axis=1)
    is_valid = (n_particles > 10) & (n_bins >= 3)  # Guyot et al. (2019)
    spec = np.copy(data_raw)
    spec[~is_valid] = 0

    area_mm2 = (
        l1.area_eff if l1.area_eff is not None else np.repeat(l1.area_nom, n_diameter)
    )
    area_m2 = area_mm2 * 1e-6
    diameter_m = l1.diameter / 1000
    interval_h = interval / 3600
    rho_w = 1e-6  # kg mm-3

    if spec.ndim == 2:
        number_concentration = spec / (
            l1.velocity * l1.diameter_spread * area_m2 * interval[:, np.newaxis]
        )
        fall_velocity = ma.masked_where(spec == 0, np.tile(l1.velocity, (n_time, 1)))
        rain_amount = np.pi / 6 * np.sum(spec * l1.diameter**3 / area_mm2, axis=1)
        radar_refl = np.sum(
            spec * l1.diameter**6 / (l1.velocity * area_m2 * interval_h[:, np.newaxis]),
            axis=1,
        )
        energy_flux = (
            np.pi
            / 12
            * rho_w
            * np.sum(
                spec
                * l1.diameter**3
                * l1.velocity**2
                / (area_m2 * interval_h[:, np.newaxis]),
                axis=1,
            )
        )
        extinction = (
            np.pi
            / 2
            * np.sum(
                spec
                * diameter_m**2
                / (l1.velocity * area_m2 * interval[:, np.newaxis]),
                axis=1,
            )
        )
    else:
        number_concentration = np.sum(
            spec
            / (
                l1.velocity
                * l1.diameter_spread[:, np.newaxis]
                * area_m2[:, np.newaxis]
                * interval[:, np.newaxis, np.newaxis]
            ),
            axis=2,
        )
        fall_velocity = ma.divide(
            np.sum(l1.velocity * spec, axis=2), np.sum(spec, axis=2)
        )
        is_rain = make_rain_mask(l1.diameter, l1.velocity)
        spec_rain = np.copy(spec)
        spec_rain[:, ~is_rain] = 0
        rain_amount = (
            np.pi
            / 6
            * np.sum(
                spec_rain * l1.diameter[:, np.newaxis] ** 3 / area_mm2[:, np.newaxis],
                axis=(1, 2),
            )
        )
        radar_refl = np.sum(
            spec_rain
            * l1.diameter[:, np.newaxis] ** 6
            / (
                l1.velocity
                * area_m2[:, np.newaxis]
                * interval[:, np.newaxis, np.newaxis]
            ),
            axis=(1, 2),
        )
        energy_flux = (
            np.pi
            / 12
            * rho_w
            * np.sum(
                spec_rain
                * l1.diameter[:, np.newaxis] ** 3
                * l1.velocity**2
                / (area_m2[:, np.newaxis] * interval_h[:, np.newaxis, np.newaxis]),
                axis=(1, 2),
            )
        )
        extinction = (
            np.pi
            / 2
            * np.sum(
                spec_rain
                * diameter_m[:, np.newaxis] ** 2
                / (
                    l1.velocity
                    * area_m2[:, np.newaxis]
                    * interval[:, np.newaxis, np.newaxis]
                ),
                axis=(1, 2),
            )
        )
    rain_rate = rain_amount / interval_h
    rain_accum = np.cumsum(rain_amount)

    min_radar_refl = -10
    radar_refl_db = 10 * ma.log10(radar_refl)
    radar_refl_db[radar_refl_db < min_radar_refl] = ma.masked

    max_visibility = 2e4
    visibility = ma.divide(3, extinction)
    visibility = np.minimum(ma.filled(visibility, max_visibility), max_visibility)

    return DisdroL2(
        diameter=l1.diameter,
        diameter_bins=l1.diameter_bins,
        diameter_spread=l1.diameter_spread,
        velocity=l1.velocity,
        velocity_bins=l1.velocity_bins,
        velocity_spread=l1.velocity_spread,
        time=time,
        interval=interval,
        area_nom=l1.area_nom,
        area_eff=l1.area_eff,
        data_raw=data_raw,
        n_particles=n_particles,
        number_concentration=number_concentration,
        fall_velocity=fall_velocity,
        rain_rate=rain_rate,
        rain_accum=rain_accum,
        radar_refl=radar_refl_db,
        energy_flux=energy_flux,
        visibility=visibility,
    )
