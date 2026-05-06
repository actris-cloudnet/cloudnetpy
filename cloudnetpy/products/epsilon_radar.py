"""Module for creating Cloudnet eddy dissipation rate product, based on the
pipeline of Griesche et al. (2020) with the inertial-subrange slope-acceptance
criterion of Borque et al. (2016).
"""

from collections.abc import Callable
from os import PathLike
from uuid import UUID

import numpy as np
import numpy.typing as npt
from numpy import ma
from scipy.interpolate import CubicSpline, RectBivariateSpline

from cloudnetpy import output, utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData
from cloudnetpy.utils import get_uuid


def generate_epsilon_from_radar(
    radar_file: str | PathLike,
    model_file: str | PathLike,
    output_file: str | PathLike,
    uuid: str | UUID | None = None,
) -> UUID:
    """Generates Cloudnet radar-based dissipation rate of turbulent kinetic
    energy product.

    Based on the pipeline of Griesche et al. (2020) with the inertial-subrange
    slope-acceptance criterion of Borque et al. (2016).

    Args:
        radar_file: Cloud radar L1b file name (provides Doppler velocity).
        model_file: Cloudnet model file name (provides horizontal wind).
        output_file: Output file name.
        uuid: Set specific UUID for the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_epsilon_from_radar
        >>> generate_epsilon_from_radar('radar.nc', 'ecmwf.nc', 'epsilon.nc')

    References:
        Griesche, H. J., Seifert, P., Ansmann, A., Baars, H., Barrientos
        Velasco, C., Bühl, J., Engelmann, R., Radenz, M., Zhenping, Y., and
        Macke, A. (2020): Application of the shipborne remote sensing
        supersite OCEANET for profiling of Arctic aerosols and clouds during
        Polarstern cruise PS106, Atmos. Meas. Tech., 13, 5335-5358,
        https://doi.org/10.5194/amt-13-5335-2020.

        Borque, P., Luke, E., and Kollias, P. (2016): On the unified
        estimation of turbulence eddy dissipation rate using Doppler cloud
        radars and lidars, J. Geophys. Res. Atmos., 120, 5972-5989,
        https://doi.org/10.1002/2015JD024543.
    """
    uuid = get_uuid(uuid)
    with (
        EpsilonRadarSource(radar_file) as epsilon_source,
        DataSource(model_file) as model_source,
    ):
        if epsilon_source.altitude is None:
            msg = "Radar file is missing 'altitude' attribute."
            raise ValueError(msg)
        wind_interp = _get_wind_interpolator(
            model_source, alt_site=float(epsilon_source.altitude)
        )
        epsilon_source.append_epsilon(wind_interp)
        epsilon_source.append_grid_variables()
        date = epsilon_source.get_date()
        attributes = output.add_time_attribute(EPSILON_RADAR_ATTRIBUTES, date)
        output.update_attributes(epsilon_source.data, attributes)
        output.save_product_file(
            "epsilon-radar",
            epsilon_source,
            output_file,
            uuid,
            extra_sources=(model_source,),
        )
    return uuid


def _get_wind_interpolator(
    model: DataSource, alt_site: float
) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
    """Returns a bilinear interpolator for horizontal wind speed on
    (time, height_amsl) where height is meters above MSL.

    Model heights are above the model's own surface; shift to absolute MSL so
    they share the radar's height frame. In complex terrain the model grid
    cell surface can differ from the site altitude, so prefer the model's own
    surface field when available.
    """
    uwind = model.getvar("uwind")
    vwind = model.getvar("vwind")
    surface_altitude = utils.get_model_surface_altitude(model.dataset, alt_site)
    heights = model.to_m(model.dataset.variables["height"]) + surface_altitude
    wind_speed = np.hypot(np.asarray(uwind), np.asarray(vwind))
    mean_height = np.asarray(ma.mean(heights, axis=0))
    common = np.empty((wind_speed.shape[0], mean_height.size))
    for i in range(wind_speed.shape[0]):
        common[i] = np.interp(mean_height, np.asarray(heights[i]), wind_speed[i])
    return RectBivariateSpline(model.time, mean_height, common, kx=1, ky=1)


class EpsilonRadarSource(DataSource):
    """Reads radar Doppler velocity and computes turbulent kinetic energy
    dissipation rate (epsilon) on the product grid.
    """

    height: npt.NDArray

    def append_epsilon(
        self, wind_interp: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
    ) -> None:
        """Estimate the dissipation rate (epsilon) on the 30 s product grid."""
        radar_time = np.asarray(self.getvar("time"))
        self._radar_time = radar_time
        height = self.height
        radar_dt_sec = float(np.round(np.median(np.diff(radar_time)) * 3600.0))

        v = self.getvar("v")
        if isinstance(v, ma.MaskedArray):
            v = v.filled(np.nan)
        v = np.asarray(v, dtype=np.float64)

        product_time = utils.time_grid(time_step=PRODUCT_TIME_STEP_SEC)
        self.time = product_time
        wind_at_product = wind_interp(product_time, height)

        n_time = product_time.size
        n_height = height.size
        epsilon = np.full((n_time, n_height), EPSILON_INVALID, dtype=np.float64)
        epsilon_error = np.full((n_time, n_height), EPSILON_INVALID, dtype=np.float64)

        starts = np.searchsorted(radar_time, product_time, side="right")
        stops = np.searchsorted(
            radar_time, product_time + AVERAGING_TIME_HR, side="right"
        )

        for time_idx in range(n_time):
            after, stop = int(starts[time_idx]), int(stops[time_idx])
            if stop <= after:
                continue
            time_window = radar_time[after:stop]
            min_valid = max(2, int(MIN_VALID_FRACTION * time_window.size))

            for height_idx in range(n_height):
                vel = v[after:stop, height_idx]
                nan_mask = np.isnan(vel)
                if nan_mask[:2].all() or nan_mask[-2:].all():
                    continue
                if (vel.size - int(nan_mask.sum())) < min_valid:
                    continue
                if nan_mask.any():
                    valid = ~nan_mask
                    vel = CubicSpline(time_window[valid], vel[valid])(time_window)

                wind_speed = float(wind_at_product[time_idx, height_idx])
                if not np.isfinite(wind_speed) or wind_speed <= 0:
                    continue

                freq_sp, power_sp = _periodogram(vel, radar_dt_sec, wind_speed)
                if freq_sp.size < 2:
                    continue

                result = _epsilon_from_spectrum(freq_sp, power_sp)
                if result is not None:
                    eps, eps_err = result
                    epsilon[time_idx, height_idx] = eps
                    epsilon_error[time_idx, height_idx] = eps_err

        self.append_data(
            ma.masked_where(epsilon == EPSILON_INVALID, epsilon), "epsilon"
        )
        self.append_data(
            ma.masked_where(
                (epsilon_error == EPSILON_INVALID) | ~np.isfinite(epsilon_error),
                epsilon_error,
            ),
            "epsilon_error",
        )

    def append_grid_variables(self) -> None:
        """Adds time/height/altitude/latitude/longitude on the product grid.

        altitude/latitude/longitude are always written as 1-D arrays on the
        product time grid: scalars from stationary radars are broadcast,
        time-varying fields from moving platforms are rebinned.
        """
        self.append_data(self.time, "time", dtype="f8")
        self.append_data(np.asarray(self.height, dtype=np.float32), "height")
        for key in ("altitude", "latitude", "longitude"):
            if key not in self.dataset.variables:
                continue
            src = np.asarray(self.dataset.variables[key][:])
            if src.ndim == 0:
                values = np.full(self.time.size, float(src), dtype=np.float32)
            else:
                values = utils.rebin_1d(self._radar_time, src, self.time)
            self.append_data(values, key)


def _periodogram(
    vel: npt.NDArray, delta_t_sec: float, adv_vel: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute angular wavenumber spectrum from a Doppler velocity series."""
    n = vel.size
    delta_x = delta_t_sec * adv_vel
    fft_result = np.fft.rfft(vel)[1:]
    power_sp = (delta_x / n) * np.abs(fft_result) ** 2 / np.pi
    freq_sp = np.fft.rfftfreq(n, d=delta_x)[1:] * 2.0 * np.pi
    keep = power_sp > 0
    return freq_sp[keep], power_sp[keep]


def _epsilon_from_spectrum(
    freq_sp: npt.NDArray,
    power_sp: npt.NDArray,
) -> tuple[float, float] | None:
    """Vectorised slope fit on log-log spectrum across all frequency bands.

    Uses cumulative sums over a sorted spectrum so each band's least-squares
    fit reduces to a constant-time index lookup. Returns ``(mean, std)`` of
    epsilon across accepted bands -- the band-spread used by Griesche et al.
    (2020) as the random retrieval uncertainty -- or ``None`` if no band
    passes the slope filter. ``std`` is NaN when only a single band passes.
    """
    log_f = np.log10(freq_sp)
    log_p = np.log10(power_sp)

    csx = np.concatenate(([0.0], np.cumsum(log_f)))
    csy = np.concatenate(([0.0], np.cumsum(log_p)))
    csxx = np.concatenate(([0.0], np.cumsum(log_f * log_f)))
    csxy = np.concatenate(([0.0], np.cumsum(log_f * log_p)))

    lo = np.searchsorted(freq_sp, _FMIN_ARR, side="right")
    hi = np.searchsorted(freq_sp, _FMAX_ARR, side="left")
    n = hi - lo

    sx = csx[hi] - csx[lo]
    sy = csy[hi] - csy[lo]
    sxx = csxx[hi] - csxx[lo]
    sxy = csxy[hi] - csxy[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n

    valid = (
        (n >= 2)
        & np.isfinite(slope)
        & (slope > THRESHOLD_SLOPE_MIN)
        & (slope < THRESHOLD_SLOPE_MAX)
    )
    if not valid.any():
        return None
    epsilon = (10.0 ** intercept[valid] / KOLMOGOROV_CONSTANT) ** 1.5
    std = float(np.std(epsilon, ddof=1)) if epsilon.size >= 2 else float("nan")
    return float(np.mean(epsilon)), std


KOLMOGOROV_CONSTANT = 0.5
THRESHOLD_SLOPE_MIN = -2.0  # -5/3 - 20% (Borque, 2016)
THRESHOLD_SLOPE_MAX = -1.33  # -5/3 + 20% (Borque, 2016)
AVERAGING_TIME_HR = 5.0 / 60.0
PRODUCT_TIME_STEP_SEC = 30
MIN_VALID_FRACTION = 0.9
EPSILON_INVALID = -999.0


COMMENTS = {
    "general": (
        "This dataset contains the dissipation rate of turbulent kinetic\n"
        "energy calculated using the pipeline of Griesche et al. (2020)\n"
        "with the inertial-subrange slope-acceptance criterion of Borque\n"
        "et al. (2016). The turbulent energy spectrum is derived as the\n"
        "power spectrum of cloud radar Doppler velocity over 5-minute\n"
        "windows and converted to a wavenumber spectrum using model\n"
        "horizontal wind via Taylor's hypothesis. The dissipation rate is\n"
        "recovered from the intercept of a log-log fit across multiple\n"
        "frequency bands; bands whose slope falls outside -5/3 +/- 20% are\n"
        "rejected and the accepted bands are averaged.\n"
    ),
    "epsilon": (
        "This variable was calculated for profiles where 5 minutes of\n"
        "continuous cloud radar Doppler velocity was available. With less\n"
        "than 10% missing data, the missing values were filled by cubic\n"
        "spline interpolation.\n"
    ),
    "epsilon_error": (
        "Random uncertainty estimated as the standard deviation of epsilon\n"
        "across the frequency bands accepted by the inertial-subrange\n"
        "slope criterion (Griesche et al. 2020). It captures the scatter\n"
        "introduced by the multi-band fit but does not include systematic\n"
        "contributions from horizontal wind uncertainty or the Kolmogorov\n"
        "constant. The variable is masked where only a single band passed\n"
        "the slope filter and no spread can be estimated.\n"
    ),
}


EPSILON_RADAR_ATTRIBUTES = {
    "comment": COMMENTS["general"],
    "epsilon": MetaData(
        long_name="Dissipation rate of turbulent kinetic energy",
        units="m2 s-3",
        ancillary_variables="epsilon_error",
        comment=COMMENTS["epsilon"],
        dimensions=("time", "height"),
    ),
    "epsilon_error": MetaData(
        long_name="Absolute error in dissipation rate of turbulent kinetic energy",
        units="m2 s-3",
        comment=COMMENTS["epsilon_error"],
        dimensions=("time", "height"),
    ),
    "height": COMMON_ATTRIBUTES["height"]._replace(dimensions=("height",)),
}


# Frequency-range pairs (rad/m) used to fit the inertial subrange slope.
# Adapted from Griesche et al. 2020 with a multi-decade extension.
freq_array = np.array(
    [
        [5e-3, 1.5e-2],
        [5e-3, 3e-2],
        [5e-3, 5e-2],
        [8e-3, 2e-2],
        [8e-3, 4e-2],
        [8e-3, 6e-2],
        [2e-2, 7e-2],
        [2e-2, 1e-1],
        [2e-2, 1.5e-1],
        [3e-2, 1e-1],
        [3e-2, 2e-1],
        [3e-2, 3e-1],
        [5e-2, 1.5e-1],
        [5e-2, 3e-1],
        [5e-2, 5e-1],
        [8e-2, 2e-1],
        [8e-2, 4e-1],
        [8e-2, 6e-1],
        [2e-1, 7e-1],
        [2e-1, 1e0],
        [2e-1, 1.5e0],
        [3e-1, 1e0],
        [3e-1, 2e0],
        [3e-1, 3e0],
        [5e-1, 1.5e0],
        [5e-1, 3e0],
        [5e-1, 5e0],
        [8e-1, 2e0],
        [8e-1, 4e0],
        [8e-1, 6e0],
        [2e0, 7e0],
        [2e0, 1e1],
        [2e0, 1.5e1],
        [3e0, 1e1],
        [3e0, 2e1],
        [3e0, 3e1],
        [5e0, 1.5e1],
        [5e0, 3e1],
        [5e0, 5e1],
        [8e0, 2e1],
        [8e0, 4e1],
        [8e0, 6e1],
    ]
)
_FMIN_ARR = freq_array[:, 0]
_FMAX_ARR = freq_array[:, 1]
