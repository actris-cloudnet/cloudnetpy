import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy import ma
from scipy.ndimage import gaussian_filter

from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.utils import Epoch

if TYPE_CHECKING:
    from cloudnetpy.instruments.instruments import Instrument


class NoiseParam(NamedTuple):
    """Noise parameters. Values are weakly instrument-dependent."""

    noise_min: float = 1e-9
    noise_smooth_min: float = 4e-9


class Ceilometer:
    """Base class for all types of ceilometers and pollyxt."""

    def __init__(self, noise_param: NoiseParam | None = None):
        self.noise_param = noise_param or NoiseParam()
        self.data: dict = {}  # Need to contain 'beta_raw', 'range' and 'time'
        self.metadata: dict = {}  # Need to contain 'date' as ('yyyy', 'mm', 'dd')
        self.expected_date: str | None = None
        self.site_meta: dict = {}
        self.date: list[str] = []
        self.instrument: Instrument | None = None
        self.serial_number: str | None = None

    def calc_screened_product(
        self,
        array: np.ndarray,
        snr_limit: int = 5,
        n_negatives: int = 5,
        *,
        range_corrected: bool = True,
    ) -> np.ndarray:
        """Screens noise from lidar variable."""
        noisy_data = NoisyData(
            self.data,
            self.noise_param,
            range_corrected=range_corrected,
        )
        return noisy_data.screen_data(
            array,
            snr_limit=snr_limit,
            n_negatives=n_negatives,
        )

    def calc_beta_smooth(
        self,
        beta: np.ndarray,
        snr_limit: int = 5,
        n_negatives: int = 5,
        *,
        range_corrected: bool = True,
    ) -> np.ndarray:
        noisy_data = NoisyData(
            self.data,
            self.noise_param,
            range_corrected=range_corrected,
        )
        beta_raw = ma.copy(self.data["beta_raw"])
        cloud_ind, cloud_values, cloud_limit = _estimate_clouds_from_beta(beta)
        beta_raw[cloud_ind] = cloud_limit
        sigma = calc_sigma_units(self.data["time"], self.data["range"])
        beta_raw_smooth = gaussian_filter(beta_raw, sigma)
        beta_raw_smooth[cloud_ind] = cloud_values
        return noisy_data.screen_data(
            beta_raw_smooth,
            is_smoothed=True,
            snr_limit=snr_limit,
            n_negatives=n_negatives,
        )

    def prepare_data(self) -> None:
        """Add common additional data / metadata and convert into CloudnetArrays."""
        zenith_angle = self.data["zenith_angle"]
        self.data["height"] = np.array(
            self.site_meta["altitude"]
            + utils.range_to_height(self.data["range"], zenith_angle),
        )
        for key in ("time", "range"):
            self.data[key] = np.array(self.data[key])
        if self.instrument is None or self.instrument.wavelength is None:
            msg = "Instrument wavelength not defined"
            raise RuntimeError(msg)
        self.data["wavelength"] = float(self.instrument.wavelength)
        for key in ("latitude", "longitude", "altitude"):
            if key in self.site_meta:
                self.data[key] = float(self.site_meta[key])

    def get_date_and_time(self, epoch: Epoch) -> None:
        if "time" not in self.data:
            msg = "Time array missing from data"
            raise ValidTimeStampError(msg)
        if self.expected_date is not None:
            self.data = utils.screen_by_time(self.data, epoch, self.expected_date)
        self.date = utils.seconds2date(self.data["time"][0], epoch=epoch)[:3]
        self.data["time"] = utils.seconds2hours(self.data["time"])

    def data_to_cloudnet_arrays(self, time_dtype="f4") -> None:
        for key, array in self.data.items():
            if key == "time":
                self.data[key] = CloudnetArray(array, key, data_type=time_dtype)
            else:
                self.data[key] = CloudnetArray(array, key)

    def screen_depol(self) -> None:
        key = "depolarisation"
        if key in self.data:
            self.data[key][self.data[key] <= 0] = ma.masked
            self.data[key][self.data[key] > 1] = ma.masked

    def screen_invalid_values(self) -> None:
        for key in self.data:
            try:
                if self.data[key][:].ndim == 2:
                    self.data[key] = ma.masked_invalid(self.data[key])
            except (IndexError, TypeError):
                continue

    def add_snr_info(self, key: str, snr_limit: float) -> None:
        if key in self.data:
            self.data[key].comment += f" SNR threshold applied: {snr_limit}."

    def check_beta_raw_shape(self) -> None:
        beta_raw = self.data["beta_raw"]
        if beta_raw.ndim != 2 or (beta_raw.shape[0] == 1 or beta_raw.shape[1] == 1):
            msg = f"Invalid beta_raw shape: {beta_raw.shape}"
            raise ValidTimeStampError(msg)


class NoisyData:
    def __init__(
        self,
        data: dict,
        noise_param: NoiseParam,
        *,
        range_corrected: bool = True,
    ):
        self.data = data
        self.noise_param = noise_param
        self.range_corrected = range_corrected

    def screen_data(
        self,
        data_in: np.ndarray,
        snr_limit: float = 5,
        n_negatives: int = 5,
        *,
        is_smoothed: bool = False,
        keep_negative: bool = False,
        filter_fog: bool = True,
        filter_negatives: bool = True,
        filter_snr: bool = True,
    ) -> np.ndarray:
        data = ma.copy(data_in)
        self._calc_range_uncorrected(data)
        noise = _estimate_background_noise(data)
        noise = self._adjust_noise(noise, is_smoothed=is_smoothed)
        if filter_negatives is True:
            is_negative = self._mask_low_values_above_consequent_negatives(
                data,
                n_negatives=n_negatives,
            )
            noise[is_negative] = 1e-12
        if filter_fog is True:
            is_fog = self._find_fog_profiles()
            self._clean_fog_profiles(data, is_fog)
            noise[is_fog] = 1e-12
        if filter_snr is True:
            data = self._remove_noise(
                data,
                noise,
                keep_negative=keep_negative,
                snr_limit=snr_limit,
            )
        self._calc_range_corrected(data)
        return data

    def _adjust_noise(self, noise: np.ndarray, *, is_smoothed: bool) -> np.ndarray:
        noise_min = (
            self.noise_param.noise_smooth_min
            if is_smoothed is True
            else self.noise_param.noise_min
        )
        noise_below_threshold = noise < noise_min
        logging.debug(
            "Adjusted noise of %s profiles",
            sum(np.array(noise_below_threshold)),
        )
        noise[noise_below_threshold] = noise_min
        return noise

    @staticmethod
    def _mask_low_values_above_consequent_negatives(
        data: np.ndarray,
        n_negatives: int = 5,
        threshold: float = 8e-6,
        n_gates: int = 95,
        n_skip_lowest: int = 5,
    ) -> np.ndarray:
        negative_data = data[:, n_skip_lowest : n_gates + n_skip_lowest] < 0
        n_consequent_negatives = utils.cumsumr(negative_data, axis=1)
        time_indices, alt_indices = np.where(n_consequent_negatives > n_negatives)
        alt_indices += n_skip_lowest
        for time_ind, alt_ind in zip(time_indices, alt_indices, strict=True):
            profile = data[time_ind, alt_ind:]
            profile[profile < threshold] = ma.masked
        cleaned_time_indices = np.unique(time_indices)
        logging.debug(
            "Cleaned %s profiles with negative filter",
            len(cleaned_time_indices),
        )
        return cleaned_time_indices

    def _find_fog_profiles(
        self,
        n_gates_for_signal_sum: int = 20,
        signal_sum_threshold: float = 1e-3,
        variance_threshold: float = 1e-15,
    ) -> np.ndarray:
        """Finds saturated (usually fog) profiles from beta_raw."""
        signal_sum = ma.sum(
            ma.abs(self.data["beta_raw"][:, :n_gates_for_signal_sum]),
            axis=1,
        )
        variance = _calc_var_from_top_gates(self.data["beta_raw"])
        is_fog = (signal_sum > signal_sum_threshold) | (variance < variance_threshold)
        logging.debug("Cleaned %s profiles with fog filter", sum(is_fog))
        return is_fog

    def _remove_noise(
        self,
        array: np.ndarray,
        noise: np.ndarray,
        *,
        keep_negative: bool,
        snr_limit: float,
    ) -> np.ndarray:
        snr = array / utils.transpose(noise)
        if self.range_corrected is False:
            snr_scale_factor = 6
            ind = self._get_altitude_ind()
            snr[:, ind] *= snr_scale_factor
        if ma.isMaskedArray(array) is False:
            array = ma.masked_array(array)
        if keep_negative is True:
            array[np.abs(snr) < snr_limit] = ma.masked
        else:
            array[snr < snr_limit] = ma.masked
        return array

    def _calc_range_uncorrected(self, data: np.ndarray) -> None:
        ind = self._get_altitude_ind()
        data[:, ind] = data[:, ind] / self._get_range_squared()[ind]

    def _calc_range_corrected(self, data: np.ndarray) -> None:
        ind = self._get_altitude_ind()
        data[:, ind] = data[:, ind] * self._get_range_squared()[ind]

    def _get_altitude_ind(self) -> tuple:
        if self.range_corrected is False:
            alt_limit = 2400.0
            logging.warning(
                "Raw data not range-corrected, correcting below %s m",
                alt_limit,
            )
        else:
            alt_limit = 1e12
        return np.where(self.data["range"] < alt_limit)

    def _get_range_squared(self) -> np.ndarray:
        """Returns range (m), squared and converted to km."""
        m2km = 0.001
        return (self.data["range"] * m2km) ** 2

    @staticmethod
    def _clean_fog_profiles(
        data: np.ndarray,
        is_fog: np.ndarray,
        threshold: float = 2e-6,
    ) -> None:
        """Removes values in saturated (e.g. fog) profiles above peak."""
        for time_ind in np.where(is_fog)[0]:
            profile = data[time_ind, :]
            peak_ind = int(np.argmax(profile))
            profile[peak_ind:][profile[peak_ind:] < threshold] = ma.masked


def _estimate_background_noise(data: np.ndarray) -> np.ndarray:
    var = _calc_var_from_top_gates(data)
    return np.sqrt(var)


def _calc_var_from_top_gates(data: np.ndarray) -> np.ndarray:
    fraction = 0.1
    n_gates = round(data.shape[1] * fraction)
    return ma.var(data[:, -n_gates:], axis=1)


def calc_sigma_units(
    time_vector: np.ndarray,
    range_los: np.ndarray,
    sigma_minutes: float = 1,
    sigma_metres: float = 10,
) -> tuple[float, float]:
    """Calculates Gaussian peak std parameters.

    The amount of smoothing is hard coded. This function calculates
    how many steps in time and height corresponds to this smoothing.

    Args:
        time_vector: 1D vector (fraction hour).
        range_los: 1D vector (m).
        sigma_minutes: Smoothing in minutes.
        sigma_metres: Smoothing in metres.

    Returns:
        tuple: Two element tuple containing number of steps in time and height to
            achieve wanted smoothing.

    """
    if len(time_vector) == 0 or np.max(time_vector) > 24:
        msg = "Invalid time vector"
        raise ValueError(msg)
    minutes_in_hour = 60
    time_step = utils.mdiff(time_vector) * minutes_in_hour
    alt_step = utils.mdiff(range_los)
    x_std = sigma_minutes / time_step
    y_std = sigma_metres / alt_step
    return x_std, y_std


def _estimate_clouds_from_beta(
    beta: np.ndarray,
) -> tuple[tuple, np.ndarray, float]:
    """Naively finds strong clouds from ceilometer backscatter."""
    cloud_limit = 1e-6
    cloud_ind = np.where(beta > cloud_limit)
    return cloud_ind, beta[cloud_ind], cloud_limit
