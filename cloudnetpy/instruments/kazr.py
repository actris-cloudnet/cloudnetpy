"""Module for reading ARM KAZR cloud radar data."""

import datetime
import logging
import os
from collections.abc import Sequence
from itertools import pairwise
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import UUID

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import RadarDataError, ValidTimeStampError
from cloudnetpy.instruments.arm_utils import read_geolocation
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.dealias import (
    CORRECTION_BITS_ATTRIBUTES,
    DEALIASED_V_ATTRIBUTES,
    add_correction_bits,
    dealias_velocity,
)
from cloudnetpy.instruments.instruments import KAZR
from cloudnetpy.instruments.nc_radar import estimate_snr_limit
from cloudnetpy.metadata import COMMON_ATTRIBUTES

# KAZR2 CF/Radial format (e.g. sgpkazrcfrgeC1.a1, hourly files)
KEYMAP_CFR = {
    "reflectivity": "Zh",
    "mean_doppler_velocity": "v",
    "spectral_width": "width",
    "linear_depolarization_ratio": "ldr",
    "signal_to_noise_ratio_copolar_h": "SNR",
    "signal_to_noise_ratio_crosspolar_v": "SNRx",
}

# Same format with short SNR names (e.g. olikazrgeM1.a1)
KEYMAP_CFR_SHORT = {
    **{k: v for k, v in KEYMAP_CFR.items() if v not in ("SNR", "SNRx")},
    "snr_copol": "SNR",
    "snr_xpol": "SNRx",
}

# KAZR moments (e.g. sgpkazrgeC1.a1) and corrected moments (sgpkazrcorgeC1.c1)
KEYMAP_COR = {
    "reflectivity_copol": "Zh",
    "mean_doppler_velocity_copol": "v",
    "spectral_width_copol": "width",
    "signal_to_noise_ratio_copol": "SNR",
    "reflectivity_xpol": "Zx",
    "signal_to_noise_ratio_xpol": "SNRx",
    "significant_detection_mask": "detection_mask",
}


def kazr2nc(
    raw_files: str | PathLike | Sequence[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ARM KAZR cloud radar data into Cloudnet Level 1b netCDF file.

    Supports the KAZR2 CF/Radial moments (`kazrcfrge.a1`, hourly files), the
    ARM corrected moments (`kazrcorge.c1`, daily files) and the original KAZR
    moments (`kazrge.a1`, daily files). With the corrected moments, noise is
    screened using the ARM significant detection mask and LDR is calculated
    from the cross- and co-polar reflectivities. LDR is screened using the
    cross-polar SNR and removed completely if the cross-polar channel is found
    unreliable. Velocities of the non-corrected formats are dealiased using
    velocity continuity.

    Args:
        raw_files: Input file, a sequence of files, or a folder containing the
            files of one day.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude` and
            `altitude` (taken from the raw file if missing) and `snr_limit`
            (fixed SNR threshold in dB; by default the threshold is estimated
            from the noise in the top range gates).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import kazr2nc
          >>> site_meta = {'name': 'Southern Great Plains'}
          >>> kazr2nc('/one/day/of/kazr/files/', 'radar.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    snr_limit = site_meta.get("snr_limit")
    kazr = Kazr(_get_files(raw_files), site_meta, date)
    kazr.read_files()
    kazr.sort_timestamps()
    kazr.remove_duplicate_timestamps()
    kazr.calc_ldr()
    kazr.screen_ldr()
    kazr.correct_ldr_leakage()
    kazr.screen_noise(snr_limit)
    kazr.dealias_velocity()
    kazr.add_correction_bits()
    kazr.mask_invalid_data()
    kazr.add_site_geolocation()
    kazr.add_height()
    kazr.test_if_all_masked()
    attributes = output.add_time_attribute(ATTRIBUTES, kazr.date)
    if kazr.ldr_floor is not None:
        attributes["ldr"] = COMMON_ATTRIBUTES["ldr"]._replace(
            comment=f"Cross-polar leakage floor of {kazr.ldr_floor:.1f} dB "
            "subtracted. Values below the floor are set to -30 dB."
        )
    if kazr.dealiased:
        attributes["v"] = DEALIASED_V_ATTRIBUTES
    output.update_attributes(kazr.data, attributes)
    output.save_level1b(kazr, output_file, uuid)
    return uuid


class Kazr(CloudnetInstrument):
    """Class for ARM KAZR radar data."""

    def __init__(
        self,
        files: list[Path],
        site_meta: dict,
        expected_date: datetime.date | None,
    ) -> None:
        super().__init__()
        self.files = files
        self.site_meta = {**site_meta}
        self.expected_date = expected_date
        self.instrument = KAZR
        self.date: datetime.date
        self.keymap: dict[str, str] = {}
        self.dealiased = False
        self.ldr_floor: float | None = None
        self.corrected = False  # ARM corrected (dealiased) moments
        self._raw: dict[str, list] = {}

    def read_files(self) -> None:
        """Reads and concatenates all input files."""
        for file in self.files:
            with netCDF4.Dataset(file) as nc:
                if not self.keymap:
                    self._init_metadata(nc)
                try:
                    self._read_file(nc)
                except (KeyError, ValueError) as err:
                    logging.warning("Skipping file %s: %s", file, err)
        if "time" not in self._raw:
            msg = "No valid KAZR files found"
            raise ValidTimeStampError(msg)
        for key, arrays in self._raw.items():
            self.data[key] = CloudnetArray(ma.concatenate(arrays), key)
        self._screen_date()

    def screen_noise(self, snr_limit: float | None = None) -> None:
        """Masks noise using ARM detection mask if available, otherwise SNR.

        Without a fixed `snr_limit`, the SNR threshold is estimated for each
        profile from the noise in the top range gates. Small isolated
        clusters remaining after either screening are removed as false
        detections (the ARM mask passes plenty of them).
        """
        if "detection_mask" in self.data:
            is_noise = ma.filled(self.data["detection_mask"][:], 0) != 1
            del self.data["detection_mask"]
        else:
            snr = self.data["SNR"][:]
            if snr_limit is None:
                limit = estimate_snr_limit(snr)[:, np.newaxis]
            else:
                limit = np.array(snr_limit)
            is_noise = ma.filled(snr < limit, fill_value=True)
            is_noise[:, 0] = True  # First gate is contaminated by the transmit pulse
            self.append_data(float(np.median(limit)), "snr_limit")
        is_noise |= ~utils.remove_small_objects(~is_noise, max_size=20, connectivity=2)
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(is_noise)

    def dealias_velocity(self) -> None:
        """Unfolds aliased Doppler velocities using continuity."""
        if self.corrected:
            return
        nyquist = float(self.data["nyquist_velocity"].data)
        self.data["v"].data = dealias_velocity(self.data["v"][:], nyquist)
        self.dealiased = True

    def add_correction_bits(self) -> None:
        # ARM corrected moments are dealiased by ARM
        if self.dealiased or self.corrected:
            add_correction_bits(self.data)

    def calc_ldr(self) -> None:
        """Calculates LDR from cross- and co-polar reflectivity if needed."""
        if "Zx" not in self.data:
            return
        ldr = self.data["Zx"][:] - self.data["Zh"][:]
        self.append_data(ldr, "ldr")
        del self.data["Zx"]

    def screen_ldr(self, snrx_limit: float = -10) -> None:
        """Screens LDR using the cross-polar SNR.

        LDR is removed completely if the cross-polar channel shows signal in
        noise gates (broken channel). Otherwise LDR is masked where the
        cross-polar signal is below the noise: there the reported LDR is only
        an upper bound set by the noise floor, which at low SNR mimics the
        high LDR of insects. Corrected files use the fixed `snrx_limit`, the
        others a limit estimated from the noise in the top range gates.
        """
        if "ldr" not in self.data or "SNRx" not in self.data:
            return
        snr = self.data["SNR"][:]
        snrx = self.data["SNRx"][:]
        is_noise = (snr < 0) & (snrx != 0)
        if np.any(is_noise) and ma.median(snrx[is_noise]) > snrx_limit:
            logging.warning("Cross-polar channel unreliable, removing LDR")
            del self.data["ldr"]
        else:
            limit: float | np.ndarray = snrx_limit
            if not self.corrected:
                limit = estimate_snr_limit(snrx)[:, np.newaxis]
            is_below = ma.filled(snrx < limit, fill_value=True)
            self.data["ldr"].mask_indices(np.where(is_below))
        del self.data["SNRx"]

    def correct_ldr_leakage(self, min_ldr: float = -30, n_sigma: float = 3) -> None:
        """Removes the cross-polar leakage floor from LDR.

        KAZR has limited polarization isolation: LDR of rain and ice sits at
        a constant floor (about -20 dB) regardless of the target. The floor is
        estimated as the mode of LDR in strong echoes and subtracted in linear
        units. Values not significantly above the floor, given the noise
        scatter of LDR at their SNR, are set to `min_ldr`.
        """
        if "ldr" not in self.data:
            return
        ldr = self.data["ldr"][:]
        snr = self.data["SNR"][:]
        valid = ~ma.getmaskarray(ldr)
        strong = valid & (snr > 20)
        if np.count_nonzero(strong) < 1000:
            return
        floor = _find_ldr_floor(ldr[strong])
        if floor is None:
            return
        logging.info("Subtracting LDR leakage floor of %.1f dB", floor)
        excess = ldr - floor
        significant = excess > n_sigma * _ldr_scatter(excess, snr, valid)
        linear = 10 ** (ldr / 10) - 10 ** (floor / 10)
        corrected = 10 * ma.log10(ma.maximum(linear, 10 ** (min_ldr / 10)))
        corrected[~significant] = min_ldr
        self.data["ldr"].data = ma.masked_invalid(corrected)
        self.ldr_floor = floor

    def mask_invalid_data(self) -> None:
        """Makes sure Z and v masks are also in other 2d variables."""
        mask = ma.getmaskarray(self.data["Zh"][:]) | ma.getmaskarray(self.data["v"][:])
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(mask)

    def append_data(self, variable: np.ndarray | float, key: str) -> None:  # type: ignore[override]
        self.data[key] = CloudnetArray(variable, key)

    def test_if_all_masked(self) -> None:
        v = self.data["v"][:]
        if ma.isMaskedArray(v) and np.all(v.mask):
            msg = "All radar data are masked"
            raise RadarDataError(msg)

    def _init_metadata(self, nc: netCDF4.Dataset) -> None:
        if "reflectivity" in nc.variables:
            self.keymap = KEYMAP_CFR
            if "snr_copol" in nc.variables:
                self.keymap = KEYMAP_CFR_SHORT
        elif "reflectivity_copol" in nc.variables:
            self.keymap = KEYMAP_COR
            self.corrected = "significant_detection_mask" in nc.variables
        else:
            msg = "Unknown KAZR file format"
            raise RadarDataError(msg)
        self.serial_number = getattr(nc, "serial_number", None) or None
        self.append_data(np.array(nc["range"][:], dtype=float), "range")
        self.append_data(0.0, "zenith_angle")
        self.append_data(self._read_frequency(nc), "radar_frequency")
        self.append_data(self._read_nyquist(nc), "nyquist_velocity")
        self.site_meta = read_geolocation(nc, self.site_meta)

    def _read_file(self, nc: netCDF4.Dataset) -> None:
        arrays: dict[str, Any] = {"time": self._read_time(nc)}
        for name, key in self.keymap.items():
            if name not in nc.variables:
                if key in ("ldr", "Zx", "SNRx", "detection_mask"):
                    continue
                msg = f"Missing variable {name}"
                raise KeyError(msg)
            data = ma.masked_invalid(nc[name][:])
            if data.ndim != 2 or data.shape[1] != self.data["range"].data.size:
                msg = f"Invalid dimensions of {name}"
                raise ValueError(msg)
            arrays[key] = data
        if "elevation" in nc.variables:
            elevation = ma.filled(nc["elevation"][:], 90)
            is_vertical = np.abs(elevation - 90) < 1
            if not np.all(is_vertical):
                logging.warning(
                    "Filtering %s non-vertical profiles", np.sum(~is_vertical)
                )
                arrays = {k: v[is_vertical] for k, v in arrays.items()}
        for key, data in arrays.items():
            self._raw.setdefault(key, []).append(data)

    def _read_time(self, nc: netCDF4.Dataset) -> np.ndarray:
        """Returns time as seconds since midnight of the date of the first file."""
        epoch = _parse_time_units(nc["time"].units)
        seconds = np.array(nc["time"][:], dtype=float)
        if not hasattr(self, "date"):
            self.date = epoch.date()
        midnight = datetime.datetime.combine(
            self.date, datetime.time.min, tzinfo=datetime.timezone.utc
        )
        return seconds + (epoch - midnight).total_seconds()

    def _screen_date(self) -> None:
        seconds = self.data["time"].data
        is_valid = (seconds >= 0) & (seconds < 24 * 3600)
        if self.expected_date is not None and self.expected_date != self.date:
            is_valid[:] = False
        if not np.any(is_valid):
            msg = f"No valid timestamps for {self.expected_date or self.date}"
            raise ValidTimeStampError(msg)
        self.screen_time_indices(is_valid)
        self.data["time"].data = utils.seconds2hours(self.data["time"].data)

    @staticmethod
    def _read_frequency(nc: netCDF4.Dataset) -> float:
        if "frequency" in nc.variables:
            return float(np.mean(nc["frequency"][:]) / 1e9)  # Hz -> GHz
        value = _parse_global_attribute(nc, "radar_operating_frequency")
        if value is not None:
            return value
        if KAZR.frequency is None:
            msg = "Radar frequency not defined"
            raise RadarDataError(msg)
        return KAZR.frequency

    @staticmethod
    def _read_nyquist(nc: netCDF4.Dataset) -> float:
        if "nyquist_velocity" in nc.variables:
            return float(ma.median(nc["nyquist_velocity"][:]))
        value = _parse_global_attribute(nc, "nyquist_velocity")
        if value is None:
            msg = "Nyquist velocity not found"
            raise RadarDataError(msg)
        return value


def _find_ldr_floor(
    ldr: ma.MaskedArray, min_floor: float = -26, max_floor: float = -15
) -> float | None:
    """Finds the leakage floor as a sharp peak in the LDR distribution."""
    values = ldr.compressed()
    if values.size < 1000:
        return None
    bin_width = 0.25
    counts, edges = np.histogram(values, bins=np.arange(-40, 0, bin_width))
    peak = int(np.argmax(counts))
    floor = float(edges[peak] + bin_width / 2)
    if not min_floor <= floor <= max_floor:
        return None
    # Compare the peak with the distribution within +/- 3 dB around it
    window = int(3 / bin_width)
    core = int(0.5 / bin_width)
    around = np.concatenate(
        (
            counts[max(peak - window, 0) : peak - core],
            counts[peak + core + 1 : peak + window + 1],
        )
    )
    baseline = np.median(around) if around.size else 0
    prominence = counts[peak] / max(baseline, 1)
    share = counts[peak - core : peak + core + 1].sum() / values.size
    if prominence < 5 or share < 0.05:
        return None
    return floor


def _ldr_scatter(
    excess: ma.MaskedArray, snr: ma.MaskedArray, valid: np.ndarray
) -> np.ndarray:
    """Estimates noise scatter (std) of LDR around the floor per SNR bin."""
    edges = np.arange(-10, 60, 5.0)
    scatter = np.full(excess.shape, np.inf)
    near_floor = valid & (np.abs(excess) < 3)
    for lo, hi in pairwise(edges):
        in_bin = (snr >= lo) & (snr < hi)
        samples = excess[in_bin & near_floor].compressed()
        if samples.size < 100:
            continue
        mad = np.median(np.abs(samples - np.median(samples)))
        scatter[in_bin] = 1.4826 * mad
    return scatter


def _parse_global_attribute(nc: netCDF4.Dataset, key: str) -> float | None:
    """Parses numeric value from attribute like '5.963381 m/s'."""
    value = getattr(nc, key, None)
    if value is None:
        return None
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def _parse_time_units(units: str) -> datetime.datetime:
    """Parses e.g. 'seconds since 2022-06-01 01:00:09 0:00' into datetime."""
    parts = units.split()
    if len(parts) < 3 or parts[1] != "since":
        msg = f"Invalid time units: {units}"
        raise ValidTimeStampError(msg)
    text = " ".join(parts[2:4])
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt).replace(
                tzinfo=datetime.timezone.utc
            )
        except ValueError:
            continue
    msg = f"Invalid time units: {units}"
    raise ValidTimeStampError(msg)


def _get_files(raw_files: str | PathLike | Sequence[str | PathLike]) -> list[Path]:
    if isinstance(raw_files, (str, PathLike)):
        if os.path.isdir(raw_files):
            files = [
                Path(raw_files) / f
                for f in os.listdir(raw_files)
                if f.lower().endswith((".nc", ".cdf"))
            ]
        else:
            files = [Path(raw_files)]
    else:
        files = [Path(f) for f in raw_files]
    return sorted(files, key=lambda f: f.name)


ATTRIBUTES = {
    "correction_bits": CORRECTION_BITS_ATTRIBUTES,
    "zenith_angle": COMMON_ATTRIBUTES["zenith_angle"]._replace(dimensions=None),
    "nyquist_velocity": COMMON_ATTRIBUTES["nyquist_velocity"]._replace(dimensions=None),
}
