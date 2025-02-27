"""Module for reading raw cloud radar data."""

import logging
from os import PathLike
from typing import TYPE_CHECKING

import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import RadarDataError, ValidTimeStampError
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument

if TYPE_CHECKING:
    from cloudnetpy.instruments.instruments import Instrument


class NcRadar(DataSource, CloudnetInstrument):
    """Class for radars providing netCDF files. Child of DataSource().

    Args:
        full_path: Filename of a radar-produced netCDF file.
        site_meta: Some metadata of the site.

    Notes:
        Used with BASTA, MIRA and Copernicus radars.
    """

    def __init__(self, full_path: PathLike | str, site_meta: dict):
        super().__init__(full_path)
        self.site_meta = site_meta
        self.date: list[str]
        self.instrument: Instrument | None = None

    def init_data(self, keymap: dict) -> None:
        """Reads selected fields and fixes the names."""
        for key in keymap:
            name = keymap[key]
            try:
                array = self.getvar(key)
            except RuntimeError:
                logging.warning("Can not find variable %s from the input file", key)
                continue
            array = np.array(array) if utils.isscalar(array) else array
            array[~np.isfinite(array)] = ma.masked
            self.append_data(array, name)

    def add_time_and_range(self, time_dtype: str | None = None) -> None:
        """Adds time and range."""
        range_instru = np.array(
            self.getvar("range", "height"),
        )  # "height" in old BASTA files
        time = np.array(self.time)
        self.append_data(range_instru, "range")
        self.append_data(time, "time", dtype=time_dtype)

    def screen_by_snr(self, snr_limit: float) -> None:
        """Mask values where SNR smaller than threshold."""
        ind = np.where(self.data["SNR"][:] < snr_limit)
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(ind)
        self.append_data(float(snr_limit), "snr_limit")

    def screen_using_top_gates_snr(self, snr_limit: float = 2) -> None:
        """Masks values where SNR is smaller than mean SNR of top gates."""
        n_gates = 50
        snr = self.data["SNR"][:]
        mean_snr = np.mean(snr[:, -n_gates:], axis=1)
        for time_ind, snr_profile in enumerate(snr):
            alt_ind = np.where(snr_profile < mean_snr[time_ind] + snr_limit)[0]
            if len(alt_ind) > 0:
                self.data["Zh"][:][time_ind, alt_ind] = ma.masked

    def mask_invalid_data(self) -> None:
        """Makes sure Z and v masks are also in other 2d variables."""
        z_mask = self.data["Zh"][:].mask
        v_mask = self.data["v"][:].mask
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(z_mask)
                cloudnet_array.mask_indices(v_mask)

    def mask_first_range_gates(self, range_limit: float = 150) -> None:
        """Masks first range gates."""
        if "v" not in self.data or "range" not in self.data:
            return
        ind = np.where(self.data["range"][:] < range_limit)[0]
        if len(ind) > 0:
            self.data["v"].data[:, ind] = ma.masked

    def add_zenith_and_azimuth_angles(
        self,
        elevation_threshold: float,
        elevation_diff_threshold: float,
        azimuth_diff_threshold: float,
        zenith_offset: float | None = None,
        azimuth_offset: float | None = None,
    ) -> list:
        """Adds non-varying instrument zenith and azimuth angles and returns valid
        time indices.
        """
        elevation = self.data["elevation"].data
        if zenith_offset is not None:
            self.append_data(zenith_offset, "zenith_offset")
            elevation -= zenith_offset

        azimuth = self.data["azimuth_angle"].data
        if azimuth_offset is not None:
            self.append_data(azimuth_offset, "azimuth_offset")
            azimuth += azimuth_offset

        elevation_diff = ma.diff(elevation, prepend=elevation[1])
        azimuth_diff = ma.diff(azimuth, prepend=azimuth[1])

        is_stable = np.abs(elevation - 90) < elevation_threshold
        is_stable &= np.abs(elevation_diff) < elevation_diff_threshold
        is_stable &= np.abs(azimuth_diff) < azimuth_diff_threshold

        # If scanning unit is broken, data are missing
        # (assume it's vertically pointing)
        missing_info = elevation.mask & azimuth.mask
        is_stable[missing_info] = True

        if ma.isMaskedArray(is_stable):
            is_stable[is_stable.mask] = False
        n_removed = np.count_nonzero(~is_stable)

        if n_removed >= len(elevation) - 1:
            msg = "Less than two profiles with valid zenith / azimuth angles"
            raise ValidTimeStampError(msg)

        if n_removed > 0:
            logging.warning(
                "Filtering %s profiles due to varying zenith / azimuth angle",
                n_removed,
            )
        zenith = 90 - elevation
        self.append_data(zenith, "zenith_angle")
        del self.data["elevation"]
        return list(is_stable)

    def add_radar_specific_variables(self) -> None:
        """Adds radar specific variables."""
        if self.instrument is None:
            msg = "Instrument not defined"
            raise RuntimeError(msg)
        key = "radar_frequency"
        if self.instrument.frequency is not None:
            self.data[key] = CloudnetArray(self.instrument.frequency, key)
        try:
            possible_nyquist_names = ("ambiguous_velocity", "NyquistVelocity")
            data = self.getvar(*possible_nyquist_names)
            key = "nyquist_velocity"
            self.data[key] = CloudnetArray(np.median(np.array(data)), key)
            if "NyquistVelocity" in self.data:
                del self.data["NyquistVelocity"]
        except RuntimeError:
            logging.warning("Unable to find nyquist_velocity")

    def test_if_all_masked(self) -> None:
        """Tests if all data are masked."""
        if np.all(self.data["v"][:].mask):
            msg = "All radar data are masked"
            raise RadarDataError(msg)


class ChilboltonRadar(NcRadar):
    """Class for Chilbolton cloud radars Galileo and Copernicus."""

    def __init__(self, full_path: str, site_meta: dict) -> None:
        super().__init__(full_path, site_meta)
        self.date = self._init_date()

    def add_nyquist_velocity(self, keymap: dict) -> None:
        """Adds nyquist velocity."""
        key = [key for key, value in keymap.items() if value == "v"][0]  # noqa: RUF015
        folding_velocity = self.dataset.variables[key].folding_velocity
        self.append_data(np.array(folding_velocity), "nyquist_velocity")

    def check_date(self, date: str) -> None:
        if self.date != date.split("-"):
            raise ValidTimeStampError

    def _init_date(self) -> list[str]:
        epoch = utils.get_epoch(self.dataset["time"].units)
        return [str(x).zfill(2) for x in epoch]
