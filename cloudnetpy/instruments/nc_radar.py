"""Module for reading raw cloud radar data."""
import logging

import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.instruments import Instrument


class NcRadar(DataSource, CloudnetInstrument):
    """Class for radars providing netCDF files. Child of DataSource().

    Args:
        full_path: Filename of a radar-produced netCDF file.
        site_meta: Some metadata of the site.

    Notes:
        Used with BASTA, MIRA and Copernicus radars.
    """

    def __init__(self, full_path: str, site_meta: dict):
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
                logging.warning(f"Can not find variable {key} from the input file")
                continue
            array = np.array(array) if utils.isscalar(array) else array
            array[~np.isfinite(array)] = ma.masked
            self.append_data(array, name)

    def add_time_and_range(self) -> None:
        """Adds time and range."""
        range_instru = np.array(
            self.getvar("range", "height")
        )  # "height" in old BASTA files
        time = np.array(self.time)
        self.append_data(range_instru, "range")
        self.append_data(time, "time")

    def screen_by_snr(self, snr_limit: float = -17) -> None:
        """Mask values where SNR smaller than threshold."""
        ind = np.where(self.data["SNR"][:] < snr_limit)
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(ind)

    def mask_invalid_data(self) -> None:
        """Makes sure Z and v masks are also in other 2d variables."""
        z_mask = self.data["Zh"][:].mask
        v_mask = self.data["v"][:].mask
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(z_mask)
                cloudnet_array.mask_indices(v_mask)

    def add_zenith_and_azimuth_angles(self) -> list:
        """Adds non-varying instrument zenith and azimuth angles and returns valid
        time indices."""
        if "azimuth_velocity" in self.data:
            azimuth = self.data["azimuth_velocity"].data
            if np.all(azimuth == azimuth[0]):
                azimuth_reference = azimuth[0]
            else:
                azimuth_reference = 0
            azimuth_tolerance = 1e-6
        else:
            azimuth = self.data["azimuth_angle"].data
            azimuth_reference = ma.median(azimuth)
            azimuth_tolerance = 0.1

        elevation = self.data["elevation"].data
        zenith = 90 - elevation
        is_stable_zenith = np.isclose(zenith, ma.median(zenith), atol=0.1)
        is_stable_azimuth = np.isclose(
            azimuth, azimuth_reference, atol=azimuth_tolerance
        )
        is_stable_profile = is_stable_zenith & is_stable_azimuth
        if ma.isMaskedArray(is_stable_profile):
            is_stable_profile[is_stable_profile.mask] = False
        n_removed = np.count_nonzero(~is_stable_profile)
        if n_removed >= len(zenith) - 1:
            raise ValidTimeStampError(
                "Less than two profiles with valid zenith / azimuth angles"
            )
        if n_removed > 0:
            logging.warning(
                f"Filtering {n_removed} profiles due to varying zenith / azimuth angle"
            )
        self.append_data(zenith, "zenith_angle")
        for key in ("elevation", "azimuth_velocity"):
            if key in self.data:
                del self.data[key]
        return list(is_stable_profile)

    def add_radar_specific_variables(self):
        """Adds radar specific variables."""
        assert self.instrument is not None
        key = "radar_frequency"
        self.data[key] = CloudnetArray(self.instrument.frequency, key)
        try:
            possible_nyquist_names = ("ambiguous_velocity", "NyquistVelocity")
            data = self.getvar(*possible_nyquist_names)
            key = "nyquist_velocity"
            self.data[key] = CloudnetArray(np.array(data), key)
        except RuntimeError:
            logging.warning("Unable to find nyquist_velocity")


class ChilboltonRadar(NcRadar):
    """Class for Chilbolton cloud radars Galileo and Copernicus."""

    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.date = self._init_date()

    def add_nyquist_velocity(self, keymap: dict):
        """Adds nyquist velocity."""
        key = [key for key, value in keymap.items() if value == "v"][0]
        folding_velocity = self.dataset.variables[key].folding_velocity
        self.append_data(np.array(folding_velocity), "nyquist_velocity")

    def check_date(self, date: str):
        if self.date != date.split("-"):
            raise ValidTimeStampError

    def _init_date(self) -> list[str]:
        epoch = utils.get_epoch(self.dataset["time"].units)
        return [str(x).zfill(2) for x in epoch]
