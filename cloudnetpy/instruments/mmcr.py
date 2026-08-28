"""Module for reading ARM MMCR cloud radar data."""

import datetime
import logging
from os import PathLike
from uuid import UUID

import netCDF4
import numpy as np

from cloudnetpy import output, utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.arm_utils import read_geolocation
from cloudnetpy.instruments.dealias import (
    CORRECTION_BITS_ATTRIBUTES,
    DEALIASED_V_ATTRIBUTES,
    add_correction_bits,
    dealias_velocity,
)
from cloudnetpy.instruments.instruments import MMCR
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import COMMON_ATTRIBUTES


def mmcr2nc(
    raw_file: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ARM MMCR cloud radar moments (mmcrmom) into Cloudnet Level 1b
    netCDF file.

    The MMCR cycles through several operating modes with different range
    resolutions and sensitivities. Only profiles from a single mode are used.
    Doppler velocities are folded at the Nyquist velocity of the mode, which
    is only about 5 m/s in the default general (GE) mode, and are dealiased
    using velocity continuity. Depolarization ratio is not available in the
    GE mode.

    Args:
        raw_file: Daily ARM `mmcrmom` netCDF file, e.g.
            `sgpmmcrmomC1.b1.20100310.000047.cdf`.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude` and
            `altitude` (taken from the raw file if missing), `mode` (operating mode
            identifier, default = 'GE') and `snr_limit` (fixed SNR threshold in
            dB; by default the threshold is estimated from the noise in the top
            range gates).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import mmcr2nc
          >>> site_meta = {'name': 'Southern Great Plains'}
          >>> mmcr2nc('sgpmmcrmomC1.b1.20100310.000047.cdf', 'radar.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    mode = site_meta.get("mode", "GE")
    snr_limit = site_meta.get("snr_limit")

    keymap = {
        "Reflectivity": "Zh",
        "MeanDopplerVelocity": "v",
        "SpectralWidth": "width",
        "SignalToNoiseRatio": "SNR",
    }

    with Mmcr(raw_file, site_meta) as mmcr:
        mmcr.init_data(keymap)
        mmcr.init_mode(mode)
        mmcr.screen_mode()
        if date is not None:
            mmcr.check_date(date)
        mmcr.sort_timestamps()
        mmcr.remove_duplicate_timestamps()
        mmcr.screen_by_snr(snr_limit)
        mmcr.mask_invalid_data()
        mmcr.flip_velocity_sign()
        mmcr.add_radar_specific_variables()
        mmcr.dealias_velocity()
        mmcr.add_zenith_angle()
        mmcr.add_site_geolocation()
        mmcr.add_height()
        mmcr.test_if_all_masked()
    attributes = output.add_time_attribute(ATTRIBUTES, mmcr.date)
    output.update_attributes(mmcr.data, attributes)
    output.save_level1b(mmcr, output_file, uuid)
    return uuid


class Mmcr(NcRadar):
    """Class for ARM MMCR radar data. Child of NcRadar().

    Args:
        full_path: Filename of a daily ARM mmcrmom netCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str | PathLike, site_meta: dict) -> None:
        super().__init__(full_path, {**site_meta})
        self.instrument = MMCR
        self.date = utils.get_epoch(self.dataset["time"].units).date()
        self.mode_index: int = 0
        self._add_geolocation_from_file()

    def init_mode(self, mode: str) -> None:
        """Adds time and range of the selected operating mode."""
        self.mode_index = self._find_mode_index(mode)
        n_heights = int(self.dataset["NumHeights"][self.mode_index])
        heights = self.dataset["heights"][self.mode_index, :n_heights]
        altitude = float(self.dataset["alt"][:])
        range_instru = np.array(heights - altitude)
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.data = cloudnet_array.data[:, :n_heights]
        self.append_data(range_instru, "range")
        self.append_data(np.array(self.time), "time")

    def screen_mode(self) -> None:
        """Keeps only profiles measured with the selected operating mode."""
        mode_num = np.array(self.dataset["ModeNum"][:])
        is_mode = mode_num == self.mode_index
        if not np.any(is_mode):
            msg = "No profiles found for the selected radar mode"
            raise ValidTimeStampError(msg)
        self.screen_time_indices(is_mode)

    def check_date(self, date: datetime.date) -> None:
        if self.date != date:
            raise ValidTimeStampError

    def add_radar_specific_variables(self) -> None:
        if self.instrument is None or self.instrument.frequency is None:
            msg = "Instrument not defined"
            raise RuntimeError(msg)
        self.append_data(self.instrument.frequency, "radar_frequency")
        nyquist = float(self.dataset["NyquistVelocity"][self.mode_index])
        self.append_data(nyquist, "nyquist_velocity")

    def flip_velocity_sign(self) -> None:
        # ARM: positive towards the radar. Cloudnet: positive away from the radar.
        self.data["v"].data *= -1

    def dealias_velocity(self) -> None:
        """Unfolds aliased Doppler velocities using continuity."""
        nyquist = float(self.data["nyquist_velocity"].data)
        self.data["v"].data = dealias_velocity(self.data["v"][:], nyquist)
        add_correction_bits(self.data)

    def add_zenith_angle(self) -> None:
        # MMCR is a fixed vertically pointing radar
        self.append_data(0.0, "zenith_angle")

    def _find_mode_index(self, mode: str) -> int:
        var = self.dataset["ModeDescription"]
        var.set_auto_mask(False)
        names = [str(d).strip() for d in netCDF4.chartostring(var[:])]
        for ind, name in enumerate(names):
            if name.endswith(f"_{mode}"):
                logging.info("Using radar mode %s", name)
                return ind
        available = [name for name in names if name and "Reserved" not in name]
        msg = f"Radar mode '{mode}' not found. Available modes: {available}"
        raise ValueError(msg)

    def _add_geolocation_from_file(self) -> None:
        self.site_meta = read_geolocation(self.dataset, self.site_meta)


ATTRIBUTES = {
    "correction_bits": CORRECTION_BITS_ATTRIBUTES,
    "v": DEALIASED_V_ATTRIBUTES,
    "zenith_angle": COMMON_ATTRIBUTES["zenith_angle"]._replace(dimensions=None),
    "nyquist_velocity": COMMON_ATTRIBUTES["nyquist_velocity"]._replace(dimensions=None),
}
