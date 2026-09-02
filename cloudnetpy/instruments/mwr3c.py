"""Module for reading ARM 3-channel microwave radiometer (mwr3c) data."""

import datetime
import tempfile
from collections.abc import Sequence
from os import PathLike
from uuid import UUID

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.arm_utils import concatenate_files, read_geolocation
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.instruments import MWR3C
from cloudnetpy.metadata import COMMON_ATTRIBUTES

CM_TO_KG_M2 = 10
MISSING_VALUE = -9999
MAX_OFF_ZENITH = 1.0  # deg
RAW_VARIABLES = (
    "time",
    "lwp",
    "pwv",
    "elevation",
    "rain_flag",
    "qc_tbsky23",
    "qc_tbsky31",
)


def mwr3c2nc(
    raw_files: str | PathLike | Sequence[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ARM 3-channel microwave radiometer retrievals (mwr3c) into
    Cloudnet Level 1b netCDF file.

    Args:
        raw_files: ARM `mwr3c` netCDF file (e.g.
            `enamwr3cC1.b1.20240315.000004.nc`), a sequence of files, or a
            folder containing the files of one day.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude` and
            `altitude` (taken from the raw file if missing).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import mwr3c2nc
          >>> site_meta = {'name': 'Graciosa'}
          >>> mwr3c2nc('enamwr3cC1.b1.20240315.000004.nc', 'mwr.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_file = concatenate_files(raw_files, temp_dir, RAW_VARIABLES)
        with Mwr3c(raw_file, site_meta) as mwr:
            if date is not None:
                mwr.check_date(date)
            mwr.init_data()
            mwr.sort_timestamps()
            mwr.remove_duplicate_timestamps()
            mwr.screen_invalid_values()
            mwr.add_zenith_angle()
            mwr.add_site_geolocation()
    attributes = output.add_time_attribute(ATTRIBUTES, mwr.date)
    output.update_attributes(mwr.data, attributes)
    output.save_level1b(mwr, output_file, uuid)
    return uuid


class Mwr3c(DataSource, CloudnetInstrument):
    """Class for ARM mwr3c data.

    Args:
        full_path: Filename of a daily ARM mwr3c netCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str | PathLike, site_meta: dict) -> None:
        super().__init__(full_path)
        self.site_meta = {**site_meta}
        self.instrument = MWR3C
        self.date = utils.get_epoch(self.dataset["time"].units).date()
        serial_number = getattr(self.dataset, "serial_number", None)
        self.serial_number = serial_number.strip() if serial_number else None
        self._add_geolocation_from_file()

    def init_data(self) -> None:
        self.append_data(np.array(self.time), "time", dtype="f8")
        for key, name, factor in (("lwp", "lwp", 1), ("pwv", "iwv", CM_TO_KG_M2)):
            data = ma.masked_invalid(self.getvar(key))
            data = ma.masked_equal(data, MISSING_VALUE) * factor
            self.append_data(data, name)

    def check_date(self, date: datetime.date) -> None:
        if self.date != date:
            raise ValidTimeStampError

    def screen_invalid_values(self) -> None:
        """Masks rainy, off-zenith and flagged retrievals."""
        is_invalid = np.zeros(len(self.data["time"].data), dtype=bool)
        if "rain_flag" in self.dataset.variables:
            is_invalid |= ma.filled(self.getvar("rain_flag"), 0) > 0
        if "elevation" in self.dataset.variables:
            elevation = ma.filled(self.getvar("elevation"), MISSING_VALUE)
            is_invalid |= np.abs(elevation - 90) > MAX_OFF_ZENITH
        for key in ("qc_tbsky23", "qc_tbsky31"):
            if key in self.dataset.variables:
                is_invalid |= ma.filled(self.getvar(key), 0) != 0
        for key in ("lwp", "iwv"):
            self.data[key].data[is_invalid] = ma.masked

    def add_zenith_angle(self) -> None:
        self.append_data(0.0, "zenith_angle")

    def _add_geolocation_from_file(self) -> None:
        self.site_meta = read_geolocation(self.dataset, self.site_meta)


ATTRIBUTES = {
    "zenith_angle": COMMON_ATTRIBUTES["zenith_angle"]._replace(dimensions=None),
}
