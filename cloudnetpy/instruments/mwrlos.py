"""Module for reading ARM microwave radiometer line-of-sight (mwrlos) data."""

import datetime
import tempfile
from collections.abc import Sequence
from os import PathLike
from uuid import UUID

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.constants import CM_TO_KG_M2
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.arm_utils import concatenate_files, read_geolocation
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.instruments import WVR1100
from cloudnetpy.metadata import COMMON_ATTRIBUTES

RAW_VARIABLES = ("time", "liq", "vap", "qc_liq", "qc_vap", "wet_window")


def mwrlos2nc(
    raw_files: str | PathLike | Sequence[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ARM microwave radiometer line-of-sight retrievals (mwrlos) into
    Cloudnet Level 1b netCDF file.

    Args:
        raw_files: ARM `mwrlos` netCDF file (e.g.
            `sgpmwrlosC1.b1.20100310.000025.cdf`), a sequence of files, or a
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
          >>> from cloudnetpy.instruments import mwrlos2nc
          >>> site_meta = {'name': 'Southern Great Plains'}
          >>> mwrlos2nc('sgpmwrlosC1.b1.20100310.000025.cdf', 'mwr.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_file = concatenate_files(raw_files, temp_dir, RAW_VARIABLES)
        with MwrLos(raw_file, site_meta) as mwr:
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


class MwrLos(DataSource, CloudnetInstrument):
    """Class for ARM mwrlos data.

    Args:
        full_path: Filename of a daily ARM mwrlos netCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str | PathLike, site_meta: dict) -> None:
        super().__init__(full_path)
        self.site_meta = {**site_meta}
        self.instrument = WVR1100
        self.date = utils.get_epoch(self.dataset["time"].units).date()
        self.serial_number = getattr(self.dataset, "serial_number", None)
        self._add_geolocation_from_file()

    def init_data(self) -> None:
        self.append_data(np.array(self.time), "time", dtype="f8")
        for key, name in (("liq", "lwp"), ("vap", "iwv")):
            data = ma.masked_invalid(self.getvar(key)) * CM_TO_KG_M2
            qc = self.getvar(f"qc_{key}")
            data[qc != 0] = ma.masked
            self.append_data(data, name)

    def check_date(self, date: datetime.date) -> None:
        if self.date != date:
            raise ValidTimeStampError

    def screen_invalid_values(self) -> None:
        """Masks retrievals made through a wet radome window."""
        if "wet_window" not in self.dataset.variables:
            return
        is_wet = ma.filled(self.getvar("wet_window"), 0) > 0
        for key in ("lwp", "iwv"):
            self.data[key].data[is_wet] = ma.masked

    def add_zenith_angle(self) -> None:
        self.append_data(0.0, "zenith_angle")

    def _add_geolocation_from_file(self) -> None:
        self.site_meta = read_geolocation(self.dataset, self.site_meta)


ATTRIBUTES = {
    "zenith_angle": COMMON_ATTRIBUTES["zenith_angle"]._replace(dimensions=None),
}
