"""Module for reading ARM microwave radiometer line-of-sight (mwrlos) data."""

import datetime
import os
import tempfile
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from uuid import UUID

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.arm_utils import read_geolocation
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.instruments import WVR1100
from cloudnetpy.metadata import COMMON_ATTRIBUTES

CM_TO_KG_M2 = 10


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
        raw_file = _concatenate(raw_files, temp_dir)
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


def _concatenate(
    raw_files: str | PathLike | Sequence[str | PathLike], temp_dir: str
) -> str | PathLike:
    files: list[Path]
    if isinstance(raw_files, (str, PathLike)):
        if not os.path.isdir(raw_files):
            return raw_files
        files = [
            Path(raw_files) / f
            for f in os.listdir(raw_files)
            if f.lower().endswith((".cdf", ".nc"))
        ]
    else:
        files = [Path(f) for f in raw_files]
    files = sorted(files, key=lambda f: f.name)
    if len(files) == 1:
        return files[0]
    output_file = Path(temp_dir) / "mwrlos.nc"
    variables = ("time", "liq", "vap", "qc_liq", "qc_vap", "wet_window")
    with netCDF4.Dataset(output_file, "w") as nc_out:
        nc_out.createDimension("time", None)
        for ind, file in enumerate(files):
            with netCDF4.Dataset(file) as nc_in:
                if ind == 0:
                    nc_out.setncatts({k: nc_in.getncattr(k) for k in nc_in.ncattrs()})
                    time_units = nc_in["time"].units
                elif nc_in["time"].units != time_units:
                    msg = "Inconsistent time units in mwrlos files"
                    raise ValidTimeStampError(msg)
                n_time = len(nc_out.dimensions["time"])
                for key in nc_in.variables:
                    if key not in variables and nc_in[key].ndim != 0:
                        continue
                    if key not in nc_out.variables:
                        var = nc_out.createVariable(
                            key, nc_in[key].dtype, nc_in[key].dimensions
                        )
                        var.setncatts(
                            {k: nc_in[key].getncattr(k) for k in nc_in[key].ncattrs()}
                        )
                        if nc_in[key].ndim == 0:
                            var[:] = nc_in[key][:]
                    if nc_in[key].ndim != 0:
                        nc_out[key][n_time:] = nc_in[key][:]
    return output_file


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
