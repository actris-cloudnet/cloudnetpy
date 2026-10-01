"""Module with a class for Lufft chm15k ceilometer."""

import datetime
from os import PathLike

import numpy as np
import numpy.typing as npt
from ceilopyter import read_chm15k
from ceilopyter.version import __version__ as ceilopyter_version
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.ceilometer import Ceilometer


class LufftCeilo(Ceilometer):
    """Class for Lufft chm15k ceilometer."""

    def __init__(
        self,
        file_name: str | PathLike | list[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__()
        self.file_name = file_name
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.software = {"ceilopyter": ceilopyter_version}

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        """Reads data and metadata from Jenoptik netCDF file."""
        ceilo = read_chm15k(self.file_name, calibration_factor)
        self.data["beta"] = ceilo.beta
        self.data["beta_raw"] = ceilo.beta_raw
        self.data["time"] = ceilo.time
        self.data["range"] = ceilo.range
        if ceilo.zenith_angle is not None:
            self.data["zenith_angle"] = ma.median(ceilo.zenith_angle)
        self.data["calibration_factor"] = ceilo.calibration_factor
        self.serial_number = ceilo.serial_number
        self.instrument = (
            instruments.CHM15KX
            if ceilo.serial_number and ceilo.serial_number.startswith("CHX")
            else instruments.CHM15K
        )

    def sort_time(self) -> None:
        """Sorts timestamps and removes duplicates."""
        time = self.data["time"]
        _time, ind = np.unique(time, return_index=True)
        self._screen_time_indices(ind)

    def screen_date(self) -> None:
        time = self.data["time"]
        self.date = time[0].date() if self.expected_date is None else self.expected_date
        is_valid = np.array([t.date() == self.date for t in time])
        self._screen_time_indices(is_valid)

    def _screen_time_indices(
        self, valid_indices: npt.NDArray[np.intp] | npt.NDArray[np.bool_]
    ) -> None:
        time = self.data["time"]
        n_time = len(time)
        if len(valid_indices) == 0 or (
            valid_indices.dtype == np.bool_ and not np.any(valid_indices)
        ):
            msg = "All timestamps screened"
            raise utils.ValidTimeStampError(msg)
        for key, array in self.data.items():
            if hasattr(array, "shape") and array.shape[:1] == (n_time,):
                self.data[key] = self.data[key][valid_indices]

    def convert_to_fraction_hour(self) -> None:
        time = self.data["time"]
        midnight = time[0].replace(hour=0, minute=0, second=0, microsecond=0)
        hour = datetime.timedelta(hours=1)
        self.data["time"] = np.array([(t - midnight) / hour for t in time])
