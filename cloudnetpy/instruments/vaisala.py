"""Module with classes for Vaisala ceilometers."""

import datetime
from collections.abc import Callable
from os import PathLike

import ceilopyter.version
import numpy as np
import numpy.typing as npt
from ceilopyter import read_cl_file, read_cs_file, read_ct_file

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.ceilometer import Ceilometer, NoiseParam


class VaisalaCeilo(Ceilometer):
    """Base class for Vaisala ceilometers."""

    def __init__(
        self,
        reader: Callable,
        full_path: str | PathLike,
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(self.noise_param)
        self.reader = reader
        self.full_path = full_path
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.software = {"ceilopyter": ceilopyter.version.__version__}

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        """Read all lines of data from the file."""
        time, data = self.reader(self.full_path)
        if not data:
            msg = "No valid data found."
            raise ValidTimeStampError(msg)
        range_res = data[0].range_resolution
        n_gates = len(data[0].beta)
        self.data["time"] = np.array(time)
        self.data["range"] = np.arange(n_gates) * range_res + range_res / 2
        self.data["beta_raw"] = np.stack([d.beta for d in data])
        self.data["calibration_factor"] = calibration_factor or 1.0
        self.data["beta_raw"] *= self.data["calibration_factor"]
        self.data["zenith_angle"] = np.median([d.tilt_angle for d in data])
        self.sort_time()
        self.screen_date()
        self.convert_to_fraction_hour()
        self._store_ceilometer_info()

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
        self, valid_indices: npt.NDArray[np.intp] | npt.NDArray[np.bool]
    ) -> None:
        time = self.data["time"]
        n_time = len(time)
        if len(valid_indices) == 0 or (
            valid_indices.dtype == np.bool and not np.any(valid_indices)
        ):
            msg = "All timestamps screened"
            raise ValidTimeStampError(msg)
        for key, array in self.data.items():
            if hasattr(array, "shape") and array.shape[:1] == (n_time,):
                self.data[key] = self.data[key][valid_indices]

    def convert_to_fraction_hour(self) -> None:
        time = self.data["time"]
        midnight = time[0].replace(hour=0, minute=0, second=0, microsecond=0)
        hour = datetime.timedelta(hours=1)
        self.data["time"] = (time - midnight) / hour

    def _store_ceilometer_info(self) -> None:
        raise NotImplementedError


class ClCeilo(VaisalaCeilo):
    """Class for Vaisala CL31/CL51 ceilometers."""

    noise_param = NoiseParam(noise_min=3.1e-8, noise_smooth_min=1.1e-8)

    def __init__(
        self,
        full_path: str | PathLike,
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(read_cl_file, full_path, site_meta, expected_date)

    def _store_ceilometer_info(self) -> None:
        n_gates = self.data["beta_raw"].shape[1]
        if n_gates < 1540:
            self.instrument = instruments.CL31
        else:
            self.instrument = instruments.CL51


class Ct25k(VaisalaCeilo):
    """Class for Vaisala CT25k ceilometer."""

    noise_param = NoiseParam(noise_min=0.7e-7, noise_smooth_min=1.2e-8)

    def __init__(
        self,
        full_path: str | PathLike,
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(read_ct_file, full_path, site_meta, expected_date)
        self._store_ceilometer_info()

    def _store_ceilometer_info(self) -> None:
        self.instrument = instruments.CT25K


class Cs135(VaisalaCeilo):
    """Class for Campbell Scientific CS135 ceilometer."""

    noise_param = NoiseParam()

    def __init__(
        self,
        full_path: str | PathLike,
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(read_cs_file, full_path, site_meta, expected_date)

    def _store_ceilometer_info(self) -> None:
        self.instrument = instruments.CS135
