"""Module with classes for Vaisala ceilometers."""

import datetime
from collections.abc import Callable

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
        full_path: str,
        site_meta: dict,
        expected_date: str | None = None,
    ):
        super().__init__(self.noise_param)
        self.reader = reader
        self.full_path = full_path
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.sane_date = (
            datetime.date.fromisoformat(self.expected_date)
            if self.expected_date
            else None
        )
        self.software = {"ceilopyter": ceilopyter.version.__version__}

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        """Read all lines of data from the file."""
        time, data = self.reader(self.full_path)
        range_res = data[0].range_resolution
        n_gates = len(data[0].beta)
        self.data["time"] = np.array(time)
        self.data["range"] = np.arange(n_gates) * range_res + range_res / 2
        self.data["beta_raw"] = np.stack([d.beta for d in data])
        self.data["calibration_factor"] = calibration_factor or 1.0
        self.data["beta_raw"] *= self.data["calibration_factor"]
        self.data["zenith_angle"] = np.median([d.tilt_angle for d in data])
        self._sort_time()
        self._screen_date()
        self._convert_to_fraction_hour()
        self._store_ceilometer_info()

    def _sort_time(self):
        """Sorts timestamps and removes duplicates."""
        time = self.data["time"]
        _time, ind = np.unique(time, return_index=True)
        self._screen_time_indices(ind)

    def _screen_date(self):
        time = self.data["time"]
        if self.sane_date is None:
            self.sane_date = time[0].date()
            self.expected_date = self.sane_date.isoformat()
        is_valid = np.array([t.date() == self.sane_date for t in time])
        self._screen_time_indices(is_valid)

    def _screen_time_indices(
        self, valid_indices: npt.NDArray[np.intp] | npt.NDArray[np.bool]
    ):
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

    def _convert_to_fraction_hour(self):
        time = self.data["time"]
        midnight = time[0].replace(hour=0, minute=0, second=0, microsecond=0)
        hour = datetime.timedelta(hours=1)
        self.data["time"] = (time - midnight) / hour
        self.date = self.expected_date.split("-")  # type: ignore[union-attr]

    def _store_ceilometer_info(self):
        raise NotImplementedError


class ClCeilo(VaisalaCeilo):
    """Class for Vaisala CL31/CL51 ceilometers."""

    noise_param = NoiseParam(noise_min=3.1e-8, noise_smooth_min=1.1e-8)

    def __init__(self, full_path, site_meta, expected_date=None):
        super().__init__(read_cl_file, full_path, site_meta, expected_date)

    def _store_ceilometer_info(self):
        n_gates = self.data["beta_raw"].shape[1]
        if n_gates < 1540:
            self.instrument = instruments.CL31
        else:
            self.instrument = instruments.CL51


class Ct25k(VaisalaCeilo):
    """Class for Vaisala CT25k ceilometer."""

    noise_param = NoiseParam(noise_min=0.7e-7, noise_smooth_min=1.2e-8)

    def __init__(self, full_path, site_meta, expected_date=None):
        super().__init__(read_ct_file, full_path, site_meta, expected_date)

    def _store_ceilometer_info(self):
        self.instrument = instruments.CT25K


class Cs135(VaisalaCeilo):
    """Class for Campbell Scientific CS135 ceilometer."""

    noise_param = NoiseParam()

    def __init__(self, full_path, site_meta, expected_date=None):
        super().__init__(read_cs_file, full_path, site_meta, expected_date)

    def _store_ceilometer_info(self):
        self.instrument = instruments.CS135
