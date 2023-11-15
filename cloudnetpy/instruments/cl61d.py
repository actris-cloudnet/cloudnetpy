"""Module with a class for Lufft chm15k ceilometer."""
import logging

import netCDF4

from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.nc_lidar import NcLidar


class Cl61d(NcLidar):
    """Class for Vaisala CL61d ceilometer."""

    def __init__(
        self,
        file_name: str,
        site_meta: dict,
        expected_date: str | None = None,
    ):
        super().__init__()
        self.file_name = file_name
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.instrument = instruments.CL61D

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        """Reads data and metadata from concatenated Vaisala CL61d netCDF file."""
        with netCDF4.Dataset(self.file_name) as dataset:
            self.dataset = dataset
            self._fetch_attributes()
            self._fetch_zenith_angle("tilt_angle", default=3.0)
            self._fetch_range(reference="lower")
            self._fetch_lidar_variables(calibration_factor)
            self._fetch_time_and_date()
            self.dataset = None

    def _fetch_lidar_variables(self, calibration_factor: float | None = None) -> None:
        if self.dataset is None:
            msg = "No dataset found"
            raise RuntimeError(msg)
        beta_raw = self.dataset.variables["beta_att"][:]
        if calibration_factor is None:
            logging.warning("Using default calibration factor")
            calibration_factor = 1
        beta_raw *= calibration_factor
        self.data["calibration_factor"] = float(calibration_factor)
        self.data["beta_raw"] = beta_raw
        self.data["depolarisation"] = (
            self.dataset.variables["x_pol"][:] / self.dataset.variables["p_pol"][:]
        )
        self.data["depolarisation_raw"] = self.data["depolarisation"].copy()

    def _fetch_attributes(self) -> None:
        self.serial_number = getattr(self.dataset, "instrument_serial_number", None)
