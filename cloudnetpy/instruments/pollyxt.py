"""Module for reading / converting pollyxt data."""
import glob
import logging

import netCDF4
import numpy as np
from numpy import ma
from numpy.testing import assert_array_equal

from cloudnetpy import output, utils
from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.ceilometer import Ceilometer
from cloudnetpy.metadata import MetaData
from cloudnetpy.utils import Epoch


def pollyxt2nc(
    input_folder: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts PollyXT Raman lidar data into Cloudnet Level 1b netCDF file.

    Args:
    ----
        input_folder: Path to pollyxt netCDF files.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site with keys:

            - `name`: Name of the site (mandatory)
            - `altitude`: Site altitude in [m] (mandatory).
            - `latitude` (optional).
            - `longitude` (optional).
            - `zenith_angle`: If not the default 5 degrees (optional).
            - `snr_limit`: If not the default 2 (optional).
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
    -------
        UUID of the generated file.

    Examples:
    --------
        >>> from cloudnetpy.instruments import pollyxt2nc
        >>> site_meta = {'name': 'Mindelo', 'altitude': 13, 'zenith_angle': 6,
        'snr_limit': 3}
        >>> pollyxt2nc('/path/to/files/', 'pollyxt.nc', site_meta)

    """
    snr_limit = site_meta.get("snr_limit", 2)
    polly = PollyXt(site_meta, date)
    epoch = polly.fetch_data(input_folder)
    polly.get_date_and_time(epoch)
    polly.fetch_zenith_angle()
    polly.calc_screened_products(snr_limit)
    polly.mask_nan_values()
    polly.prepare_data()
    polly.data_to_cloudnet_arrays()
    attributes = output.add_time_attribute(ATTRIBUTES, polly.date)
    output.update_attributes(polly.data, attributes)
    polly.add_snr_info("beta", snr_limit)
    return output.save_level1b(polly, output_file, uuid)


class PollyXt(Ceilometer):
    def __init__(self, site_meta: dict, expected_date: str | None):
        super().__init__()
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.instrument = instruments.POLLYXT

    def mask_nan_values(self) -> None:
        for array in self.data.values():
            if getattr(array, "ndim", 0) > 0:
                array[np.isnan(array)] = ma.masked

    def calc_screened_products(self, snr_limit: float = 5.0) -> None:
        keys = ("beta", "depolarisation")
        for key in keys:
            self.data[key] = ma.masked_where(
                self.data["snr"] < snr_limit,
                self.data[f"{key}_raw"],
            )
        self.data["depolarisation"][self.data["depolarisation"] > 1] = ma.masked
        self.data["depolarisation"][self.data["depolarisation"] < 0] = ma.masked
        self.data["beta"][self.data["beta"] < 0] = ma.masked
        del self.data["snr"]

    def fetch_zenith_angle(self) -> None:
        default = 5
        self.data["zenith_angle"] = float(self.metadata.get("zenith_angle", default))

    def fetch_data(self, input_folder: str) -> Epoch:
        """Read input data."""
        bsc_files = glob.glob(f"{input_folder}/*[0-9]_att*.nc")
        depol_files = glob.glob(f"{input_folder}/*[0-9]_vol*.nc")
        bsc_files.sort()
        depol_files.sort()
        if not bsc_files:
            msg = "No pollyxt bsc files found"
            raise RuntimeError(msg)
        if len(bsc_files) != len(depol_files):
            msg = "Inconsistent number of pollyxt bsc / depol files"
            raise InconsistentDataError(msg)
        self._fetch_attributes(bsc_files[0])
        self.data["range"] = _read_array_from_multiple_files(
            bsc_files,
            depol_files,
            "height",
        )
        calibration_factors: np.ndarray = np.array([])
        beta_channel = self._get_valid_beta_channel(bsc_files)
        bsc_key = f"attenuated_backscatter_{beta_channel}nm"
        for bsc_file, depol_file in zip(bsc_files, depol_files, strict=True):
            with (
                netCDF4.Dataset(bsc_file, "r") as nc_bsc,
                netCDF4.Dataset(depol_file, "r") as nc_depol,
            ):
                epoch = utils.get_epoch(nc_bsc["time"].unit)
                try:
                    time = np.array(
                        _read_array_from_file_pair(nc_bsc, nc_depol, "time"),
                    )
                except AssertionError as err:
                    logging.warning(
                        "Ignoring files '%s' and '%s': %s",
                        nc_bsc,
                        nc_depol,
                        err,
                    )
                    continue
                beta_raw = nc_bsc.variables[bsc_key][:]
                depol_raw = nc_depol.variables["volume_depolarization_ratio_532nm"][:]
                snr = nc_bsc.variables[f"SNR_{beta_channel}nm"][:]
                for array, key in zip(
                    [beta_raw, depol_raw, time, snr],
                    ["beta_raw", "depolarisation_raw", "time", "snr"],
                    strict=True,
                ):
                    self.data = utils.append_data(self.data, key, array)
                calibration_factor = nc_bsc.variables[
                    bsc_key
                ].Lidar_calibration_constant_used
                calibration_factor = np.repeat(calibration_factor, len(time))
                calibration_factors = np.concatenate(
                    [calibration_factors, calibration_factor],
                )
        self.data["calibration_factor"] = calibration_factors
        return epoch

    def _get_valid_beta_channel(self, files: list) -> str:
        polly_channels = ("1064", "532", "355")
        for channel in polly_channels:
            for file in files:
                with netCDF4.Dataset(file, "r") as nc:
                    beta = nc.variables[f"attenuated_backscatter_{channel}nm"][:]
                    if not _only_zeros_or_masked(beta):
                        if channel != polly_channels[0]:
                            logging.warning(
                                "Using %s nm pollyXT channel for backscatter",
                                channel,
                            )
                            if self.instrument is None:
                                msg = "No instrument defined"
                                raise RuntimeError(msg)
                            self.instrument.wavelength = float(channel)
                        return channel
        msg = "No functional pollyXT backscatter channels found"
        raise ValidTimeStampError(msg)

    def _fetch_attributes(self, file: str) -> None:
        with netCDF4.Dataset(file, "r") as nc:
            if hasattr(nc, "source"):
                self.serial_number = nc.source.lower()


def _read_array_from_multiple_files(files1: list, files2: list, key) -> np.ndarray:
    array: np.ndarray = np.array([])
    for ind, (file1, file2) in enumerate(zip(files1, files2, strict=True)):
        with netCDF4.Dataset(file1, "r") as nc1, netCDF4.Dataset(file2, "r") as nc2:
            array1 = _read_array_from_file_pair(nc1, nc2, key)
            if ind == 0:
                array = array1
        assert_array_equal(array, array1, f"Inconsistent variable '{key}'")
    return np.array(array)


def _read_array_from_file_pair(
    nc_file1: netCDF4.Dataset,
    nc_file2: netCDF4.Dataset,
    key: str,
) -> np.ndarray:
    array1 = nc_file1.variables[key][:]
    array2 = nc_file2.variables[key][:]
    assert_array_equal(array1, array2, f"Inconsistent variable '{key}'")
    return array1


def _only_zeros_or_masked(data: ma.MaskedArray) -> bool:
    return ma.sum(data) == 0 or data.mask.all()


ATTRIBUTES = {
    "depolarisation": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="SNR-screened lidar volume linear depolarisation ratio at 532 nm.",
    ),
    "depolarisation_raw": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="Non-screened lidar volume linear depolarisation ratio at 532 nm.",
    ),
    "calibration_factor": MetaData(
        long_name="Attenuated backscatter calibration factor",
        units="1",
        comment="Calibration factor applied.",
    ),
}
