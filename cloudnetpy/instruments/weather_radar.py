import datetime
from collections.abc import Iterable, Sequence
from os import PathLike
from uuid import UUID

import netCDF4
import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import CM_TO_M, HZ_TO_GHZ, KM_TO_M, M_TO_KM, SPEED_OF_LIGHT
from cloudnetpy.exceptions import CloudnetException, ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.metadata import MetaData


def wr2nc(
    input_files: str | PathLike | Sequence[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts OPERA HDF5 weather radar data into Cloudnet Level 1b netCDF file.

    Args:
        input_files: List of OPERA HDF5 files.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required
            keys are `name`, `latitude`, `longitude` and `altitude`.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD.

    Returns:
        UUID of the generated file.
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    if isinstance(input_files, str | PathLike):
        input_files = [input_files]
    wr = WeatherRadar(input_files, site_meta, date)
    wr.sort_and_dedup_timestamps()
    wr.convert_to_cloudnet_arrays()
    wr.screen_noise()
    wr.add_meta()
    attributes = output.add_time_attribute(ATTRIBUTES, wr.date)
    output.update_attributes(wr.data, attributes)
    output.save_level1b(wr, output_file, uuid)
    return uuid


class WeatherRadar(CloudnetInstrument):
    def __init__(
        self,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__()
        self.site_meta = site_meta
        self._read_data(filenames)
        self._screen_time(expected_date)
        self.instrument = instruments.WRM200

    def _read_data(self, filenames: Iterable[str | PathLike]) -> None:
        times = []
        ranges = []
        data = []
        for filename in filenames:
            try:
                file_time, file_range, file_data, file_scalars = _read_opera_h5(
                    filename
                )
            except InvalidRangeError:
                continue
            times.append(file_time)
            ranges.append(file_range)
            data.append(file_data)
        if not times:
            raise ValidTimeStampError
        target_range = max(ranges, key=lambda rng: rng[-1])
        all_data: dict = {key: [] for d in data for key in d}
        for src_range, values in zip(ranges, data, strict=True):
            for key in all_data:
                if key in values:
                    block = ma.array([values[key]])
                    if not np.array_equal(src_range, target_range):
                        block = utils.interpolate_2D_along_y(
                            src_range, block, target_range
                        )
                else:
                    block = ma.masked_all((1, len(target_range)))
                all_data[key].append(block)
        self.raw_time = np.array(times)
        self.raw_range = target_range
        self.raw_data = {key: ma.concatenate(value) for key, value in all_data.items()}
        self.scalars = file_scalars

    def _screen_time(self, expected_date: datetime.date | None = None) -> None:
        if expected_date is None:
            self.date = self.raw_time[0].date()
        else:
            is_valid = [dt.date() == expected_date for dt in self.raw_time]
            self.raw_time = self.raw_time[is_valid]
            for key in self.raw_data:
                self.raw_data[key] = self.raw_data[key][is_valid]
            self.date = expected_date

    def sort_and_dedup_timestamps(self) -> None:
        self.raw_time, time_ind = np.unique(self.raw_time, return_index=True)
        for key in self.raw_data:
            self.raw_data[key] = self.raw_data[key][time_ind]

    def add_meta(self) -> None:
        valid_keys = ("latitude", "longitude", "altitude")
        for key, value in self.site_meta.items():
            name = key.lower()
            if name in valid_keys:
                self.data[name] = CloudnetArray(float(value), name)

    def convert_to_cloudnet_arrays(self) -> None:
        epoch = datetime.datetime.combine(self.date, datetime.time())
        hour = (self.raw_time - epoch) / datetime.timedelta(hours=1)
        height = self.site_meta["altitude"] + self.raw_range
        self.data["time"] = CloudnetArray(hour.astype(np.float32), "time")
        self.data["range"] = CloudnetArray(self.raw_range, "range")
        self.data["height"] = CloudnetArray(height, "height")
        self.data["SNR"] = CloudnetArray(self.raw_data["SNR"], "SNR")
        self.data["Zh"] = CloudnetArray(self.raw_data["DBZH"], "Zh")
        self.data["v"] = CloudnetArray(self.raw_data["VRADH"], "v")
        self.data["width"] = CloudnetArray(self.raw_data["WRADH"], "width")
        self.data["zdr"] = CloudnetArray(self.raw_data["ZDR"], "zdr")
        self.data["rho_hv"] = CloudnetArray(self.raw_data["RHOHV"], "rho_hv")
        self.data["radar_frequency"] = CloudnetArray(
            self.scalars["FREQ"], "radar_frequency"
        )
        self.data["nyquist_velocity"] = CloudnetArray(
            self.scalars["NI"], "nyquist_velocity"
        )
        self.data["calibration_reflectivity_factor"] = CloudnetArray(
            self.scalars["NEZ"], "calibration_reflectivity_factor"
        )

    def screen_noise(self) -> None:
        is_noise = self.data["SNR"].data < 0
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.mask_indices(is_noise)


class InvalidRangeError(CloudnetException):
    pass


def _read_opera_h5(
    file: str | PathLike,
) -> tuple[datetime.datetime, npt.NDArray, dict[str, npt.NDArray], dict[str, float]]:
    all_data = {}
    with netCDF4.Dataset(file) as rootgrp:
        date = rootgrp["what"].date
        time = rootgrp["what"].time
        dt = datetime.datetime.strptime(date + time, "%Y%m%d%H%M%S")

        dataset = rootgrp["dataset1"]
        nbins = dataset["where"].nbins
        rstart = dataset["where"].rstart * KM_TO_M
        rscale = dataset["where"].rscale  # m
        halfbin = rscale / 2
        rng = rstart + halfbin + rscale * np.arange(nbins)
        # Sometimes nbins is larger than normal (e.g. 119 vs 8274), leading to
        # very long range and data that looks like garbage.
        if rng[-1] > 100_000:
            raise InvalidRangeError

        nez = dataset["how"].NEZH
        ni = dataset["how"].NI
        wavelength = rootgrp["how"].wavelength * CM_TO_M
        frequency = HZ_TO_GHZ * SPEED_OF_LIGHT / wavelength
        scalars = {"NEZ": nez, "NI": ni, "FREQ": frequency}

        grpnames = [group for group in dataset.groups if group.startswith("data")]
        for grpname in grpnames:
            grp = dataset[grpname]
            quantity = grp["what"].quantity
            is_db = quantity in ("ZDR", "SNR")
            offset = grp["what"].offset
            gain = grp["what"].gain
            nodata = grp["what"].nodata
            undetect = grp["what"].undetect
            grpdata = grp["data"][:]
            is_masked = (grpdata == nodata) | (grpdata == undetect)
            grpdata = offset + gain * ma.masked_where(is_masked, grpdata)
            if is_db:
                grpdata = utils.db2lin(grpdata)
            grpdata = ma.mean(grpdata, axis=0)
            if is_db:
                grpdata = utils.lin2db(grpdata)
            all_data[quantity] = grpdata

        # DBZH is available in some files for some days, but for consistency,
        # always recalculate it from SNR. The value calculated from SNR is close
        # to DBZH but not exactly for some unknown reason.
        all_data["DBZH"] = all_data["SNR"] + nez + 20 * np.log10(rng * M_TO_KM)

    return dt, rng, all_data, scalars


ATTRIBUTES = {
    "calibration_reflectivity_factor": MetaData(
        long_name="Calibration reflectivity factor",
        comment="This parameter is the equivalent radar reflectivity factor at 1 km\n"
        "when the return signal power is equal to the noise power (SNR=0 dB).",
        units="dBZ",
        dimensions=(),
    )
}
