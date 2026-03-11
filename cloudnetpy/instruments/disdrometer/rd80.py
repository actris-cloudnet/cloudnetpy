import datetime
from collections import defaultdict
from collections.abc import Iterable
from os import PathLike
from uuid import UUID

import numpy as np
from numpy import ma

from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import MM_TO_M, SEC_IN_HOUR
from cloudnetpy.disdronator.rd80 import A, Dlow, Dmid, Dspr, Dupp, Vmid, read_rd80
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.utils import get_uuid

from .common import ATTRIBUTES


def rd802nc(
    input_file: str | PathLike | list[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts Distromet RD-80 disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        input_file: Filename(s) of RD-80 disdrometer data file(s). Can be a
            single file or a list of files.
        output_file: Output filename for the netCDF file.
        site_meta: Dictionary containing information about the site. Required
            key is `name`.
        uuid: Set specific UUID for the file. If not provided, a new UUID will
            be generated.
        date: Expected date of the measurements as YYYY-MM-DD or datetime.date
            object. If not provided, the date will be inferred from the input
            file(s).

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.instruments import rd802nc
        >>> site_meta = {'name': 'Campina', 'altitude': 30, 'latitude': -2.18,
            'longitude': -59.02}
        >>> uuid = rd802nc('RD-220101-181400.txt', 'rd80.nc', site_meta)
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = get_uuid(uuid)
    if isinstance(input_file, str | PathLike):
        input_file = [input_file]
    disdrometer = Rd80(input_file, site_meta, date)
    disdrometer.sort_and_dedup_timestamps()
    disdrometer.convert_to_cloudnet_arrays()
    disdrometer.add_meta()
    attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    output.save_level1b(disdrometer, output_file, uuid)
    return uuid


class Rd80(CloudnetInstrument):
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
        self.n_velocity = 20
        self.n_diameter = 20
        self.serial_number = None
        self.instrument = instruments.RD80

    def _read_data(self, filenames: Iterable[str | PathLike]) -> None:
        times = []
        data = defaultdict(list)
        for filename in filenames:
            file_time, file_data = read_rd80(filename)
            times.append(file_time)
            for key, value in file_data.items():
                data[key].append(value)
        self.raw_time = np.concatenate(times)
        self.raw_data = {key: np.concatenate(value) for key, value in data.items()}

    def _screen_time(self, expected_date: datetime.date | None = None) -> None:
        if expected_date is None:
            self.date = self.time[0].date()
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
        mmh_to_ms = SEC_IN_HOUR / MM_TO_M
        mm_to_m = 1000
        hour = (
            self.raw_time - datetime.datetime.combine(self.date, datetime.time())
        ) / datetime.timedelta(hours=1)
        rainfall_rate = self.raw_data["RI [mm/h]"]
        n_particles = np.sum(self.raw_data["n"], axis=1)
        dt = self.raw_data["Interval [s]"]
        n = self.raw_data["n"]
        numcon = n / (A * dt[:, np.newaxis] * Vmid * Dspr)
        Z = np.sum(n / Vmid * Dmid**6, axis=1) / (A * dt)
        ZdB = 10 * ma.log10(ma.masked_where(Z == 0, Z))
        ZdB[ZdB < -10] = ma.masked
        self.data["diameter"] = CloudnetArray(Dmid / mm_to_m, "diameter")
        self.data["diameter_spread"] = CloudnetArray(Dspr / mm_to_m, "diameter_spread")
        self.data["diameter_bnds"] = CloudnetArray(
            np.dstack([Dlow, Dupp]) / mm_to_m, "diameter_bnds"
        )
        self.data["time"] = CloudnetArray(hour.astype(np.float32), "time")
        self.data["interval"] = CloudnetArray(dt, "interval")
        self.data["rainfall_rate"] = CloudnetArray(
            rainfall_rate / mmh_to_ms, "rainfall_rate"
        )
        self.data["n_particles"] = CloudnetArray(n_particles, "n_particles")
        self.data["number_concentration"] = CloudnetArray(
            numcon, "number_concentration"
        )
        self.data["radar_reflectivity"] = CloudnetArray(ZdB, "radar_reflectivity")
