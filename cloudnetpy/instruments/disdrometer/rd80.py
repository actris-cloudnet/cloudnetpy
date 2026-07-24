import datetime
from collections import defaultdict
from collections.abc import Iterable
from os import PathLike
from uuid import UUID

import numpy as np

from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray, MetaData
from cloudnetpy.constants import MM_H_TO_M_S, MM_TO_M
from cloudnetpy.disdronator.process import process_l2
from cloudnetpy.disdronator.rd80 import read_rd80, read_rd80_l1
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.utils import get_uuid

from .common import ATTRIBUTES


def rd802nc(
    input_file: str | PathLike | Iterable[str | PathLike],
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
    return _process_disdrometer(
        Rd80, RD80_ATTRIBUTES, input_file, output_file, site_meta, uuid, date
    )


def _process_disdrometer(
    klass,
    attributes: dict[str, MetaData],
    input_file: str | PathLike | Iterable[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = get_uuid(uuid)
    if isinstance(input_file, str | PathLike):
        input_file = [input_file]
    disdrometer = klass(input_file, site_meta, date)
    disdrometer.sort_and_dedup_timestamps()
    disdrometer.convert_to_cloudnet_arrays()
    disdrometer.add_meta()
    attributes = output.add_time_attribute(attributes, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    output.save_level1b(disdrometer, output_file, uuid)
    return uuid


class Disdro(CloudnetInstrument):
    def __init__(
        self,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__()
        self.site_meta = site_meta
        self._process_data(filenames)
        self._screen_time(expected_date)

    def _process_data(self, filenames):
        l1 = self._read_data(filenames)
        l2 = process_l2(l1)
        self.raw_time = l2.time
        self.raw_meta = {
            "diameter": l2.diameter * MM_TO_M,
            "diameter_spread": l2.diameter_spread * MM_TO_M,
            "velocity": l2.velocity,
            "sampling_area": l2.area,
        }
        self.raw_data = {
            "interval": l2.interval,
            "data_raw": l2.data_raw,
            "n_particles": l2.n_particles,
            "number_concentration": l2.number_concentration,
            "fall_velocity": l2.fall_velocity,
            "rainfall_rate": l2.rain_rate * MM_H_TO_M_S,
            "rainfall_amount": l2.rain_accum * MM_TO_M,
            "radar_reflectivity": l2.radar_refl,
            "kinetic_energy": l2.energy_flux,
        }

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
        for key, value in self.raw_meta.items():
            self.data[key] = CloudnetArray(value, key)
        for key, value in self.raw_data.items():
            self.data[key] = CloudnetArray(value, key)
        hour = (
            self.raw_time - datetime.datetime.combine(self.date, datetime.time())
        ) / datetime.timedelta(hours=1)
        self.data["time"] = CloudnetArray(hour.astype(np.float32), "time")


class Rd80(Disdro):
    def __init__(
        self,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(filenames, site_meta, expected_date)
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
        raw_time = np.concatenate(times)
        raw_data = {key: np.concatenate(value) for key, value in data.items()}
        return read_rd80_l1(raw_time, raw_data)


RD80_ATTRIBUTES = ATTRIBUTES | {
    "data_raw": MetaData(
        long_name="Raw data as a function of particle diameter",
        units="1",
        dimensions=("time", "diameter"),
    ),
}
