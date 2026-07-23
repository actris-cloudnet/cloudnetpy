import datetime
import functools
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from os import PathLike
from typing import Any, TypeAlias
from uuid import UUID

import numpy as np
import numpy.typing as npt

from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import MM2_TO_M2, MM_H_TO_M_S, MM_TO_M
from cloudnetpy.disdronator.lpm import read_lpm, read_lpm_l1
from cloudnetpy.disdronator.parsivel import read_parsivel, read_parsivel_l1
from cloudnetpy.disdronator.process import DisdroL1, process_l2
from cloudnetpy.disdronator.rd80 import read_rd80, read_rd80_l1
from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.metadata import MetaData
from cloudnetpy.utils import get_uuid


def parsivel2nc(
    disdrometer_file: str | PathLike | Iterable[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
    telegram: Sequence[int | str | None] | None = None,
    field_separator: str = ";",
    decimal_separator: str = ".",
) -> UUID:
    """Converts OTT Parsivel-2 disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        disdrometer_file: Filename of disdrometer file or list of filenames.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.
        telegram: List of measured value numbers as specified in section 11.2 of
            the instrument's operating instructions. Unknown values are indicated
            with None. Telegram is required if the input file doesn't contain a
            header.
        field_separator: Field separator.
        decimal_separator: Decimal separator.

    Returns:
        UUID of the generated file.

    Raises:
        DisdrometerDataError: Timestamps do not match the expected date, or unable
            to read the disdrometer file.

    Examples:
        >>> from cloudnetpy.instruments import parsivel2nc
        >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2,
        'longitude': 14.1}
        >>> uuid = parsivel2nc('parsivel.log', 'parsivel.nc', site_meta)

    """
    return _process_disdrometer(
        Parsivel,
        functools.partial(
            read_parsivel,
            telegram=telegram,
            field_separator=field_separator,
            decimal_separator=decimal_separator,
        ),
        read_parsivel_l1,
        ATTRIBUTES,
        disdrometer_file,
        output_file,
        site_meta,
        uuid,
        date,
    )


def thies2nc(
    disdrometer_file: str | PathLike | Iterable[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
    au: int | None = None,
) -> UUID:
    """Converts Thies LPM disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.
        au: Device-specific AU parameter that defines sampling area.

    Returns:
        UUID of the generated file.

    Raises:
        DisdrometerDataError: Timestamps do not match the expected date, or unable
            to read the disdrometer file.

    Examples:
        >>> from cloudnetpy.instruments import thies2nc
        >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2,
        'longitude': 14.1}
        >>> uuid = thies2nc('thies-lnm.log', 'thies-lnm.nc', site_meta)

    """
    return _process_disdrometer(
        Thies,
        read_lpm,
        functools.partial(read_lpm_l1, au=au),
        ATTRIBUTES,
        disdrometer_file,
        output_file,
        site_meta,
        uuid,
        date,
    )


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
        Rd80,
        read_rd80,
        read_rd80_l1,
        RD80_ATTRIBUTES,
        input_file,
        output_file,
        site_meta,
        uuid,
        date,
    )


L0Reader: TypeAlias = Callable[
    [str | PathLike], tuple[npt.NDArray, dict[Any, npt.NDArray]]
]
L1Reader: TypeAlias = Callable[[npt.NDArray, dict[Any, npt.NDArray]], DisdroL1]


class Disdrometer(CloudnetInstrument):
    def __init__(
        self,
        l0_reader: L0Reader,
        l1_reader: L1Reader,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__()
        self.l0_reader = l0_reader
        self.l1_reader = l1_reader
        self.site_meta = site_meta
        self._read_data(filenames)
        self._process_data()
        self._screen_time(expected_date)

    def _read_data(self, filenames: Iterable[str | PathLike]) -> None:
        times = []
        data = defaultdict(list)
        for filename in filenames:
            file_time, file_data = self.l0_reader(filename)
            times.append(file_time)
            for key, value in file_data.items():
                data[key].append(value)
        self.l0_time = np.concatenate(times)
        if len(self.l0_time) == 0:
            msg = "No data found"
            raise DisdrometerDataError(msg)
        self.l0_data = {key: np.concatenate(value) for key, value in data.items()}

    def _process_data(self) -> None:
        l1 = self.l1_reader(self.l0_time, self.l0_data)
        l2 = process_l2(l1)
        self.raw_time = l2.time
        diameter_bnds = np.stack([l2.diameter_bins[:-1], l2.diameter_bins[1:]], axis=1)
        velocity_bnds = np.stack([l2.velocity_bins[:-1], l2.velocity_bins[1:]], axis=1)
        self.raw_meta: dict[str, float | npt.NDArray] = {
            "diameter": l2.diameter * MM_TO_M,
            "diameter_spread": l2.diameter_spread * MM_TO_M,
            "diameter_bnds": diameter_bnds * MM_TO_M,
            "velocity": l2.velocity,
            "velocity_spread": l2.velocity_spread,
            "velocity_bnds": velocity_bnds,
            "nominal_area": l2.area_nom * MM2_TO_M2,
        }
        if l2.area_eff is not None:
            self.raw_meta["effective_area"] = l2.area_eff * MM2_TO_M2
        self.raw_data = {
            "interval": l2.interval,
            "data_raw": l2.data_raw.astype(np.int16),
            "n_particles": l2.n_particles,
            "number_concentration": l2.number_concentration,
            "fall_velocity": l2.fall_velocity,
            "rainfall_rate": l2.rain_rate * MM_H_TO_M_S,
            "rainfall_amount": l2.rain_accum * MM_TO_M,
            "radar_reflectivity": l2.radar_refl,
            "kinetic_energy": l2.energy_flux,
            "visibility": np.round(l2.visibility).astype(np.int32),
        }

    def _screen_time(self, expected_date: datetime.date | None = None) -> None:
        if expected_date is None:
            expected_date = self.raw_time[0].date()
        is_valid = [dt.date() == expected_date for dt in self.raw_time]
        if not np.any(is_valid):
            msg = "No data found for expected date"
            raise DisdrometerDataError(msg)
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

    def add_raw_data(self) -> None:
        pass


class Parsivel(Disdrometer):
    def __init__(
        self,
        l0_reader: L0Reader,
        l1_reader: L1Reader,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(l0_reader, l1_reader, filenames, site_meta, expected_date)
        self.instrument = instruments.PARSIVEL2

    def add_raw_data(self) -> None:
        if 3 in self.l0_data:
            self.raw_data["synop_WaWa"] = self.l0_data[3]
        if 4 in self.l0_data:
            self.raw_data["synop_WW"] = self.l0_data[4]
        if 13 in self.l0_data:
            self.serial_number = self.l0_data[13][0]


class Thies(Disdrometer):
    def __init__(
        self,
        l0_reader: L0Reader,
        l1_reader: L1Reader,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(l0_reader, l1_reader, filenames, site_meta, expected_date)
        self.instrument = instruments.THIES

    def add_raw_data(self) -> None:
        self.raw_data["synop_WaWa"] = self.l0_data[12]
        self.raw_data["synop_WW"] = self.l0_data[11]
        self.serial_number = self.l0_data[3][0]


class Rd80(Disdrometer):
    def __init__(
        self,
        l0_reader: L0Reader,
        l1_reader: L1Reader,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(l0_reader, l1_reader, filenames, site_meta, expected_date)
        self.instrument = instruments.RD80


def _process_disdrometer(
    klass: type[Disdrometer],
    l0_reader: L0Reader,
    l1_reader: L1Reader,
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
    disdrometer = klass(l0_reader, l1_reader, input_file, site_meta, date)
    disdrometer.add_raw_data()
    disdrometer.sort_and_dedup_timestamps()
    disdrometer.convert_to_cloudnet_arrays()
    disdrometer.add_meta()
    attributes = output.add_time_attribute(attributes, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    output.save_level1b(disdrometer, output_file, uuid)
    return uuid


ATTRIBUTES = {
    "velocity": MetaData(
        long_name="Center fall velocity of precipitation particles",
        units="m s-1",
        comment="Predefined velocity classes.",
        dimensions=("velocity",),
    ),
    "velocity_spread": MetaData(
        long_name="Width of velocity interval",
        units="m s-1",
        comment="Bin size of each velocity interval.",
        dimensions=("velocity",),
    ),
    "velocity_bnds": MetaData(
        long_name="Velocity bounds",
        units="m s-1",
        comment="Upper and lower bounds of velocity interval.",
        dimensions=("velocity", "nv"),
    ),
    "diameter": MetaData(
        long_name="Center diameter of precipitation particles",
        units="m",
        comment="Predefined diameter classes.",
        dimensions=("diameter",),
    ),
    "diameter_spread": MetaData(
        long_name="Width of diameter interval",
        units="m",
        comment="Bin size of each diameter interval.",
        dimensions=("diameter",),
    ),
    "diameter_bnds": MetaData(
        long_name="Diameter bounds",
        units="m",
        comment="Upper and lower bounds of diameter interval.",
        dimensions=("diameter", "nv"),
    ),
    "rainfall_rate": MetaData(
        long_name="Rainfall rate",
        units="m s-1",
        standard_name="rainfall_rate",
        dimensions=("time",),
    ),
    "synop_WaWa": MetaData(
        long_name="Synop code WaWa", units="1", dimensions=("time",)
    ),
    "synop_WW": MetaData(long_name="Synop code WW", units="1", dimensions=("time",)),
    "radar_reflectivity": MetaData(
        long_name="Equivalent radar reflectivity factor",
        units="dBZ",
        standard_name="equivalent_reflectivity_factor",
        dimensions=("time",),
    ),
    "visibility": MetaData(
        long_name="Meteorological optical range (MOR) visibility",
        units="m",
        standard_name="visibility_in_air",
        comment="Visibility estimation by the disdrometer is valid\n"
        "only during precipitation events.",
        dimensions=("time",),
    ),
    "interval": MetaData(
        long_name="Length of measurement interval", units="s", dimensions=("time",)
    ),
    "n_particles": MetaData(
        long_name="Number of particles in time interval",
        units="1",
        dimensions=("time",),
    ),
    "number_concentration": MetaData(
        long_name="Number of particles per diameter class",
        units="m-3 mm-1",
        dimensions=("time", "diameter"),
    ),
    "fall_velocity": MetaData(
        long_name="Average velocity of each diameter class",
        units="m s-1",
        dimensions=("time", "diameter"),
    ),
    "data_raw": MetaData(
        long_name="Raw data as a function of particle diameter and velocity",
        units="1",
        dimensions=("time", "diameter", "velocity"),
    ),
    "kinetic_energy": MetaData(
        long_name="Kinetic energy of the hydrometeors",
        units="J m-2 h-1",
        dimensions=("time",),
    ),
    "nominal_area": MetaData(
        long_name="Nominal sampling area of the instrument",
        units="m2",
        dimensions=None,
    ),
    "effective_area": MetaData(
        long_name="Effective sampling area as a function of diameter",
        units="m2",
        dimensions=("diameter",),
    ),
}


RD80_ATTRIBUTES = ATTRIBUTES | {
    "data_raw": MetaData(
        long_name="Raw data as a function of particle diameter",
        units="1",
        dimensions=("time", "diameter"),
    ),
}
