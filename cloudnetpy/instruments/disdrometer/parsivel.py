import datetime
import logging
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np

from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument

from .common import ATTRIBUTES, Disdrometer


def parsivel2nc(
    disdrometer_file: Path | str | bytes,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | datetime.date | None = None,
    telegram: Sequence[int | None] | None = None,
) -> str:
    """Converts OTT Parsivel-2 disdrometer data into Cloudnet Level 1b netCDF
    file.

    Args:
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.
        telegram: List of measured value numbers as specified in section 11.2 of
            the instrument's operating instructions. Unknown values are indicated
            with None. Telegram is required if the input file doesn't contain a
            header.

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
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    disdrometer = Parsivel(disdrometer_file, site_meta, telegram, date)
    disdrometer.sort_timestamps()
    disdrometer.remove_duplicate_timestamps()
    disdrometer.convert_units()
    disdrometer.add_meta()
    attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    uuid = output.save_level1b(disdrometer, output_file, uuid)
    return uuid


class Parsivel(CloudnetInstrument):
    def __init__(
        self,
        filename: Path | str | bytes,
        site_meta: dict,
        telegram: Sequence[int | None] | None = None,
        expected_date: datetime.date | None = None,
    ):
        super().__init__()
        self.site_meta = site_meta
        self.raw_data = _read_parsivel(filename, telegram)
        self._screen_time(expected_date)
        self.n_velocity = 32
        self.n_diameter = 32
        self.serial_number = None
        self.instrument = instruments.PARSIVEL2

        self._append_data()
        self._create_velocity_vectors()
        self._create_diameter_vectors()

    def _screen_time(self, expected_date: datetime.date | None = None):
        if expected_date is None:
            self.date = self.raw_data["time"][0].astype(object).date()
            return
        self.date = expected_date
        valid_mask = self.raw_data["time"].astype("datetime64[D]") == self.date
        if np.count_nonzero(valid_mask) == 0:
            raise DisdrometerDataError(f"No data found on {expected_date}")
        for key in self.raw_data:
            self.raw_data[key] = self.raw_data[key][valid_mask]

    def _append_data(self):
        for key, values in self.raw_data.items():
            if key.startswith("_"):
                continue
            match key:
                case "spectrum":
                    key = "data_raw"
                    dimensions = ["time", "diameter", "velocity"]
                case "number_concentration" | "fall_velocity":
                    dimensions = ["time", "diameter"]
                case "time":
                    dimensions = []
                    base = values[0].astype("datetime64[D]")
                    values = (values - base) / np.timedelta64(1, "h")
                case _:
                    dimensions = ["time"]
            self.data[key] = CloudnetArray(values, key, dimensions=dimensions)
        if "_sensor_id" in self.raw_data:
            first_id = self.raw_data["_sensor_id"][0]
            for sensor_id in self.raw_data["_sensor_id"]:
                if sensor_id != first_id:
                    raise DisdrometerDataError("Multiple sensor IDs are not supported")
            self.serial_number = first_id

    def _create_velocity_vectors(self):
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        Disdrometer.store_vectors(self.data, n_values, spreads, "velocity")

    def _create_diameter_vectors(self):
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.125, 0.25, 0.5, 1, 2, 3]
        Disdrometer.store_vectors(self.data, n_values, spreads, "diameter")

    def convert_units(self):
        mm_to_m = 1e3
        mmh_to_ms = 3600 * mm_to_m
        c_to_k = 273.15
        self._convert_data(("rainfall_rate",), mmh_to_ms)
        self._convert_data(("snowfall_rate",), mmh_to_ms)
        self._convert_data(("diameter", "diameter_spread", "diameter_bnds"), mm_to_m)
        self._convert_data(("V_sensor_supply",), 10)
        self._convert_data(("T_sensor",), c_to_k, method="add")

    def add_meta(self):
        valid_keys = ("latitude", "longitude", "altitude")
        for key, value in self.site_meta.items():
            key = key.lower()
            if key in valid_keys:
                self.data[key] = CloudnetArray(float(value), key)

    def _convert_data(
        self,
        keys: tuple[str, ...],
        value: float,
        method: Literal["divide", "add"] = "divide",
    ):
        for key in keys:
            if key not in self.data:
                continue
            variable = self.data[key]
            if method == "divide":
                variable.data = variable.data.astype("f4") / value
                variable.data_type = "f4"
            elif method == "add":
                variable.data = variable.data.astype("f4") + value
                variable.data_type = "f4"
            else:
                raise ValueError


HEADERS = {
    "Date": "_date",
    "Time": "_time",
    "Intensity of precipitation (mm/h)": "rainfall_rate",
    "Precipitation since start (mm)": "_rain_accum",
    "Radar reflectivity (dBz)": "radar_reflectivity",
    "MOR Visibility (m)": "visibility",
    "Signal amplitude of Laserband": "sig_laser",
    "Number of detected particles": "n_particles",
    "Temperature in sensor (°C)": "T_sensor",
    "Heating current (A)": "I_heating",
    "Sensor voltage (V)": "V_power_supply",
    "Kinetic Energy": "kinetic_energy",
    "Snow intensity (mm/h)": "snowfall_rate",
    "Weather code SYNOP WaWa": "synop_WaWa",
    "Weather code METAR/SPECI": "_metar",
    "Weather code NWS": "_nws",
    "Optics status": "state_sensor",
    "Spectrum": "spectrum",
}

TELEGRAM = {
    1: "rainfall_rate",
    2: "_rain_accum",
    3: "synop_WaWa",
    4: "synop_WW",
    5: "_metar_speci",
    6: "_nws",
    7: "radar_reflectivity",
    8: "visibility",
    9: "interval",
    10: "sig_laser",
    11: "n_particles",
    12: "T_sensor",
    13: "_sensor_id",
    14: "_iop_firmware_version",
    15: "_dsp_firmware_version",
    16: "I_heating",
    17: "V_power_supply",
    18: "state_sensor",
    19: "_datetime",
    20: "_time",
    21: "_date",
    22: "_station_name",
    23: "_station_number",
    24: "_rain_amount_absolute",
    25: "error_code",
    26: "_T_pcb",
    27: "_T_right",
    28: "_T_left",
    30: "rainfall_rate",
    31: "rainfall_rate",
    32: "_rain_accum",
    33: "radar_reflectivity",
    34: "kinetic_energy",
    35: "snowfall_rate",
    # 60, 61 = all particles detected
    90: "number_concentration",
    91: "fall_velocity",
    93: "spectrum",
}


T = TypeVar("T")


def _take(it: Iterator[T], n: int) -> Iterator[T]:
    for _ in range(n):
        yield next(it)  # pylint: disable=stop-iteration-return


def _parse_int(tokens: Iterator[str]) -> int:
    return int(next(tokens))


def _parse_float(tokens: Iterator[str]) -> float:
    token = next(tokens)
    token = token.replace(",", ".")
    return float(token)


def _parse_date(tokens: Iterator[str]) -> datetime.date:
    token = next(tokens)
    if "/" in token:
        year, month, day = token.split("/")
    elif "." in token:
        day, month, year = token.split(".")
    else:
        raise ValueError(f"Unsupported date: '{input}'")
    if len(year) != 4:
        raise ValueError(f"Unsupported date: '{input}'")
    return datetime.date(int(year), int(month), int(day))


def _parse_time(tokens: Iterator[str]) -> datetime.time:
    token = next(tokens)
    hour, minute, second = token.split(":")
    return datetime.time(int(hour), int(minute), int(second))


def _parse_datetime(tokens: Iterator[str]) -> datetime.datetime:
    token = next(tokens)
    year = int(token[:4])
    month = int(token[4:6])
    day = int(token[6:8])
    hour = int(token[8:10])
    minute = int(token[10:12])
    second = int(token[12:14])
    return datetime.datetime(year, month, day, hour, minute, second)


def _parse_vector(tokens: Iterator[str]) -> np.ndarray:
    return np.array([_parse_float(tokens) for _i in range(32)])


def _parse_spectrum(tokens: Iterator[str]) -> np.ndarray:
    first = next(tokens)
    if first == "<SPECTRUM>ZERO</SPECTRUM>":
        return np.zeros((32, 32), dtype="i2")
    if first.startswith("<SPECTRUM>"):
        raw = [first.removeprefix("<SPECTRUM>")]
        raw.extend(_take(tokens, 1023))
        if next(tokens) != "</SPECTRUM>":
            raise ValueError("Invalid spectrum format")
        values = [int(x) if x != "" else 0 for x in raw]
    else:
        values = [int(first)]
        values.extend(int(x) for x in _take(tokens, 1023))
    return np.array(values, dtype="i2").reshape((32, 32))


PARSERS: dict[str, Callable[[Iterator[str]], Any]] = {
    "I_heating": _parse_float,
    "T_sensor": _parse_int,
    "_T_pcb": _parse_int,
    "_T_right": _parse_int,
    "_T_left": _parse_int,
    "V_power_supply": _parse_float,
    "_date": _parse_date,
    "_rain_accum": _parse_float,
    "_rain_amount_absolute": _parse_float,
    "_time": _parse_time,
    "error_code": _parse_int,
    "fall_velocity": _parse_vector,
    "interval": _parse_int,
    "kinetic_energy": _parse_float,
    "n_particles": _parse_int,
    "number_concentration": _parse_vector,
    "_datetime": _parse_datetime,
    "radar_reflectivity": _parse_float,
    "rainfall_rate": _parse_float,
    "sig_laser": _parse_int,
    "snowfall_rate": _parse_float,
    "spectrum": _parse_spectrum,
    "state_sensor": _parse_int,
    "synop_WaWa": _parse_int,
    "synop_WW": _parse_int,
    "visibility": _parse_int,
}


def _parse_headers(line: str) -> list[str]:
    return [HEADERS[header.strip()] for header in line.split(";")]


def _parse_telegram(telegram: Sequence[int | None]) -> list[str]:
    return [
        TELEGRAM[num] if num is not None else f"_unknown_{i}"
        for i, num in enumerate(telegram)
    ]


def _read_rows(headers: list[str], rows: list[str]) -> dict[str, list]:
    result: dict[str, list] = {header: [] for header in headers}
    invalid_rows = 0
    for row in rows:
        if row == "":
            continue
        try:
            tokens = iter(row.removesuffix(";").split(";"))
            parsed = [PARSERS.get(header, next)(tokens) for header in headers]
            unread_tokens = list(tokens)
            if unread_tokens:
                raise ValueError("More values than expected")
            for header, value in zip(headers, parsed):
                result[header].append(value)
        except (ValueError, StopIteration):
            invalid_rows += 1
            continue
    if invalid_rows == len(rows):
        raise DisdrometerDataError("No valid data in file")
    if invalid_rows > 0:
        logging.info(f"Skipped {invalid_rows} invalid rows")
    return result


def _read_parsivel(
    filename: Path | str | bytes, telegram: Sequence[int | None] | None = None
) -> dict[str, np.ndarray]:
    with open(filename, encoding="latin1", errors="ignore") as file:
        lines = file.read().splitlines()
    if not lines:
        raise DisdrometerDataError("File is empty")
    if "Date" in lines[0]:
        headers = _parse_headers(lines[0])
        data = _read_rows(headers, lines[1:])
    elif telegram is not None:
        headers = _parse_telegram(telegram)
        data = _read_rows(headers, lines)
    else:
        raise ValueError("telegram must be specified for files without header")
    if "_datetime" not in data:
        data["_datetime"] = [
            datetime.datetime.combine(date, time)
            for date, time in zip(data["_date"], data["_time"])
        ]
    result = {key: np.array(value) for key, value in data.items()}
    result["time"] = result["_datetime"].astype("datetime64[s]")
    return result
