import datetime
from collections import defaultdict
from os import PathLike
from typing import Any

import numpy as np
from numpy import ma

from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import MM_TO_M, SEC_IN_HOUR
from cloudnetpy.exceptions import DisdrometerDataError, ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.toa5 import read_toa5

from .common import ATTRIBUTES, Disdrometer

TELEGRAM4 = [
    (1, "_serial_number"),
    (2, "_software_version"),
    (3, "_date"),
    (4, "_time"),
    (5, "_synop_5min_ww"),
    (6, "_synop_5min_WaWa"),
    (7, "_metar_5min_4678"),
    (8, "_rainfall_rate_5min"),
    (9, "synop_WW"),  # 1min
    (10, "synop_WaWa"),  # 1min
    (11, "_metar_1_min_4678"),
    (12, "rainfall_rate_1min_total"),
    (13, "rainfall_rate"),  # liquid, mm h-1
    (14, "rainfall_rate_1min_solid"),
    (15, "_precipition_amount"),  # mm
    (16, "visibility"),
    (17, "radar_reflectivity"),
    (18, "measurement_quality"),
    (19, "maximum_hail_diameter"),
    (20, "status_laser"),
    (21, "static_signal"),
    (22, "status_T_laser_analogue"),
    (23, "status_T_laser_digital"),
    (24, "status_I_laser_analogue"),
    (25, "status_I_laser_digital"),
    (26, "status_sensor_supply"),
    (27, "status_laser_heating"),
    (28, "status_receiver_heating"),
    (29, "status_temperature_sensor"),
    (30, "status_heating_supply"),
    (31, "status_heating_housing"),
    (32, "status_heating_heads"),
    (33, "status_heating_carriers"),
    (34, "status_laser_power"),
    (35, "_status_reserve"),
    (36, "T_interior"),
    (37, "T_laser_driver"),  # 0-80 C
    (38, "I_mean_laser"),
    (39, "V_control"),  # mV 4005-4015
    (40, "V_optical_output"),  # mV 2300-6500
    (41, "V_sensor_supply"),  # 1/10V
    (42, "I_heating_laser_head"),  # mA
    (43, "I_heating_receiver_head"),  # mA
    (44, "T_ambient"),  # C
    (45, "_V_heating_supply"),
    (46, "_I_housing"),
    (47, "_I_heating_heads"),
    (48, "_I_heating_carriers"),
    (49, "n_particles"),
]


def thies2nc(
    disdrometer_file: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | datetime.date | None = None,
) -> str:
    """Converts Thies-LNM disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

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
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    try:
        disdrometer = Thies(disdrometer_file, site_meta, date)
    except (ValueError, IndexError) as err:
        msg = "Unable to read disdrometer file"
        raise DisdrometerDataError(msg) from err
    disdrometer.sort_timestamps()
    disdrometer.remove_duplicate_timestamps()
    disdrometer.mask_invalid_values()
    disdrometer.add_meta()
    disdrometer.convert_units()
    attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    return output.save_level1b(disdrometer, output_file, uuid)


class Thies(Disdrometer):
    def __init__(
        self,
        filename: str | PathLike,
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ):
        super().__init__()
        self.instrument = instruments.THIES
        self.n_velocity = 20
        self.n_diameter = 22
        self.site_meta = site_meta
        self.raw_data: dict[str, Any] = defaultdict(list)
        self._read_data(filename)
        self._screen_time(expected_date)
        self.data = {}
        self._append_data()
        self._create_velocity_vectors()
        self._create_diameter_vectors()

    def convert_units(self) -> None:
        mmh_to_ms = SEC_IN_HOUR / MM_TO_M
        c_to_k = 273.15
        self._convert_data(("rainfall_rate_1min_total",), mmh_to_ms)
        self._convert_data(("rainfall_rate",), mmh_to_ms)
        self._convert_data(("rainfall_rate_1min_solid",), mmh_to_ms)
        self._convert_data(("diameter", "diameter_spread", "diameter_bnds"), 1e3)
        self._convert_data(("V_sensor_supply",), 10)
        self._convert_data(("I_mean_laser",), 100)
        self._convert_data(("T_interior",), c_to_k, method="add")
        self._convert_data(("T_ambient",), c_to_k, method="add")
        self._convert_data(("T_laser_driver",), c_to_k, method="add")

    def _read_data(self, filename: str | PathLike) -> None:
        with open(filename, errors="ignore") as file:
            first_line = file.readline()
        if "TOA5" in first_line:
            units, process, rows = read_toa5(filename)
            for row in rows:
                self._read_line(row["RawString"], row["TIMESTAMP"])
        elif first_line.lower().startswith("datetime [utc];"):
            with open(filename, errors="ignore") as file:
                first_line = file.readline()
                for line in file:
                    timestamp, telegram = line.split(";", maxsplit=1)
                    fixed_telegram = telegram.strip().rstrip(";") + ";"
                    parsed_timestamp = datetime.datetime.strptime(
                        timestamp, "%Y-%m-%d %H:%M:%S"
                    )
                    self._read_line(fixed_telegram, parsed_timestamp)
        else:
            with open(filename, errors="ignore") as file:
                for line in file:
                    self._read_line(line)
        if len(self.raw_data["time"]) == 0:
            raise ValidTimeStampError
        for key, value in self.raw_data.items():
            array = np.array(value)
            if key == "time":
                array = array.astype("datetime64[s]")
            self.raw_data[key] = array

    def _append_data(self) -> None:
        for key, values in self.raw_data.items():
            if key.startswith("_"):
                continue
            name_out = key
            values_out = values
            match key:
                case "spectrum":
                    name_out = "data_raw"
                    dimensions = ["time", "diameter", "velocity"]
                case "time":
                    dimensions = []
                    base = values[0].astype("datetime64[D]")
                    values_out = (values - base) / np.timedelta64(1, "h")
                case _:
                    dimensions = ["time"]
            self.data[name_out] = CloudnetArray(
                values_out, name_out, dimensions=dimensions
            )

        first_id = self.raw_data["_serial_number"][0]
        for sensor_id in self.raw_data["_serial_number"]:
            if sensor_id != first_id:
                msg = "Multiple serial numbers are not supported"
                raise DisdrometerDataError(msg)
        self.serial_number = first_id

    def _read_line(self, line: str, timestamp: datetime.datetime | None = None):
        raw_values = line.split(";")
        # Support custom truncated format used in Leipzig LIM.
        expected_columns = self.site_meta.get("truncate_columns", 521)
        if len(raw_values) != expected_columns:
            return
        for i, key in TELEGRAM4:
            if i >= expected_columns - 1:
                break
            value: Any
            if key == "_date":
                value = _parse_date(raw_values[i])
            elif key == "_time":
                value = _parse_time(raw_values[i])
            elif key in (
                "I_heating",
                "T_ambient",
                "T_interior",
                "T_laser_driver",
                "V_power_supply",
                "_precipition_amount",
                "_rainfall_rate_5min",
                "maximum_hail_diameter",
                "radar_reflectivity",
                "rainfall_rate",
                "rainfall_rate_1min_solid",
                "rainfall_rate_1min_total",
            ):
                value = float(raw_values[i])
            elif key in (
                "_serial_number",
                "_software_version",
                "_metar_5min_4678",
                "_metar_1_min_4678",
            ):
                value = raw_values[i]
            else:
                value = int(raw_values[i])
            self.raw_data[key].append(value)
        if expected_columns > 79:
            self.raw_data["spectrum"].append(
                np.array(list(map(int, raw_values[79:-2])), dtype="i2").reshape(
                    self.n_diameter, self.n_velocity
                )
            )
        if timestamp is not None:
            self.raw_data["time"].append(timestamp)
        else:
            self.raw_data["time"].append(
                datetime.datetime.combine(
                    self.raw_data["_date"][-1], self.raw_data["_time"][-1]
                )
            )

    def _screen_time(self, expected_date: datetime.date | None = None) -> None:
        if expected_date is None:
            self.date = self.raw_data["time"][0].astype(object).date()
            return
        self.date = expected_date
        valid_mask = self.raw_data["time"].astype("datetime64[D]") == self.date
        if np.count_nonzero(valid_mask) == 0:
            msg = f"No data found on {expected_date}"
            raise DisdrometerDataError(msg)
        for key in self.raw_data:
            self.raw_data[key] = self.raw_data[key][valid_mask]

    def mask_invalid_values(self) -> None:
        rainfall_rate = self.data["rainfall_rate"]
        rainfall_rate.data = ma.masked_where(
            rainfall_rate.data > 999, rainfall_rate.data
        )

    def _create_velocity_vectors(self) -> None:
        n_values = [5, 6, 7, 1, 1]
        spreads = [0.2, 0.4, 0.8, 1, 10]
        self.store_vectors(n_values, spreads, "velocity")

    def _create_diameter_vectors(self) -> None:
        n_values = [3, 6, 13]
        spreads = [0.125, 0.25, 0.5]
        self.store_vectors(n_values, spreads, "diameter", start=0.125)


def _parse_date(date: str) -> datetime.date:
    day, month, year = map(int, date.split("."))
    if year < 100:
        year += 2000
    return datetime.date(year, month, day)


def _parse_time(time: str) -> datetime.time:
    hour, minute, second = map(int, time.split(":"))
    return datetime.time(hour, minute, second)
