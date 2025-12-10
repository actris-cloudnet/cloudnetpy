import csv
import datetime
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from os import PathLike
from uuid import UUID

import numpy as np
from numpy import ma

from cloudnetpy import output
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import HPA_TO_PA, MM_H_TO_M_S, SEC_IN_HOUR
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CSVFile
from cloudnetpy.instruments.toa5 import read_toa5
from cloudnetpy.metadata import MetaData
from cloudnetpy.utils import datetime2decimal_hours, get_uuid


def ws2nc(
    weather_station_file: str | PathLike | Sequence[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts weather station data into Cloudnet Level 1b netCDF file.

    Args:
        weather_station_file: Filename of weather-station ASCII file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.
    """
    if isinstance(weather_station_file, str | PathLike):
        weather_station_file = [weather_station_file]
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = get_uuid(uuid)
    ws: WS
    if site_meta["name"] == "Palaiseau":
        ws = PalaiseauWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Bucharest":
        ws = BucharestWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Granada":
        ws = GranadaWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Kenttärova":
        ws = KenttarovaWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Hyytiälä":
        ws = HyytialaWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Galați":
        ws = GalatiWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Jülich":
        ws = JuelichWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Lampedusa":
        ws = LampedusaWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Limassol":
        ws = LimassolWS(weather_station_file, site_meta)
    elif site_meta["name"] == "L'Aquila":
        ws = LAquilaWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Maïdo Observatory":
        ws = MaidoWS(weather_station_file, site_meta)
    elif site_meta["name"] == "Cluj-Napoca":
        ws = ClujWS(weather_station_file, site_meta)
    else:
        msg = "Unsupported site"
        raise ValueError(msg)
    if date is not None:
        ws.screen_timestamps(date)
    ws.convert_time()
    ws.add_date()
    ws.add_site_geolocation()
    ws.add_data()
    ws.remove_duplicate_timestamps()
    ws.convert_temperature_and_humidity()
    ws.convert_pressure()
    ws.convert_rainfall_rate()
    ws.convert_rainfall_amount()
    ws.normalize_cumulative_amount("rainfall_amount")
    ws.calculate_rainfall_amount()
    ws.wrap_wind_direction()
    attributes = output.add_time_attribute(ATTRIBUTES, ws.date)
    output.update_attributes(ws.data, attributes)
    output.save_level1b(ws, output_file, uuid)
    return uuid


class WS(CSVFile):
    def __init__(self, site_meta: dict) -> None:
        super().__init__(site_meta)
        self.instrument = instruments.GENERIC_WEATHER_STATION

    date: datetime.date

    def calculate_rainfall_amount(self) -> None:
        if "rainfall_amount" in self.data or "rainfall_rate" not in self.data:
            return
        time = self.data["time"].data
        if len(time) == 1:
            rainfall_amount = np.array([0])
        else:
            resolution = np.median(np.diff(time)) * SEC_IN_HOUR
            rainfall_amount = ma.cumsum(self.data["rainfall_rate"].data * resolution)
        self.data["rainfall_amount"] = CloudnetArray(rainfall_amount, "rainfall_amount")

    def screen_timestamps(self, date: datetime.date) -> None:
        dates = np.array([d.date() for d in self._data["time"]])
        valid_mask = dates == date
        if not valid_mask.any():
            raise ValidTimeStampError
        for key in self._data:
            self._data[key] = self._data[key][valid_mask]

    @staticmethod
    def format_data(data: dict) -> dict:
        for key, value in data.items():
            new_value = np.array(value)
            if key != "time":
                new_value = ma.masked_where(np.isnan(new_value), new_value)
            data[key] = new_value
        return data

    def convert_temperature_and_humidity(self) -> None:
        temperature_kelvins = atmos_utils.c2k(self.data["air_temperature"][:])
        self.data["air_temperature"].data = temperature_kelvins
        self.data["relative_humidity"].data = self.data["relative_humidity"][:] / 100

    def convert_rainfall_rate(self) -> None:
        if "rainfall_rate" not in self.data:
            return
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate / 60 / 1000  # mm/min -> m/s

    def convert_pressure(self) -> None:
        if "air_pressure" not in self.data:
            return
        self.data["air_pressure"].data = self.data["air_pressure"][:] * HPA_TO_PA

    def convert_time(self) -> None:
        pass

    def convert_rainfall_amount(self) -> None:
        pass

    def wrap_wind_direction(self) -> None:
        if "wind_direction" not in self.data:
            return
        # Wrap values little outside of [0, 360), keep original values
        # otherwise.
        threshold = 2
        values = self.data["wind_direction"].data
        values[(values > -threshold) & (values < 0)] += 360
        values[(values >= 360) & (values < 360 + threshold)] -= 360


class PalaiseauWS(WS):
    expected_header_identifiers: tuple[str, ...] = (
        "DateTime(yyyy-mm-ddThh:mm:ssZ)",
        "Windspeed(m/s)",
        "Winddirection(deg",
        "Airtemperature",
        "Relativehumidity(%)",
        "Pressure(hPa)",
        "Precipitationrate(mm/min)",
        "precipitation",
    )
    keys: tuple[str, ...] = (
        "wind_speed",
        "wind_direction",
        "air_temperature",
        "relative_humidity",
        "air_pressure",
        "rainfall_rate",
        "rainfall_amount",
    )

    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filenames = filenames
        self._data = self._read_data()

    def _read_data(self) -> dict:
        timestamps, values, header = [], [], []
        for filename in self.filenames:
            with open(filename, encoding="latin-1") as f:
                data = f.readlines()
            for row in data:
                if not (columns := row.split()):
                    continue
                if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", columns[0]):
                    if len(columns) != len(self.keys) + 1:
                        msg = (
                            f"Skipping row '{row.strip()}' due to unexpected "
                            "number of values"
                        )
                        logging.warning(msg)
                        continue
                    timestamp = datetime.datetime.strptime(
                        columns[0], "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=datetime.timezone.utc)
                    values.append([timestamp] + [float(x) for x in columns[1:]])
                    timestamps.append(timestamp)
                else:
                    header_row = "".join(columns)
                    if header_row not in header:
                        header.append(header_row)

        self._validate_header(header)
        return {"time": timestamps, "values": values}

    def convert_time(self) -> None:
        decimal_hours = datetime2decimal_hours(self._data["time"])
        self.data["time"] = CloudnetArray(decimal_hours, "time")

    def screen_timestamps(self, date: datetime.date) -> None:
        dates = [d.date() for d in self._data["time"]]
        valid_ind = [ind for ind, d in enumerate(dates) if d == date]
        if not valid_ind:
            raise ValidTimeStampError
        for key in self._data:
            self._data[key] = [
                x for ind, x in enumerate(self._data[key]) if ind in valid_ind
            ]

    def add_data(self) -> None:
        for ind, key in enumerate(self.keys):
            if key.startswith("_"):
                continue
            array = [row[ind + 1] for row in self._data["values"]]
            array_masked = ma.masked_invalid(array)
            self.data[key] = CloudnetArray(array_masked, key)

    def convert_rainfall_amount(self) -> None:
        self.data["rainfall_amount"].data = (
            self.data["rainfall_amount"][:] / 1000
        )  # mm -> m

    def _validate_header(self, header: list[str]) -> None:
        column_titles = [row for row in header if "Col." in row]
        error_msg = "Unexpected weather station file format"
        if len(column_titles) != len(self.expected_header_identifiers):
            raise ValueError(error_msg)
        for title, identifier in zip(
            column_titles, self.expected_header_identifiers, strict=True
        ):
            if identifier not in title:
                raise ValueError(error_msg)


class MaidoWS(PalaiseauWS):
    expected_header_identifiers = (
        "DateTimeyyyy-mm-ddThh:mm:ssZ",
        "Winddirection-average",
        "Windspeed-maximumvalue(m/s)",
        "Windspeed-average(m/s)",
        "Pressure-average(hPa)",
        "Relativehumidity-maximumvalue(%)",
        "Relativehumidity-average(%)",
        "Airtemperature-minimumvalue",
        "Airtemperature-average",
    )

    keys = (
        "wind_direction",
        "_wind_speed_max",
        "wind_speed",
        "air_pressure",
        "_relative_humidity_max",
        "relative_humidity",
        "_air_temperature_min",
        "air_temperature",
    )

    def convert_rainfall_amount(self) -> None:
        pass


class BucharestWS(PalaiseauWS):
    def convert_rainfall_rate(self) -> None:
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate * MM_H_TO_M_S


class GranadaWS(WS):
    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        if len(filenames) != 1:
            raise ValueError
        super().__init__(site_meta)
        self.filename = filenames[0]
        self._data = self._read_data()

    def _read_data(self) -> dict:
        keymap = {
            "TIMESTAMP": "time",
            "air_t_Avg": "air_temperature",
            "rh_Avg": "relative_humidity",
            "pressure_Avg": "air_pressure",
            "wind_speed_avg": "wind_speed",
            "wind_dir_avg": "wind_direction",
            "rain_Tot": "rainfall_rate",
        }
        expected_units = {
            "air_t_Avg": "degC",
            "rh_Avg": "%",
            "pressure_Avg": "hPa",
            "wind_speed_avg": "m/s",
            "wind_dir_avg": "Deg",
            "rain_Tot": "mm",
        }
        units, _process, rows = read_toa5(self.filename)
        for key in units:
            if key in expected_units and expected_units[key] != units[key]:
                msg = (
                    f"Expected {key} to have units {expected_units[key]},"
                    f" got {units[key]} instead"
                )
                raise ValueError(msg)

        data: dict[str, list] = {keymap[key]: [] for key in units if key in keymap}
        for row in rows:
            for key, value in row.items():
                if key not in keymap:
                    continue
                parsed = value
                if keymap[key] != "time":
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = math.nan
                data[keymap[key]].append(parsed)
        return self.format_data(data)


class KenttarovaWS(WS):
    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filenames = filenames
        self._data = self._read_data()

    def _read_data(self) -> dict:
        merged: dict = {}
        for filename in self.filenames:
            with open(filename, newline="") as f:
                reader = csv.DictReader(f)
                raw_data: dict = {key: [] for key in reader.fieldnames}  # type: ignore[union-attr]
                for row in reader:
                    for key, value in row.items():
                        parsed_value: float | datetime.datetime
                        if key == "Read time (UTC+2)":
                            try:
                                parsed_value = datetime.datetime.strptime(
                                    value, "%Y-%m-%d %H:%M:%S"
                                ) - datetime.timedelta(hours=2)
                            except ValueError:
                                break  # Should be first column, so skip whole row.
                        else:
                            try:
                                parsed_value = float(value)
                            except ValueError:
                                parsed_value = math.nan
                        raw_data[key].append(parsed_value)
            data = {
                "time": raw_data["Read time (UTC+2)"],
                "air_temperature": raw_data["Temp 2m (C)"],
                "relative_humidity": raw_data["Humidity 2m (%)"],
                "air_pressure": raw_data["Pressure (hPa)"],
                "wind_speed": raw_data["Wind speed (m/s)"],
                "wind_direction": raw_data["Wind dir (deg)"],
                "rainfall_rate": raw_data["Precipitation (?)"],
            }
            if merged:
                merged = {key: [*merged[key], *data[key]] for key in merged}
            else:
                merged = data
        return self.format_data(merged)

    def convert_rainfall_rate(self) -> None:
        # Rainfall rate is 10-minute averaged in mm h-1
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate * MM_H_TO_M_S / 10

    def convert_pressure(self) -> None:
        # Magic number 10 to convert to realistic Pa
        self.data["air_pressure"].data = self.data["air_pressure"][:] * 10


class HyytialaWS(WS):
    """Hyytiälä rain-gauge variables: a = Pluvio400 and b = Pluvio200.
    E.g.
    - AaRNRT/mm = amount of non-real-time rain total (Pluvio400) [mm]
    - BbRT/mm = Bucket content in real-time (Pluvio200) [mm].
    """

    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filename = filenames[0]
        self._data = self._read_data()

    def _read_data(self) -> dict:
        with open(self.filename, newline="") as f:
            # Skip first two lines
            for _ in range(2):
                next(f)
            # Read header
            header_line = f.readline().strip()
            fields = header_line[1:].strip().split()
            reader = csv.DictReader(
                f, delimiter=" ", skipinitialspace=True, fieldnames=fields
            )
            if reader.fieldnames is None:
                raise ValueError
            raw_data: dict = {key: [] for key in reader.fieldnames}
            raw_data["time"] = []
            # Read data
            for row in reader:
                for key, value in row.items():
                    if key:
                        parsed_value: float | datetime.datetime
                        if key == "y":
                            current_time = datetime.datetime(
                                int(value),
                                int(row["m"]),
                                int(row["d"]),
                                int(row["minute"]) // 60,
                                int(row["minute"]) % 60,
                            )
                            raw_data["time"].append(current_time)
                        else:
                            try:
                                parsed_value = float(value)
                            except (TypeError, ValueError):
                                parsed_value = math.nan
                            if parsed_value in (-99.99, -99.9):
                                parsed_value = math.nan
                            raw_data[key].append(parsed_value)

        data = {
            "time": raw_data["time"],
            "air_temperature": raw_data["Ta/dsC"],
            "relative_humidity": raw_data["RH/pcnt"],
            "air_pressure": raw_data["Pa/kPa"],
            "wind_speed": raw_data["WS/(m/s)"],
            "wind_direction": raw_data["WD/ds"],
            "rainfall_rate": raw_data["AaNRT/mm"],
        }
        return self.format_data(data)

    def convert_pressure(self) -> None:
        self.data["air_pressure"].data = (
            self.data["air_pressure"][:] * 1000
        )  # kPa to Pa


class GalatiWS(WS):
    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filename = filenames[0]
        self._data = self._read_data()

    def _read_data(self) -> dict:
        with open(self.filename, newline="") as f:
            reader = csv.DictReader(f)
            raw_data: dict = {key: [] for key in reader.fieldnames}  # type: ignore[union-attr]
            for row in reader:
                for key, value in row.items():
                    parsed_value: float | datetime.datetime
                    if key == "TimeStamp":
                        parsed_value = datetime.datetime.strptime(
                            value, "%Y-%m-%d %H:%M:%S.%f"
                        )
                    else:
                        try:
                            parsed_value = float(value)
                        except ValueError:
                            parsed_value = math.nan
                    raw_data[key].append(parsed_value)

        def read_value(keys: Iterable[str]) -> list:
            for key in keys:
                if key in raw_data:
                    return raw_data[key]
            raise KeyError("Didn't find any keys: " + ", ".join(keys))

        data = {
            "time": read_value(["TimeStamp"]),
            "air_temperature": read_value(["Temperature", "Temperatura"]),
            "relative_humidity": read_value(["RH", "Umiditate_relativa"]),
            "air_pressure": read_value(
                ["Atmospheric_pressure", "Presiune_atmosferica"]
            ),
            "rainfall_rate": read_value(
                ["Precipitations", "Precipitatii", "Precipitatii_Tot"]
            ),
            "wind_speed": read_value(["Wind_speed", "Viteza_vant"]),
            "wind_direction": read_value(["Wind_direction", "Directie_vant"]),
            "visibility": read_value(["Visibility", "Vizibilitate"]),
        }
        return self.format_data(data)

    def add_data(self) -> None:
        # Skip wind measurements where range was limited to 0-180 degrees
        if self.date < datetime.date(2024, 10, 29):
            del self._data["wind_speed"]
            del self._data["wind_direction"]
        self._data["visibility"] = self._data["visibility"].astype(np.int32)
        return super().add_data()

    def convert_pressure(self) -> None:
        mmHg2Pa = 133.322
        self.data["air_pressure"].data = self.data["air_pressure"][:] * mmHg2Pa


class JuelichWS(WS):
    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filename = filenames[0]
        self._data = self._read_data()

    def _read_data(self) -> dict:
        keymap = {
            "TIMESTAMP": "time",
            "AirTC_Avg": "air_temperature",
            "RH": "relative_humidity",
            "BV_BP_Avg": "air_pressure",
            "WS_ms_S_WVT": "wind_speed",
            "WindDir_D1_WVT": "wind_direction",
        }
        expected_units = {
            "AirTC_Avg": "Deg C",
            "RH": "%",
            "BV_BP_Avg": "hPa",
            "WS_ms_S_WVT": "meters/Second",
            "WindDir_D1_WVT": "Deg",
        }
        units, _process, rows = read_toa5(self.filename)
        for key in units:
            if key in expected_units and expected_units[key] != units[key]:
                msg = (
                    f"Expected {key} to have units {expected_units[key]},"
                    f" got {units[key]} instead"
                )
                raise ValueError(msg)

        data: dict[str, list] = {keymap[key]: [] for key in units if key in keymap}
        for row in rows:
            for key, value in row.items():
                if key not in keymap:
                    continue
                parsed = value
                if keymap[key] != "time":
                    parsed = float(value)
                data[keymap[key]].append(parsed)

        return self.format_data(data)


class LampedusaWS(WS):
    """Read Lampedusa weather station data in ICOS format."""

    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filename = filenames[0]
        self._data = self._read_data()

    def _read_data(self) -> dict:
        with open(self.filename, newline="") as f:
            fields = [
                "time",
                "str1",
                "str2",
                "T",
                "RH",
                "Td",
                "P",
                "WSi",
                "WDi",
                "WS10m",
                "WD10m",
                "rain1m",
                "rain2h",
                "empty",
            ]
            reader = csv.DictReader(f, fieldnames=fields)
            raw_data: dict = {key: [] for key in fields}
            for row in reader:
                for key, value in row.items():
                    fixed_value = value.strip("\0")
                    parsed_value: float | datetime.datetime
                    if key == "time":
                        parsed_value = datetime.datetime.strptime(
                            fixed_value, "%y%m%d %H%M%S"
                        )
                    else:
                        try:
                            parsed_value = float(fixed_value)
                        except ValueError:
                            parsed_value = math.nan
                    raw_data[key].append(parsed_value)

        data = {
            "time": raw_data["time"],
            "air_temperature": raw_data["T"],
            "relative_humidity": raw_data["RH"],
            "air_pressure": raw_data["P"],
            "wind_speed": raw_data["WSi"],
            "wind_direction": raw_data["WDi"],
            "rainfall_rate": raw_data["rain1m"],
        }
        return self.format_data(data)


class LimassolWS(WS):
    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filenames = filenames
        self._data = defaultdict(list)
        for filename in filenames:
            for key, values in _parse_sirta(filename).items():
                self._data[key].extend(values)
        self._data["time"] = self._data.pop("Date Time (yyyy-mm-ddThh:mm:ss)")

    def convert_time(self) -> None:
        decimal_hours = datetime2decimal_hours(self._data["time"])
        self.data["time"] = CloudnetArray(decimal_hours, "time")

    def screen_timestamps(self, date: datetime.date) -> None:
        dates = [d.date() for d in self._data["time"]]
        valid_ind = [ind for ind, d in enumerate(dates) if d == date]
        if not valid_ind:
            raise ValidTimeStampError
        for key in self._data:
            self._data[key] = [
                x for ind, x in enumerate(self._data[key]) if ind in valid_ind
            ]

    def add_data(self) -> None:
        self.data["air_temperature"] = CloudnetArray(
            np.array(self._data["Air temperature (°C)"]), "air_temperature"
        )
        self.data["relative_humidity"] = CloudnetArray(
            np.array(self._data["Relative humidity (%)"]), "relative_humidity"
        )
        self.data["rainfall_rate"] = CloudnetArray(
            np.array(self._data["Total precipitation (mm)"]), "rainfall_rate"
        )
        # Wind speed and direction are available since 2025-02-13:
        if (
            "Wind speed at 10m (m/s)" in self._data
            and "Wind direction at 10m (degrees)" in self._data
        ):
            self.data["wind_speed"] = CloudnetArray(
                np.array(self._data["Wind speed at 10m (m/s)"]), "wind_speed"
            )
            self.data["wind_direction"] = CloudnetArray(
                np.array(self._data["Wind direction at 10m (degrees)"]),
                "wind_direction",
            )
        else:
            self.data["wind_speed"] = CloudnetArray(
                np.array(self._data["Wind speed (m/s)"]), "wind_speed"
            )

    def convert_rainfall_rate(self) -> None:
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = (
            rainfall_rate / (10 * 60) / 1000
        )  # mm/(10 min) -> m/s


def _parse_sirta(filename: str | PathLike) -> dict:
    """Parse SIRTA-style weather station file."""
    with open(filename, "rb") as f:
        raw_content = f.read()
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_content.decode("latin-1")
    lines = [line.strip() for line in content.splitlines()]
    columns: list[str] = []
    output: dict = {}
    for line in lines:
        m = re.fullmatch(r"#\s*Col.\s*(\d+)\s*:\s*(.*)", line)
        if m is None:
            continue
        if m[1] != str(len(columns) + 1):
            msg = f"Expected column {m[1]}, found {len(columns) + 1}"
            raise ValueError(msg)
        columns.append(m[2])
        output[m[2]] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(columns) != len(values):
            continue
        for column, value in zip(columns, values, strict=False):
            parsed: float | datetime.datetime
            if column == "Date Time (yyyy-mm-ddThh:mm:ss)":
                parsed = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(
                    tzinfo=datetime.timezone.utc
                )
            elif column == "Date Time (yyyy-mm-ddThh:mm:ssZ)":
                parsed = datetime.datetime.strptime(
                    value, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=datetime.timezone.utc)
            else:
                parsed = float(value)
            output[column].append(parsed)
    return output


class LAquilaWS(WS):
    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filenames = filenames
        self._data = self._read_data()

    def _read_data(self) -> dict:
        data: dict[str, list] = {
            key: []
            for key in [
                "time",
                "air_temperature",
                "air_pressure",
                "relative_humidity",
                "rainfall_rate",
                "wind_speed",
                "wind_direction",
            ]
        }
        for filename in self.filenames:
            with open(filename) as f:
                for row in f:
                    if row.startswith("#"):
                        continue
                    columns = row.split(",")
                    if len(columns) != 7:
                        continue
                    timestamp = datetime.datetime.strptime(
                        columns[0], "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=datetime.timezone.utc)
                    data["time"].append(timestamp)
                    data["air_temperature"].append(self._parse_value(columns[1]))
                    data["air_pressure"].append(self._parse_value(columns[2]))
                    data["relative_humidity"].append(self._parse_value(columns[3]))
                    data["rainfall_rate"].append(self._parse_value(columns[4]))
                    data["wind_speed"].append(self._parse_value(columns[5]))
                    data["wind_direction"].append(self._parse_value(columns[6]))
        output = self.format_data(data)
        _, time_ind = np.unique(output["time"], return_index=True)
        for key in output:
            output[key] = output[key][time_ind]
        return output

    def _parse_value(self, value: str) -> float:
        value = value.strip()
        return float(value) if value else math.nan


class ClujWS(WS):
    def __init__(self, filenames: Sequence[str | PathLike], site_meta: dict) -> None:
        super().__init__(site_meta)
        self.filenames = filenames
        self._data = self._read_data()

    def _read_data(self) -> dict:
        with open(self.filenames[0]) as f:
            rows = f.readlines()
            headers = rows[0].strip().split("\t")
            raw_data: dict[str, list[str]] = {header: [] for header in headers}
            for row in rows[1:]:
                columns = row.strip().split("\t")
                for key, value in zip(headers, columns, strict=True):
                    raw_data[key].append(value)
        return self.format_data(
            {
                "time": [self._parse_datetime(x) for x in raw_data["DateTime"]],
                "air_temperature": [
                    self._parse_value(x) for x in raw_data["Air_temperature_C"]
                ],
                "air_pressure": [
                    self._parse_value(x) for x in raw_data["air_pressure_hPA"]
                ],
                "relative_humidity": [
                    self._parse_value(x) for x in raw_data["rel_humidity_pct"]
                ],
                "rainfall_rate": [
                    self._parse_value(x) for x in raw_data["Precipitation_mm"]
                ],
                "wind_speed": [self._parse_value(x) for x in raw_data["WS_azimuth_ms"]],
                "wind_direction": [
                    self._parse_value(x) for x in raw_data["WD_azimuth_deg"]
                ],
            }
        )

    def _parse_datetime(self, value: str) -> datetime.datetime:
        return datetime.datetime.strptime(value, "%d.%m.%y %H:%M:%S.%f").replace(
            tzinfo=datetime.timezone.utc
        )

    def _parse_value(self, value: str) -> float:
        value = value.strip()
        return float(value) if value else math.nan

    def convert_rainfall_rate(self) -> None:
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate / (
            1000 * 600
        )  # mm/10min => m/s


ATTRIBUTES = {
    "visibility": MetaData(
        long_name="Meteorological optical range (MOR) visibility",
        units="m",
        standard_name="visibility_in_air",
        dimensions=("time",),
    ),
}
