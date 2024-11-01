import csv
import datetime
import math
from collections.abc import Iterable

import numpy as np
from numpy import ma

from cloudnetpy import output
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import HPA_TO_PA, MM_H_TO_M_S, SEC_IN_HOUR
from cloudnetpy.exceptions import ValidTimeStampError, WeatherStationDataError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.toa5 import read_toa5
from cloudnetpy.metadata import MetaData
from cloudnetpy.utils import datetime2decimal_hours


def ws2nc(
    weather_station_file: str | list[str],
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
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
        WeatherStationDataError : Unable to read the file.
        ValidTimeStampError: No valid timestamps found.
    """
    if not isinstance(weather_station_file, list):
        weather_station_file = [weather_station_file]
    try:
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
        else:
            msg = "Unsupported site"
            raise ValueError(msg)  # noqa: TRY301
        if date is not None:
            ws.screen_timestamps(date)
        ws.convert_time()
        ws.add_date()
        ws.add_site_geolocation()
        ws.add_data()
        ws.convert_temperature_and_humidity()
        ws.convert_pressure()
        ws.convert_rainfall_rate()
        ws.convert_rainfall_amount()
        ws.normalize_rainfall_amount()
        ws.calculate_rainfall_amount()
        attributes = output.add_time_attribute(ATTRIBUTES, ws.date)
        output.update_attributes(ws.data, attributes)
    except ValueError as err:
        raise WeatherStationDataError from err
    return output.save_level1b(ws, output_file, uuid)


class WS(CloudnetInstrument):
    def __init__(self, site_meta: dict):
        super().__init__()
        self._data: dict
        self.site_meta = site_meta
        self.instrument = instruments.GENERIC_WEATHER_STATION

    date: list[str]

    def add_date(self) -> None:
        first_date = self._data["time"][0].date()
        self.date = [
            str(first_date.year),
            str(first_date.month).zfill(2),
            str(first_date.day).zfill(2),
        ]

    def add_data(self) -> None:
        for key, value in self._data.items():
            parsed = datetime2decimal_hours(value) if key == "time" else ma.array(value)
            self.data[key] = CloudnetArray(parsed, key)

    def calculate_rainfall_amount(self) -> None:
        if "rainfall_amount" in self.data:
            return
        resolution = np.median(np.diff(self.data["time"].data)) * SEC_IN_HOUR
        rainfall_amount = ma.cumsum(self.data["rainfall_rate"].data * resolution)
        self.data["rainfall_amount"] = CloudnetArray(rainfall_amount, "rainfall_amount")

    def screen_timestamps(self, date: str) -> None:
        dates = np.array([str(d.date()) for d in self._data["time"]])
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
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate / 60 / 1000  # mm/min -> m/s

    def convert_pressure(self) -> None:
        self.data["air_pressure"].data = self.data["air_pressure"][:] * HPA_TO_PA

    def normalize_rainfall_amount(self) -> None:
        if "rainfall_amount" in self.data:
            amount = self.data["rainfall_amount"][:]
            offset = 0
            for i in range(1, len(amount)):
                if amount[i] + offset < amount[i - 1]:
                    offset += amount[i - 1]
                amount[i] += offset
            amount -= amount[0]
            self.data["rainfall_amount"].data = amount

    def convert_time(self) -> None:
        pass

    def convert_rainfall_amount(self) -> None:
        pass


class PalaiseauWS(WS):
    def __init__(self, filenames: list[str], site_meta: dict):
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
                if row.startswith("#"):
                    header_row = "".join(columns)
                    if header_row not in header:
                        header.append(header_row)
                else:
                    timestamp = datetime.datetime.strptime(
                        columns[0], "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=datetime.timezone.utc)
                    values.append([timestamp] + [float(x) for x in columns[1:]])
                    timestamps.append(timestamp)

        self._validate_header(header)
        return {"time": timestamps, "values": values}

    def convert_time(self) -> None:
        decimal_hours = datetime2decimal_hours(self._data["time"])
        self.data["time"] = CloudnetArray(decimal_hours, "time")

    def screen_timestamps(self, date: str) -> None:
        dates = [str(d.date()) for d in self._data["time"]]
        valid_ind = [ind for ind, d in enumerate(dates) if d == date]
        if not valid_ind:
            raise ValidTimeStampError
        for key in self._data:
            self._data[key] = [
                x for ind, x in enumerate(self._data[key]) if ind in valid_ind
            ]

    def add_data(self) -> None:
        keys = (
            "wind_speed",
            "wind_direction",
            "air_temperature",
            "relative_humidity",
            "air_pressure",
            "rainfall_rate",
            "rainfall_amount",
        )
        for ind, key in enumerate(keys):
            array = [row[ind + 1] for row in self._data["values"]]
            array_masked = ma.masked_invalid(array)
            self.data[key] = CloudnetArray(array_masked, key)

    def convert_rainfall_amount(self) -> None:
        self.data["rainfall_amount"].data = (
            self.data["rainfall_amount"][:] / 1000
        )  # mm -> m

    @staticmethod
    def _validate_header(header: list[str]) -> None:
        expected_identifiers = [
            "DateTime(yyyy-mm-ddThh:mm:ssZ)",
            "Windspeed(m/s)",
            "Winddirection(deg",
            "Airtemperature",
            "Relativehumidity(%)",
            "Pressure(hPa)",
            "Precipitationrate(mm/min)",
            "precipitation",
        ]
        column_titles = [row for row in header if "Col." in row]
        error_msg = "Unexpected weather station file format"
        if len(column_titles) != len(expected_identifiers):
            raise ValueError(error_msg)
        for title, identifier in zip(column_titles, expected_identifiers, strict=True):
            if identifier not in title:
                raise ValueError(error_msg)


class BucharestWS(PalaiseauWS):
    def convert_rainfall_rate(self) -> None:
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate * MM_H_TO_M_S


class GranadaWS(WS):
    def __init__(self, filenames: list[str], site_meta: dict):
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
        units, process, rows = read_toa5(self.filename)
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
    def __init__(self, filenames: list[str], site_meta: dict):
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
                            parsed_value = datetime.datetime.strptime(
                                value, "%Y-%m-%d %H:%M:%S"
                            ) - datetime.timedelta(hours=2)
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

    def __init__(self, filenames: list[str], site_meta: dict):
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
    def __init__(self, filenames: list[str], site_meta: dict):
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

        def read_value(keys: Iterable[str]):
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
            "rainfall_rate": read_value(["Precipitations", "Precipitatii"]),
            "wind_speed": read_value(["Wind_speed", "Viteza_vant"]),
            "wind_direction": read_value(["Wind_direction", "Directie_vant"]),
        }
        return self.format_data(data)

    def add_data(self) -> None:
        # Skip wind measurements where range was limited to 0-180 degrees
        if datetime.date(*map(int, self.date)) < datetime.date(2024, 10, 29):
            del self._data["wind_speed"]
            del self._data["wind_direction"]
        return super().add_data()

    def convert_pressure(self) -> None:
        mmHg2Pa = 133.322
        self.data["air_pressure"].data = self.data["air_pressure"][:] * mmHg2Pa


ATTRIBUTES = {
    "rainfall_amount": MetaData(
        long_name="Rainfall amount",
        standard_name="thickness_of_rainfall_amount",
        units="m",
        comment="Cumulated precipitation since 00:00 UTC",
    ),
}
