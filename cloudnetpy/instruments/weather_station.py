import csv
import datetime
import math

import numpy as np
from numpy import ma

from cloudnetpy import output
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import SEC_IN_HOUR
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
        elif site_meta["name"] == "Granada":
            ws = GranadaWS(weather_station_file, site_meta)
        elif site_meta["name"] == "Kenttärova":
            ws = KenttarovaWS(weather_station_file, site_meta)
        else:
            msg = "Unsupported site"
            raise ValueError(msg)  # noqa: TRY301
        if date is not None:
            ws.screen_timestamps(date)
        ws.convert_time()
        ws.add_date()
        ws.add_site_geolocation()
        ws.add_data()
        ws.convert_units()
        ws.calculate_rainfall_amount()
        attributes = output.add_time_attribute(ATTRIBUTES, ws.date)
        output.update_attributes(ws.data, attributes)
    except ValueError as err:
        raise WeatherStationDataError from err
    return output.save_level1b(ws, output_file, uuid)


class WS(CloudnetInstrument):
    date: list[str]

    def convert_time(self) -> None:
        pass

    def screen_timestamps(self, date: str) -> None:
        pass

    def add_date(self) -> None:
        pass

    def add_data(self) -> None:
        pass

    def convert_units(self) -> None:
        pass

    def calculate_rainfall_amount(self) -> None:
        if "rainfall_amount" in self.data:
            return
        resolution = np.median(np.diff(self.data["time"].data)) * SEC_IN_HOUR
        rainfall_amount = ma.cumsum(self.data["rainfall_rate"].data * resolution)
        self.data["rainfall_amount"] = CloudnetArray(rainfall_amount, "rainfall_amount")


class PalaiseauWS(WS):
    def __init__(self, filenames: list[str], site_meta: dict):
        super().__init__()
        if len(filenames) != 1:
            raise ValueError
        self.filename = filenames[0]
        self.site_meta = site_meta
        self.instrument = instruments.GENERIC_WEATHER_STATION
        self._data = self._read_data()

    def _read_data(self) -> dict:
        timestamps, values, header = [], [], []
        with open(self.filename, encoding="latin-1") as f:
            data = f.readlines()
        for row in data:
            splat = row.split()
            try:
                timestamp = datetime.datetime.strptime(
                    splat[0],
                    "%Y-%m-%dT%H:%M:%SZ",
                ).replace(tzinfo=datetime.timezone.utc)
                temp: list[str | float] = list(splat)
                temp[1:] = [float(x) for x in temp[1:]]
                values.append(temp)
                timestamps.append(timestamp)
            except ValueError:
                header.append("".join(splat))

        # Simple validation for now:
        expected_identifiers = [
            "DateTime(yyyy-mm-ddThh:mm:ssZ)",
            "Windspeed(m/s)",
            "Winddirection(degres)",
            "Airtemperature(°C)",
            "Relativehumidity(%)",
            "Pressure(hPa)",
            "Precipitationrate(mm/min)",
            "24-hrcumulatedprecipitationsince00UT(mm)",
        ]
        column_titles = [row for row in header if "Col." in row]
        error_msg = "Unexpected weather station file format"
        if len(column_titles) != len(expected_identifiers):
            raise ValueError(error_msg)
        for title, identifier in zip(column_titles, expected_identifiers, strict=True):
            if identifier not in title:
                raise ValueError(error_msg)
        return {"timestamps": timestamps, "values": values}

    def convert_time(self) -> None:
        decimal_hours = datetime2decimal_hours(self._data["timestamps"])
        self.data["time"] = CloudnetArray(decimal_hours, "time")

    def screen_timestamps(self, date: str) -> None:
        dates = [str(d.date()) for d in self._data["timestamps"]]
        valid_ind = [ind for ind, d in enumerate(dates) if d == date]
        if not valid_ind:
            raise ValidTimeStampError
        for key in self._data:
            self._data[key] = [
                x for ind, x in enumerate(self._data[key]) if ind in valid_ind
            ]

    def add_date(self) -> None:
        first_date = self._data["timestamps"][0].date()
        self.date = [
            str(first_date.year),
            str(first_date.month).zfill(2),
            str(first_date.day).zfill(2),
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

    def convert_units(self) -> None:
        temperature_kelvins = atmos_utils.c2k(self.data["air_temperature"][:])
        self.data["air_temperature"].data = temperature_kelvins
        self.data["relative_humidity"].data = self.data["relative_humidity"][:] / 100
        self.data["air_pressure"].data = self.data["air_pressure"][:] * 100  # hPa -> Pa
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate / 60 / 1000  # mm/min -> m/s
        self.data["rainfall_amount"].data = (
            self.data["rainfall_amount"][:] / 1000
        )  # mm -> m


class GranadaWS(WS):
    def __init__(self, filenames: list[str], site_meta: dict):
        if len(filenames) != 1:
            raise ValueError
        super().__init__()
        self.filename = filenames[0]
        self.site_meta = site_meta
        self.instrument = instruments.GENERIC_WEATHER_STATION
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
                    parsed = float(value)
                data[keymap[key]].append(parsed)
        return data

    def convert_time(self) -> None:
        pass

    def screen_timestamps(self, date: str) -> None:
        dates = [str(d.date()) for d in self._data["time"]]
        valid_ind = [ind for ind, d in enumerate(dates) if d == date]
        if not valid_ind:
            raise ValidTimeStampError
        for key in self._data:
            self._data[key] = [
                x for ind, x in enumerate(self._data[key]) if ind in valid_ind
            ]

    def add_date(self) -> None:
        first_date = self._data["time"][0].date()
        self.date = [
            str(first_date.year),
            str(first_date.month).zfill(2),
            str(first_date.day).zfill(2),
        ]

    def add_data(self) -> None:
        for key, value in self._data.items():
            parsed = datetime2decimal_hours(value) if key == "time" else np.array(value)
            self.data[key] = CloudnetArray(parsed, key)

    def convert_units(self) -> None:
        temperature_kelvins = atmos_utils.c2k(self.data["air_temperature"][:])
        self.data["air_temperature"].data = temperature_kelvins
        self.data["relative_humidity"].data = self.data["relative_humidity"][:] / 100
        self.data["air_pressure"].data = self.data["air_pressure"][:] * 100  # hPa -> Pa
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate / 60 / 1000  # mm/min -> m/s


class KenttarovaWS(WS):
    def __init__(self, filenames: list[str], site_meta: dict):
        super().__init__()
        self.filenames = filenames
        self.site_meta = site_meta
        self.instrument = instruments.GENERIC_WEATHER_STATION
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
        for key, value in merged.items():
            new_value = np.array(value)
            if key != "time":
                new_value = ma.masked_where(np.isnan(new_value), new_value)
            merged[key] = new_value
        return merged

    def convert_time(self) -> None:
        pass

    def screen_timestamps(self, date: str) -> None:
        dates = [str(d.date()) for d in self._data["time"]]
        valid_ind = [ind for ind, d in enumerate(dates) if d == date]
        if not valid_ind:
            raise ValidTimeStampError
        for key in self._data:
            self._data[key] = [
                x for ind, x in enumerate(self._data[key]) if ind in valid_ind
            ]

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

    def convert_units(self) -> None:
        temperature_kelvins = atmos_utils.c2k(self.data["air_temperature"][:])
        self.data["air_temperature"].data = temperature_kelvins
        self.data["relative_humidity"].data = self.data["relative_humidity"][:] / 100
        self.data["air_pressure"].data = self.data["air_pressure"][:] * 100  # hPa -> Pa
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = (
            rainfall_rate / 3600 / 10 / 1000
        )  # not sure about units


ATTRIBUTES = {
    "rainfall_amount": MetaData(
        long_name="Rainfall amount",
        standard_name="thickness_of_rainfall_amount",
        units="m",
        comment="Cumulated precipitation since 00:00 UTC",
    ),
}
