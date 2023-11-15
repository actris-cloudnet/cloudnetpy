import datetime

from numpy import ma

from cloudnetpy import output
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import ValidTimeStampError, WeatherStationDataError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.metadata import MetaData
from cloudnetpy.utils import datetime2decimal_hours


def ws2nc(
    weather_station_file: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts weather-station data into Cloudnet Level 1b netCDF file.

    Args:
    ----
        weather_station_file: Filename of weather-station ASCII file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
    -------
        UUID of the generated file.

    Raises:
    ------
        WeatherStationDataError : Unable to read the file.
        ValidTimeStampError: No valid timestamps found.
    """
    try:
        ws = WS(weather_station_file, site_meta)
        if date is not None:
            ws.screen_timestamps(date)
        ws.convert_time()
        ws.add_date()
        ws.add_site_geolocation()
        ws.add_data()
        ws.convert_units()
        attributes = output.add_time_attribute(ATTRIBUTES, ws.date)
        output.update_attributes(ws.data, attributes)
    except ValueError as err:
        raise WeatherStationDataError from err
    return output.save_level1b(ws, output_file, uuid)


class WS(CloudnetInstrument):
    def __init__(self, filename: str, site_meta: dict):
        super().__init__()
        self.filename = filename
        self.site_meta = site_meta
        self.date: list[str] = []
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
        self.data["rainfall_amount"].data = self.data["rainfall_amount"][:] / 1000


ATTRIBUTES = {
    "air_temperature": MetaData(
        long_name="Air temperature",
        standard_name="air_temperature",
        units="K",
    ),
    "air_pressure": MetaData(
        long_name="Air pressure",
        standard_name="air_pressure",
        units="Pa",
    ),
    "rainfall_amount": MetaData(
        long_name="Rainfall amount",
        standard_name="thickness_of_rainfall_amount",
        units="m",
        comment="Cumulated precipitation since 00:00 UTC",
    ),
}
