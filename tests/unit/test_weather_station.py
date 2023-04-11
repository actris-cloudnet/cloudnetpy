import os
from tempfile import TemporaryDirectory

import pytest

from cloudnetpy.exceptions import ValidTimeStampError, WeatherStationDataError
from cloudnetpy.instruments import weather_station
from tests.unit.all_products_fun import Check

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SITE_META = {
    "name": "Kumpula",
    "latitude": 50,
    "longitude": 104.5,
    "altitude": 50,
}


class TestWeatherStation(Check):
    date = "2022-01-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/ws/palaiseau-ws.asc"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta)

    def test_global_attributes(self):
        assert self.nc.cloudnet_file_type == "weather-station"
        assert self.nc.title == f"Weather station from Kumpula"
        assert self.nc.year == "2022"
        assert self.nc.month == "01"
        assert self.nc.day == "01"
        assert self.nc.location == "Kumpula"

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 29


class TestDateArgument(Check):
    date = "2022-01-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/ws/palaiseau-ws.asc"
    site_meta = SITE_META
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_invalid_date(self):
        with pytest.raises(ValidTimeStampError):
            weather_station.ws2nc(
                self.filename,
                self.temp_path,
                SITE_META,
                date="2022-01-05",
            )


class TestTimestampScreening(Check):
    date = "2022-01-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/ws/palaiseau-ws.asc"
    site_meta = SITE_META
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 28


def test_invalid_file():
    filename = f"{SCRIPT_PATH}/data/parsivel/norunda.log"
    temp_path = TemporaryDirectory().name + "/test.nc"
    with pytest.raises(WeatherStationDataError):
        weather_station.ws2nc(filename, temp_path, SITE_META)


def test_invalid_header():
    filename = f"{SCRIPT_PATH}/data/ws/bad-ws.asc"
    temp_path = "test.nc"
    with pytest.raises(WeatherStationDataError):
        weather_station.ws2nc(filename, temp_path, SITE_META)


def test_invalid_header2():
    filename = f"{SCRIPT_PATH}/data/ws/bad-ws2.asc"
    temp_path = "test.nc"
    with pytest.raises(WeatherStationDataError):
        weather_station.ws2nc(filename, temp_path, SITE_META)
