import os
from tempfile import TemporaryDirectory

import pytest
import numpy as np
from numpy import ma

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import weather_station
from cloudnetpy.cloudnetarray import CloudnetArray
from tests.unit.all_products_fun import Check

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SITE_META = {
    "name": "Palaiseau",
    "latitude": 50,
    "longitude": 104.5,
    "altitude": 50,
}


class WS(Check):
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"

    def test_pressure_values(self):
        if "air_pressure" not in self.nc.variables:
            return
        assert self.nc.variables["air_pressure"].units == "Pa"
        min_pressure = ma.min(self.nc.variables["air_pressure"][:])
        max_pressure = ma.max(self.nc.variables["air_pressure"][:])
        assert min_pressure > 90000
        assert max_pressure < 110000

    def test_wind_speed_values(self):
        if "wind_speed" not in self.nc.variables:
            return
        assert self.nc.variables["wind_speed"].units == "m s-1"
        min_wind_speed = ma.min(self.nc.variables["wind_speed"][:])
        assert min_wind_speed >= 0

    def test_wind_direction_values(self):
        if "wind_direction" not in self.nc.variables:
            return
        assert self.nc.variables["wind_direction"].units == "degree"
        min_wind_dir = ma.min(self.nc.variables["wind_direction"][:])
        max_wind_dir = ma.max(self.nc.variables["wind_direction"][:])
        assert min_wind_dir >= 0
        assert max_wind_dir < 360

    def test_rainfall_rate_values(self):
        if "rainfall_rate" not in self.nc.variables:
            return
        assert self.nc.variables["rainfall_rate"].units == "m s-1"
        min_rainfall = ma.min(self.nc.variables["rainfall_rate"][:])
        max_rainfall = ma.max(self.nc.variables["rainfall_rate"][:])
        assert min_rainfall >= 0
        assert max_rainfall <= 1.4e-6

    def test_rainfall_amount(self):
        if "rainfall_amount" not in self.nc.variables:
            return
        assert self.nc.variables["rainfall_amount"][0] == 0.0
        assert (np.diff(self.nc.variables["rainfall_amount"][:]) >= 0).all()

    def test_global_attributes(self):
        assert self.nc.cloudnet_file_type == "weather-station"
        assert self.nc.title == f"Weather station from {self.site_meta['name']}"
        assert self.nc.source == "Weather station"
        assert self.nc.year == self.date[:4]
        assert self.nc.month == self.date[5:7]
        assert self.nc.day == self.date[8:10]
        assert self.nc.location == self.site_meta["name"]


class TestWeatherStation(WS):
    date = "2022-01-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/ws/palaiseau-ws.asc"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 28


class TestDateArgument(WS):
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


class TestTimestampScreening(WS):
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
    with pytest.raises(ValueError):
        weather_station.ws2nc(filename, temp_path, SITE_META)


def test_invalid_header():
    filename = f"{SCRIPT_PATH}/data/ws/bad-ws.asc"
    temp_path = "test.nc"
    with pytest.raises(ValueError):
        weather_station.ws2nc(filename, temp_path, SITE_META)


def test_invalid_header2():
    filename = f"{SCRIPT_PATH}/data/ws/bad-ws2.asc"
    temp_path = "test.nc"
    with pytest.raises(ValueError):
        weather_station.ws2nc(filename, temp_path, SITE_META)


class TestWeatherStationGranada(WS):
    date = "2024-04-19"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Granada"}
    filename = f"{SCRIPT_PATH}/data/ws/granada.dat"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 10


class TestWeatherStationKenttarova(WS):
    date = "2024-05-20"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Kenttärova"}
    filename = [
        f"{SCRIPT_PATH}/data/ws/Krova_aws_pqBARLog5_20240520.csv",
        f"{SCRIPT_PATH}/data/ws/Krova_aws_pqBARLog5_20240521.csv",
    ]
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 24 * (60 / 10) + 1


class TestWeatherStationHyytiala(WS):
    date = "2024-01-10"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Hyytiälä"}
    filename = f"{SCRIPT_PATH}/data/ws/hyy20240110swx.txt"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 24 * 60


class TestWeatherStationGalati(WS):
    date = "2025-08-05"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Galați"}
    filename = f"{SCRIPT_PATH}/data/ws/galati.csv"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 5


class TestWeatherStationBucharest(WS):
    date = "2024-06-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Bucharest"}
    filename = f"{SCRIPT_PATH}/data/ws/bucharest.csv"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 19


class TestWeatherStationBucharestII(WS):
    date = "2024-07-14"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Bucharest"}
    filename = f"{SCRIPT_PATH}/data/ws/bucharest2.csv"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 48


class TestWeatherStationJuelich(WS):
    date = "2025-01-27"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Jülich"}
    filename = f"{SCRIPT_PATH}/data/ws/20250127_JOYCE_WST_01m.dat"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 9


class TestWeatherStationLampedusa(WS):
    date = "2025-01-19"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Lampedusa"}
    filename = f"{SCRIPT_PATH}/data/ws/lampedusa.rep"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 5


class TestWeatherStationLimassol(WS):
    date = "2024-04-27"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Limassol"}
    filename = f"{SCRIPT_PATH}/data/ws/WeatherStation_20240427.csv"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 5


class TestWeatherStationLimassol2(WS):
    date = "2025-04-13"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "Limassol"}
    filename = f"{SCRIPT_PATH}/data/ws/WeatherStation_20250413.csv"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 5


class TestWeatherStationLAquila(WS):
    date = "2025-06-30"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = {**SITE_META, "name": "L'Aquila"}
    filename = f"{SCRIPT_PATH}/data/ws/2025063009_weather-station.csv"
    uuid = weather_station.ws2nc(filename, temp_path, site_meta, date=date)

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 3


@pytest.mark.parametrize(
    "original, expected",
    [
        ([0, 1, 2], [0, 1, 2]),
        ([1, 2, 3], [0, 1, 2]),
        ([10, 11, 15, 16, 2], [0, 1, 5, 6, 8]),
        ([10, 11, 15, 16, 0, 2], [0, 1, 5, 6, 6, 8]),
    ],
)
def test_normalize_rainfall_amount(original, expected):
    filename = [f"{SCRIPT_PATH}/data/ws/bucharest2.csv"]
    site_meta = {**SITE_META, "name": "Bucharest"}
    a = weather_station.BucharestWS(filename, site_meta)
    original = np.array(original)
    expected = np.array(expected)
    a.data["rainfall_amount"] = CloudnetArray(original, "rainfall_amount")
    a.normalize_cumulative_amount("rainfall_amount")
    assert np.array_equal(a.data["rainfall_amount"].data, expected)
