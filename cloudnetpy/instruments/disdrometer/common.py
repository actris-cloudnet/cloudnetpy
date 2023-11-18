"""Module for reading / converting disdrometer data."""
import logging

import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import MM_TO_M, SEC_IN_HOUR, SEC_IN_MINUTE
from cloudnetpy.exceptions import DisdrometerDataError, ValidTimeStampError
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.vaisala import values_to_dict
from cloudnetpy.metadata import MetaData

PARSIVEL = "OTT Parsivel-2"
THIES = "Thies-LNM"


class Disdrometer(CloudnetInstrument):
    def __init__(self, filename: str, site_meta: dict, source: str):
        super().__init__()
        self.filename = filename
        self.site_meta = site_meta
        self.source = source
        self.date: list[str] = []
        self.sensor_id = None
        self.n_diameter: int = 0
        self.n_velocity: int = 0
        self._file_data = self._read_file()

    def convert_units(self) -> None:
        mmh_to_ms = SEC_IN_HOUR / MM_TO_M
        c_to_k = 273.15
        self._convert_data(("rainfall_rate_1min_total",), mmh_to_ms)
        self._convert_data(("rainfall_rate",), mmh_to_ms)
        self._convert_data(("rainfall_rate_1min_solid",), mmh_to_ms)
        self._convert_data(("diameter", "diameter_spread", "diameter_bnds"), 1e3)
        self._convert_data(("V_sensor_supply",), 10)
        self._convert_data(("I_mean_laser",), 100)
        self._convert_data(("T_sensor",), c_to_k, method="add")
        self._convert_data(("T_interior",), c_to_k, method="add")
        self._convert_data(("T_ambient",), c_to_k, method="add")
        self._convert_data(("T_laser_driver",), c_to_k, method="add")

    def add_meta(self) -> None:
        valid_names = ("latitude", "longitude", "altitude")
        for key, value in self.site_meta.items():
            name = key.lower()
            if name in valid_names:
                self.data[name] = CloudnetArray(float(value), name)

    def validate_date(self, expected_date: str) -> None:
        valid_ind = []
        for ind, row in enumerate(self._file_data["scalars"]):
            if self.source == PARSIVEL:
                raise NotImplementedError
            date = _format_thies_date(row[3])
            if date == expected_date:
                valid_ind.append(ind)
        if not valid_ind:
            raise ValidTimeStampError
        for key, value in self._file_data.items():
            if value:
                self._file_data[key] = [self._file_data[key][ind] for ind in valid_ind]
        self.date = expected_date.split("-")

    def sort_time(self) -> None:
        time = self.data["time"][:]
        ind = time.argsort()
        for _, data in self.data.items():
            if data.data.shape[0] == len(time):
                data.data[:] = data.data[ind]

    def _read_file(self) -> dict:
        data: dict = {"scalars": [], "vectors": [], "spectra": []}
        with open(self.filename, encoding="utf8", errors="ignore") as file:
            for row in file:
                if row == "\n":
                    continue
                if self.source == PARSIVEL:
                    values = row.split(";")
                    if "\n" in values:
                        values.remove("\n")
                    if len(values) != 1106:
                        continue
                    data["scalars"].append(values[:18])
                    data["vectors"].append(values[18 : 18 + 64])
                    data["spectra"].append(values[18 + 64 :])
                else:
                    values = row.split(";")
                    data["scalars"].append(values[:79])
                    data["spectra"].append(values[79:-2])
        if len(data["scalars"]) == 0:
            raise ValueError
        return data

    def _append_data(self, column_and_key: list) -> None:
        indices, keys = zip(*column_and_key, strict=True)
        data = self._parse_useful_data(indices)
        data_dict = values_to_dict(keys, data)
        for key in keys:
            if key.startswith("_"):
                continue
            invalid_value = -9999.0
            float_array = ma.array([])
            for value_str in data_dict[key]:
                try:
                    float_array = ma.append(float_array, float(value_str))
                except ValueError:
                    logging.warning(
                        "Invalid character: %s, masking a data point",
                        value_str,
                    )
                    float_array = ma.append(float_array, invalid_value)
            float_array[float_array == invalid_value] = ma.masked
            if key in (
                "rainfall_rate",
                "radar_reflectivity",
                "T_sensor",
                "I_heating",
                "V_power_supply",
                "T_interior",
                "T_ambient",
                "T_laser_driver",
            ):
                data_type = "f4"
            else:
                data_type = "i4"
            self.data[key] = CloudnetArray(float_array, key, data_type=data_type)
        self.data["time"] = self._convert_time(data_dict)
        if "_serial_number" in data_dict:
            first_id = data_dict["_serial_number"][0]
            for sensor_id in data_dict["_serial_number"]:
                if sensor_id != first_id:
                    msg = "Multiple serial numbers are not supported"
                    raise DisdrometerDataError(msg)

            self.serial_number = first_id

    def _parse_useful_data(self, indices: tuple) -> list:
        data = []
        for row in self._file_data["scalars"]:
            useful_data = [row[ind] for ind in indices]
            data.append(useful_data)
        return data

    def _convert_time(self, data: dict) -> CloudnetArray:
        seconds = []
        for timestamp in data["_time"]:
            if self.source == PARSIVEL:
                raise NotImplementedError
            hour, minute, sec = timestamp.split(":")
            seconds.append(
                int(hour) * SEC_IN_HOUR + int(minute) * SEC_IN_MINUTE + int(sec)
            )
        return CloudnetArray(utils.seconds2hours(np.array(seconds)), "time")

    def _convert_data(self, keys: tuple, value: float, method: str = "divide") -> None:
        for key in keys:
            if key in self.data:
                if method == "divide":
                    self.data[key].data /= value
                elif method == "add":
                    self.data[key].data += value
                else:
                    raise ValueError

    def _append_spectra(self) -> None:
        array = ma.masked_all(
            (len(self._file_data["scalars"]), self.n_diameter, self.n_velocity),
        )
        for time_ind, row in enumerate(self._file_data["spectra"]):
            values = _parse_int(row)
            array[time_ind, :, :] = np.reshape(
                values,
                (self.n_diameter, self.n_velocity),
            )
        self.data["data_raw"] = CloudnetArray(
            array,
            "data_raw",
            dimensions=("time", "diameter", "velocity"),
            data_type="i2",
        )

    @classmethod
    def store_vectors(
        cls,
        data,
        n_values: list,
        spreads: list,
        name: str,
        start: float = 0.0,
    ):
        mid, bounds, spread = cls._create_vectors(n_values, spreads, start)
        data[name] = CloudnetArray(mid, name, dimensions=(name,))
        key = f"{name}_spread"
        data[key] = CloudnetArray(spread, key, dimensions=(name,))
        key = f"{name}_bnds"
        data[key] = CloudnetArray(bounds, key, dimensions=(name, "nv"))

    @staticmethod
    def _create_vectors(
        n_values: list[int],
        spreads: list[float],
        start: float,
    ) -> tuple:
        mid_value: np.ndarray = np.array([])
        lower_limit: np.ndarray = np.array([])
        upper_limit: np.ndarray = np.array([])
        for spread, n in zip(spreads, n_values, strict=True):
            lower = np.linspace(start, start + (n - 1) * spread, n)
            upper = lower + spread
            lower_limit = np.append(lower_limit, lower)
            upper_limit = np.append(upper_limit, upper)
            mid_value = np.append(mid_value, (lower + upper) / 2)
            start = upper[-1]
        bounds = np.stack((lower_limit, upper_limit)).T
        spread = bounds[:, 1] - bounds[:, 0]
        return mid_value, bounds, spread


def _format_thies_date(date: str) -> str:
    day, month, year = date.split(".")
    year = f"20{year}"
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def _parse_int(row: np.ndarray) -> np.ndarray:
    values = ma.masked_all((len(row),))
    for ind, value in enumerate(row):
        try:
            value_int = int(value)
            if value_int != 0:
                values[ind] = value_int
        except ValueError:
            pass
    return values


ATTRIBUTES = {
    "velocity": MetaData(
        long_name="Center fall velocity of precipitation particles",
        units="m s-1",
        comment="Predefined velocity classes.",
    ),
    "velocity_spread": MetaData(
        long_name="Width of velocity interval",
        units="m s-1",
        comment="Bin size of each velocity interval.",
    ),
    "velocity_bnds": MetaData(
        long_name="Velocity bounds",
        units="m s-1",
        comment="Upper and lower bounds of velocity interval.",
    ),
    "diameter": MetaData(
        long_name="Center diameter of precipitation particles",
        units="m",
        comment="Predefined diameter classes.",
    ),
    "diameter_spread": MetaData(
        long_name="Width of diameter interval",
        units="m",
        comment="Bin size of each diameter interval.",
    ),
    "diameter_bnds": MetaData(
        long_name="Diameter bounds",
        units="m",
        comment="Upper and lower bounds of diameter interval.",
    ),
    "rainfall_rate_1min_total": MetaData(
        long_name="Total precipitation rate",
        units="m s-1",
    ),
    "rainfall_rate": MetaData(
        long_name="Rainfall rate",
        units="m s-1",
        standard_name="rainfall_rate",
    ),
    "rainfall_rate_1min_solid": MetaData(
        long_name="Solid precipitation rate",
        units="m s-1",
    ),
    "snowfall_rate": MetaData(
        long_name="Snowfall rate",
        units="m s-1",
        comment="Snow depth intensity (volume equivalent)",
    ),
    "synop_WaWa": MetaData(long_name="Synop code WaWa", units="1"),
    "synop_WW": MetaData(long_name="Synop code WW", units="1"),
    "radar_reflectivity": MetaData(
        long_name="Equivalent radar reflectivity factor",
        units="dBZ",
        standard_name="equivalent_reflectivity_factor",
    ),
    "visibility": MetaData(
        long_name="Visibility range in precipitation after MOR",
        units="m",
        standard_name="visibility_in_air",
    ),
    "interval": MetaData(long_name="Length of measurement interval", units="s"),
    "sig_laser": MetaData(long_name="Signal amplitude of the laser strip", units="1"),
    "n_particles": MetaData(
        long_name="Number of particles in time interval",
        units="1",
    ),
    "T_sensor": MetaData(
        long_name="Temperature in the sensor housing",
        units="K",
    ),
    "I_heating": MetaData(
        long_name="Heating current",
        units="A",
    ),
    "V_sensor_supply": MetaData(
        long_name="Sensor supply voltage",
        units="V",
    ),
    "V_power_supply": MetaData(
        long_name="Power supply voltage",
        units="V",
    ),
    "state_sensor": MetaData(
        long_name="State of the sensor",
        comment="0 = OK, 1 = Dirty, 2 = No measurement possible.",
        units="1",
    ),
    "error_code": MetaData(long_name="Error code", units="1"),
    "number_concentration": MetaData(
        long_name="Number of particles per diameter class",
        comment="Unit is actually logarithmic log10(m-3 mm-1).",
        units="m-3 mm-1",
    ),
    "fall_velocity": MetaData(
        long_name="Average velocity of each diameter class",
        units="m s-1",
    ),
    "data_raw": MetaData(
        long_name="Raw data as a function of particle diameter and velocity",
        units="1",
    ),
    "kinetic_energy": MetaData(
        long_name="Kinetic energy of the hydrometeors",
        units="J m-2 h-1",
    ),
    # Thies-specific:
    "T_ambient": MetaData(long_name="Ambient temperature", units="K"),
    "T_interior": MetaData(long_name="Interior temperature", units="K"),
    "status_T_laser_analogue": MetaData(
        long_name="Status of laser temperature (analogue)",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_T_laser_digital": MetaData(
        long_name="Status of laser temperature (digital)",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_I_laser_analogue": MetaData(
        long_name="Status of laser current (analogue)",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_I_laser_digital": MetaData(
        long_name="Status of laser current (digital)",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_sensor_supply": MetaData(
        long_name="Status of sensor supply",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_laser_heating": MetaData(
        long_name="Status of laser heating",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_receiver_heating": MetaData(
        long_name="Status of receiver heating",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_temperature_sensor": MetaData(
        long_name="Status of temperature sensor",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_heating_supply": MetaData(
        long_name="Status of heating supply",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_heating_housing": MetaData(
        long_name="Status of heating housing",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_heating_heads": MetaData(
        long_name="Status of heating heads",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_heating_carriers": MetaData(
        long_name="Status of heating carriers",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_laser_power": MetaData(
        long_name="Status of laser power",
        comment="0 = OK , 1 = Error",
        units="1",
    ),
    "status_laser": MetaData(
        long_name="Status of laser",
        comment="0 = OK/on , 1 = Off",
        units="1",
    ),
    "measurement_quality": MetaData(long_name="Measurement quality", units="%"),
    "maximum_hail_diameter": MetaData(long_name="Maximum hail diameter", units="mm"),
    "static_signal": MetaData(
        long_name="Static signal",
        comment="0 = OK, 1 = ERROR",
        units="1",
    ),
    "T_laser_driver": MetaData(long_name="Temperature of laser driver", units="K"),
    "I_mean_laser": MetaData(long_name="Mean value of laser current", units="mA"),
    "V_control": MetaData(
        long_name="Control voltage",
        units="mV",
        comment="Reference value: 4010+-5",
    ),
    "V_optical_output": MetaData(
        long_name="Voltage of optical control output",
        units="mV",
    ),
    "I_heating_laser_head": MetaData(
        long_name="Laser head heating current",
        units="mA",
    ),
    "I_heating_receiver_head": MetaData(
        long_name="Receiver head heating current",
        units="mA",
    ),
}
