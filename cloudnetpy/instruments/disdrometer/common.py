"""Module for reading / converting disdrometer data."""

from typing import Literal

import numpy as np

from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.metadata import MetaData


class Disdrometer(CloudnetInstrument):
    def add_meta(self) -> None:
        valid_keys = ("latitude", "longitude", "altitude")
        for key, value in self.site_meta.items():
            name = key.lower()
            if name in valid_keys:
                self.data[name] = CloudnetArray(float(value), name)

    def _convert_data(
        self,
        keys: tuple[str, ...],
        value: float,
        method: Literal["divide", "add"] = "divide",
    ) -> None:
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

    def store_vectors(
        self,
        n_values: list,
        spreads: list,
        name: str,
        start: float = 0.0,
    ):
        mid, bounds, spread = self._create_vectors(n_values, spreads, start)
        self.data[name] = CloudnetArray(mid, name, dimensions=(name,))
        key = f"{name}_spread"
        self.data[key] = CloudnetArray(spread, key, dimensions=(name,))
        key = f"{name}_bnds"
        self.data[key] = CloudnetArray(bounds, key, dimensions=(name, "nv"))

    def _create_vectors(
        self,
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
        long_name="Meteorological optical range (MOR) visibility",
        units="m",
        standard_name="visibility_in_air",
        comment="Visibility estimation by the disdrometer is valid\n"
        "only during precipitation events.",
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
