from os import PathLike
from uuid import UUID

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize.disdrometer import DataSource
from cloudnetpy.categorize.model import Model
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import T0
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData


def generate_iwc_from_weather_radar(
    weather_radar_file: str | PathLike,
    model_file: str | PathLike,
    output_file: str | PathLike,
    uuid: str | UUID | None = None,
) -> UUID:
    uuid = utils.get_uuid(uuid)
    radar = WrRadar(weather_radar_file)
    if radar.altitude is None:
        msg = "Altitude is missing"
        raise ValueError(msg)
    if radar.height is None or radar.height_agl is None:
        msg = "Height is missing"
        raise ValueError(msg)
    model = Model(model_file, radar.altitude)
    model.interpolate_to_common_height()
    model.interpolate_to_grid(radar.time, radar.height)

    height_agl = radar.height_agl
    z = radar.data["Z"][:]
    v = radar.data["v"][:]
    rho_hv = radar.data["rho_hv"][:]
    temp = model.data_dense["temperature"]

    is_noise = (height_agl < 3000) & (rho_hv < 0.8) & (z < -20)
    is_ice = (z > -30) & (np.abs(v) < 2) & (temp < T0)
    is_rain = (z > 10) & (v < 0) & (temp > T0 - 10)

    output_data = {
        "time": CloudnetArray(radar.time, "time"),
        "height": CloudnetArray(radar.height, "height"),
        "radar_frequency": CloudnetArray(radar.radar_frequency, "radar_frequency"),
    }

    z_rain = ma.masked_where(is_noise | ~is_rain, z)
    rainfall_rate = (utils.db2lin(z_rain) / 200) ** 0.625  # Marshall-Palmer
    output_data["rainfall_rate"] = CloudnetArray(
        rainfall_rate * 1e-3 / 3600, "rainfall_rate"
    )

    z_ice = ma.masked_where(is_noise | is_rain | ~is_ice, z)
    iwc = ma.power(10, 0.060 * z_ice - 0.0197 * temp - 1.70)  # Hogan et al. (2006)
    output_data["iwc"] = CloudnetArray(1000 * iwc, "iwc")

    date = radar.get_date()
    attributes = output.add_time_attribute(WEATHER_RADAR_RET_ATTRIBUTES, date)
    output.update_attributes(output_data, attributes)

    dimensions = {"time": len(radar.time), "height": len(radar.height)}
    with output.init_file(output_file, dimensions, output_data, uuid) as nc:
        nc.cloudnet_file_type = "iwc-weather-radar"
        vars_from_source = (
            "altitude",
            "latitude",
            "longitude",
        )
        output.copy_variables(radar.dataset, nc, vars_from_source)
        output.copy_global(radar.dataset, nc, ("year", "month", "day", "location"))
        nc.title = f"Ice water content (weather radar) retrieval from {radar.location}"
        nc.source = "\n".join(sorted([radar.source, model.source]))
        nc.source_file_uuids = output.get_source_uuids([radar, model])
        output.merge_history(nc, nc.cloudnet_file_type, radar, (model,))

    return uuid


class WrRadar(DataSource):
    def __init__(self, full_path: str | PathLike) -> None:
        super().__init__(full_path, radar=True)
        self.radar_frequency = float(self.getvar("radar_frequency"))
        self.location = getattr(self.dataset, "location", "")
        self._init_data()

    def _init_data(self) -> None:
        self.append_data(self.getvar("Zh"), "Z", units="dBZ")
        self.append_data(self.getvar("v"), "v")
        self.append_data(self.getvar("rho_hv"), "rho_hv")


WEATHER_RADAR_RET_ATTRIBUTES = {
    "height": COMMON_ATTRIBUTES["height"]._replace(dimensions=("height",)),
    "iwc": MetaData(
        long_name="Ice water content",
        units="kg m-3",
        dimensions=("time", "height"),
    ),
    "rainfall_rate": MetaData(
        long_name="Rainfall rate",
        units="m s-1",
        dimensions=("time", "height"),
    ),
}
