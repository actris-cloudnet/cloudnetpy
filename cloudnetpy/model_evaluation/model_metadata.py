from typing import NamedTuple, Optional


class ModelMetaData(NamedTuple):
    long_name: Optional[str] = None
    cycle_var: Optional[str] = None
    common_var: Optional[str] = None
    cycle: Optional[str] = None
    level: Optional[int] = None
    model_name: Optional[str] = None


MODELS = {
    "ecmwf": ModelMetaData(
        model_name="ECMWF", long_name="European Centre for Medium-Range Weather Forecasts", level=88
    ),
    "icon": ModelMetaData(
        model_name="ICON-Iglo",
        long_name="Icosahedral Nonhydrostatic Model",
        level=62,
        cycle="12-23, 24-35, 36-47",
    ),
    "era5": ModelMetaData(
        model_name="ERA5", long_name="Earth Re-Analysis System", level=88, cycle="1-12, 7-18"
    ),
    "harmonie": ModelMetaData(
        model_name="HARMONIE-AROME",
        long_name="the HIRLAM–ALADIN Research on Mesoscale Operational NWP in Euromed",
        level=65,
        cycle="6-11",
    ),
}

VARIABLES = {
    "variables": ModelMetaData(
        common_var="time, level, latitude, longitude, horizontal_resolution",
        cycle_var="forecast_time, height",
    ),
    "T": ModelMetaData(long_name="temperature"),
    "p": ModelMetaData(long_name="pressure"),
    "h": ModelMetaData(long_name="height"),
    "iwc": ModelMetaData(long_name="qi"),
    "lwc": ModelMetaData(long_name="ql"),
    "cf": ModelMetaData(long_name="cloud_fraction"),
}
