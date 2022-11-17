from typing import NamedTuple, Optional


class MetaData(NamedTuple):
    long_name: str
    units: str
    comment: Optional[str] = None
    standard_name: Optional[str] = None
    axis: Optional[str] = None
    positive: Optional[str] = None


MODEL_ATTRIBUTES = {
    "time": MetaData(
        units="decimal hours since midnight",
        long_name="Time UTC",
        axis="T",
    ),
    "latitude": MetaData(long_name="Latitude of grid point", units="dergees_north"),
    "longitude": MetaData(long_name="Longitude of grid point", units="degrees_east"),
    "horizontal_resolution": MetaData(
        long_name="Horizontal resolution of model",
        units="km",
        comment="Distance between two grid point",
    ),
    "level": MetaData(
        long_name="Model level",
        units="1",
        comment="Level 1 describes the highest height from ground.",
        axis="Z",
        positive="down",
    ),
}

CYCLE_ATTRIBUTES = {
    "forecast_time": MetaData(
        long_name="Time since initialization of forecast",
        units="hours",
        comment="The time elapsed since the initialization time of the forecast from which it\n"
        "was taken."
        "Note that the profiles in this file may be taken from more than one forecast.",
    ),
    "height": MetaData(
        long_name="Height above ground",
        units="m",
        comment="Height have been calculated using pressure, temperature and specific humidity.",
        positive="up",
    ),
    "pressure": MetaData(long_name="Pressure", units="Pa"),
    "temperature": MetaData(long_name="Temperature", units="K"),
    "uwind": MetaData(long_name="Zonal wind", units="m s-1", standard_name="eastward_wind"),
    "vwind": MetaData(long_name="Meridional wind", units="m s-1", standard_name="northward_wind"),
    "wwind": MetaData(
        long_name="Vertical wind",
        units="m s-1",
        standard_name="upward_wind",
        comment="The vertical wind has been calculated from omega (Pa s-1),\n"
        " height and pressure using: w=omega*dz/dp",
    ),
    "omega": MetaData(
        long_name="Vertical wind in pressure coordinates", units="PA s-1", standard_name="omega"
    ),
    "q": MetaData(long_name="Specific humidity", units="1"),
    "rh": MetaData(
        long_name="Relative humidity",
        units="1",
        comment="With respect to liquid above 0 degrees C and with respect to ice below \n"
        "0 degrees C",
    ),
}

MODEL_L3_ATTRIBUTES = {
    "cf": MetaData(long_name="Cloud fraction of model grid point", units="1"),
    "cf_cirrus": MetaData(
        long_name="Cloud fraction of model grid point with filtered cirrus fraction",
        units="1",
        comment="High level cirrus fraction is reduce do to lack if a radar capability to observe "
        "correctly particles small and high.",
    ),
    "iwc": MetaData(
        long_name="Ice water content of model grid point",
        units="kg m-3",
        comment="Calculated using model ice water mixing ration, pressure and temperature: \n"
        "qi*P/287*T",
    ),
    "lwc": MetaData(
        long_name="Liquid water content of model grid point",
        units="kg m-3",
        comment="Calculated using model liquid water mixing ration, pressure and temperature: \n"
        "ql*P/287*T",
    ),
}

REGRID_PRODUCT_ATTRIBUTES = {
    "cf_V": MetaData(
        long_name="Observed cloud fraction by volume",
        units="1",
        comment="Cloud fraction generated from observations and by volume, "
        "averaged onto the models grid with height and time. Volume is "
        "space withing four grid points",
    ),
    "cf_A": MetaData(
        long_name="Observed cloud fraction by area",
        units="1",
        comment="Cloud fraction generated from observation  and by area, "
        "averaged onto the models grid with height and time. Area is "
        "sum of time columns with any cloud fraction.",
    ),
    "cf_V_adv": MetaData(
        long_name="Observed cloud fraction by advection volume",
        units="1",
        comment="This variable is the same as the observed cloud fraction by volume, cf_V "
        "except that model winds were used to estimate the time taken to advect "
        "airflow a distance equivalent to the models horizontal resolution.",
    ),
    "cf_A_adv": MetaData(
        long_name="Observed cloud fraction by advection area",
        units="1",
        comment="This variable is the same as the observed cloud fraction by area, cf_A "
        "except that model winds were used to estimate the time taken to advect "
        "airflow a distance equivalent to the models horizontal resolution.",
    ),
    "iwc": MetaData(
        long_name="Observed ice water content reshaped to model dimensions by averaging",
        units="kg m-3",
        comment="This variable is the observed mean ice water content derived from radar \n"
        "reflectivity factor averaged onto the model grid with height and time. The formula \n"
        "has been applied where the categorization data has diagnosed \n"
        "that the radar echo is due to ice.",
    ),
    "iwc_att": MetaData(
        long_name="Observed ice water content with attenuation reshaped to model grid by averaging",
        units="kg m-3",
        comment="This variable is the same as the observed mean ice water content, iwc, except\n"
        "that profiles with uncorrected attenuation of the radar reflectivity were \n"
        "included.",
    ),
    "iwc_rain": MetaData(
        long_name="Observed ice water content with raining reshaped to model grid by averaging",
        units="kg m-3",
        comment="This variable is the same as the observed mean ice water content \n"
        "including attenuation, iwc_att, "
        "except that profiles with rain at the surface were also included.",
    ),
    "iwc_adv": MetaData(
        long_name="Observed ice water content reshaped to model advection grid by averaging",
        units="kg m-3",
        comment="This variable is the same as the observed mean ice water content, iwc, "
        "except that model winds were used to estimate the time taken to advect "
        "the flow a distance equivalent to the models horizontal resolution.",
    ),
    "iwc_att_adv": MetaData(
        long_name="Observed ice water content with attenuation reshaped to model \n"
        "advection grid by averaging",
        units="kg m-3",
        comment="This variable is the same as the observed mean ice water content, iwc_adv, \n"
        "except that profiles with "
        "uncorrected attenuation of the radar reflectivity were included.",
    ),
    "iwc_rain_adv": MetaData(
        long_name="Observed ice water content with raining reshaped to model advection\n"
        " grid by averaging",
        units="kg m-3",
        comment="This variable is the same as the observed mean ice water content including\n"
        " attenuation, iwc_att_adv, \n"
        "except that profiles with rain at the surface were also included.",
    ),
    "lwc": MetaData(
        long_name="Observed liquid water content reshaped to model grid by averaging",
        units="kg m-3",
        comment="This variable is the observed mean liquid water content estimated for\n"
        "pixels where the categorization \n"
        "data has diagnosed that liquid water is present, averaged onto the model grid\n"
        " with height and time.",
    ),
    "lwc_adv": MetaData(
        long_name="Observed liquid water content reshaped to model advection grid by averaging",
        units="kg m-3",
        comment="This variable is the same as the observed mean liquid water content, lwc, "
        "except that model winds were used to estimate the time taken to advect flow over\n"
        " the models grid points.",
    ),
}
