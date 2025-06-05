from cloudnetpy.metadata import MetaData

MODEL_ATTRIBUTES = {
    "time": MetaData(
        units="",
        long_name="Time UTC",
        axis="T",
        calendar="standard",
        dimensions=("time",),
    ),
    "latitude": MetaData(
        long_name="Latitude of grid point", units="degree_north", dimensions=None
    ),
    "longitude": MetaData(
        long_name="Longitude of grid point", units="degree_east", dimensions=None
    ),
    "horizontal_resolution": MetaData(
        long_name="Horizontal resolution of model",
        units="km",
        comment="Distance between two grid point",
        dimensions=None,
    ),
    "level": MetaData(
        long_name="Model level",
        units="1",
        comment="Level 1 describes the highest height from ground.",
        axis="Z",
        positive="down",
        dimensions=("level",),
    ),
}

CYCLE_ATTRIBUTES = {
    "forecast_time": MetaData(
        long_name="Time since initialization of forecast",
        units="hours",
        comment=(
            "The time elapsed since the initialization time of the forecast from\n"
            "which it was taken. Note that the profiles in this file may be taken\n"
            "from more than one forecast."
        ),
        dimensions=("time",),
    ),
    "height": MetaData(
        long_name="Height above ground",
        units="m",
        comment=(
            "Height have been calculated using pressure, temperature and\n"
            "specific humidity."
        ),
        positive="up",
        dimensions=("time", "level"),
    ),
    "pressure": MetaData(long_name="Pressure", units="Pa", dimensions=None),
    "temperature": MetaData(long_name="Temperature", units="K", dimensions=None),
    "uwind": MetaData(
        long_name="Zonal wind",
        units="m s-1",
        standard_name="eastward_wind",
        dimensions=None,
    ),
    "vwind": MetaData(
        long_name="Meridional wind",
        units="m s-1",
        standard_name="northward_wind",
        dimensions=None,
    ),
    "wwind": MetaData(
        long_name="Vertical wind",
        units="m s-1",
        standard_name="upward_wind",
        comment=(
            "The vertical wind has been calculated from omega (Pa s-1),\n"
            "height and pressure using: w=omega*dz/dp"
        ),
        dimensions=None,
    ),
    "omega": MetaData(
        long_name="Vertical wind in pressure coordinates",
        units="PA s-1",
        standard_name="omega",
        dimensions=None,
    ),
    "q": MetaData(long_name="Specific humidity", units="1", dimensions=None),
    "rh": MetaData(
        long_name="Relative humidity",
        units="1",
        comment=(
            "With respect to liquid above 0 degrees C and with respect to ice\n"
            "below 0 degrees C"
        ),
        dimensions=None,
    ),
}

MODEL_L3_ATTRIBUTES = {
    "cf": MetaData(
        long_name="Cloud fraction of model grid point",
        units="1",
        dimensions=("time", "level"),
    ),
    "cf_cirrus": MetaData(
        long_name="Cloud fraction of model grid point with filtered cirrus fraction",
        units="1",
        comment="High level cirrus fraction is reduce do to lack if a radar\n"
        "capability to observe correctly particles small and high.",
        dimensions=("time", "level"),
    ),
    "iwc": MetaData(
        long_name="Ice water content of model grid point",
        units="kg m-3",
        comment="Calculated using model ice water mixing ration, pressure\n"
        "and temperature: qi*P/287*T",
        dimensions=("time", "level"),
    ),
    "lwc": MetaData(
        long_name="Liquid water content of model grid point",
        units="kg m-3",
        comment="Calculated using model liquid water mixing ration, pressure\n"
        "and temperature:  ql*P/287*T",
        dimensions=("time", "level"),
    ),
}

REGRID_PRODUCT_ATTRIBUTES = {
    "cf_V": MetaData(
        long_name="Observed cloud fraction by volume",
        units="1",
        comment=(
            "Cloud fraction generated from observations and by volume,\n"
            "averaged onto the models grid with height and time. Volume is\n"
            "space within four grid points."
        ),
        dimensions=("time", "level"),
    ),
    "cf_A": MetaData(
        long_name="Observed cloud fraction by area",
        units="1",
        comment=(
            "Cloud fraction generated from observation  and by area,\n"
            "averaged onto the models grid with height and time. Area is\n"
            "sum of time columns with any cloud fraction."
        ),
        dimensions=("time", "level"),
    ),
    "cf_V_adv": MetaData(
        long_name="Observed cloud fraction by advection volume",
        units="1",
        comment=(
            "This variable is the same as the observed cloud fraction by volume, cf_V\n"
            "except that model winds were used to estimate the time taken to advect\n"
            "airflow a distance equivalent to the models horizontal resolution."
        ),
        dimensions=("time", "level"),
    ),
    "cf_A_adv": MetaData(
        long_name="Observed cloud fraction by advection area",
        units="1",
        comment=(
            "This variable is the same as the observed cloud fraction by area, cf_A\n"
            "except that model winds were used to estimate the time taken to advect\n"
            "airflow a distance equivalent to the models horizontal resolution."
        ),
        dimensions=("time", "level"),
    ),
    "iwc": MetaData(
        long_name="Observed ice water content reshaped to model dimensions by"
        " averaging",
        units="kg m-3",
        comment=(
            "This variable is the observed mean ice water content derived from radar\n"
            "reflectivity factor averaged onto the model grid with height and time.\n"
            "The formula  has been applied where the categorization data has\n"
            "diagnosed that the radar echo is due to ice."
        ),
        dimensions=("time", "level"),
    ),
    "iwc_att": MetaData(
        long_name="Observed ice water content with attenuation reshaped to model grid"
        " by averaging",
        units="kg m-3",
        comment=(
            "This variable is the same as the observed mean ice water content, iwc,\n"
            "except that profiles with uncorrected attenuation of the radar\n"
            "reflectivity were included."
        ),
        dimensions=("time", "level"),
    ),
    "iwc_rain": MetaData(
        long_name="Observed ice water content with raining reshaped to model grid"
        " by averaging",
        units="kg m-3",
        comment=(
            "This variable is the same as the observed mean ice water content\n"
            "including attenuation, iwc_att, except that profiles with rain\n"
            "at the surface were also included."
        ),
        dimensions=("time", "level"),
    ),
    "iwc_adv": MetaData(
        long_name="Observed ice water content reshaped to model advection grid"
        " by averaging",
        units="kg m-3",
        comment=(
            "This variable is the same as the observed mean ice water content, iwc,\n"
            "except that model winds were used to estimate the time taken to advect\n"
            "the flow a distance equivalent to the models horizontal resolution."
        ),
        dimensions=("time", "level"),
    ),
    "iwc_att_adv": MetaData(
        long_name="Observed ice water content with attenuation reshaped to model"
        " advection grid by averaging",
        units="kg m-3",
        comment=(
            "This variable is the same as the observed mean ice water content,\n"
            "iwc_adv,  except that profiles with uncorrected attenuation of the radar\n"
            "reflectivity were included."
        ),
        dimensions=("time", "level"),
    ),
    "iwc_rain_adv": MetaData(
        long_name="Observed ice water content with raining reshaped to model"
        " advection rid by averaging",
        units="kg m-3",
        comment=(
            "This variable is the same as the observed mean ice water content\n"
            "including attenuation, iwc_att_adv, except that profiles with rain\n"
            "at the surface were also included."
        ),
        dimensions=("time", "level"),
    ),
    "lwc": MetaData(
        long_name="Observed liquid water content reshaped to model grid by averaging",
        units="kg m-3",
        comment=(
            "This variable is the observed mean liquid water content estimated for\n"
            "pixels where the categorization data has diagnosed that liquid water \n"
            "is present, averaged onto the model grid with height and time."
        ),
        dimensions=("time", "level"),
    ),
    "lwc_adv": MetaData(
        long_name="Observed liquid water content reshaped to model advection grid by "
        "averaging",
        units="kg m-3",
        comment=(
            "This variable is the same as the observed mean liquid water content,\n"
            "lwc, except that model winds were used to estimate the time taken to\n"
            "advect flow over the models grid points."
        ),
        dimensions=("time", "level"),
    ),
}
