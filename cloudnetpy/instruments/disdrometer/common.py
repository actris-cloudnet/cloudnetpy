"""Module for reading / converting disdrometer data."""

from cloudnetpy.metadata import MetaData

ATTRIBUTES = {
    "velocity": MetaData(
        long_name="Center fall velocity of precipitation particles",
        units="m s-1",
        comment="Predefined velocity classes.",
        dimensions=("velocity",),
    ),
    "velocity_spread": MetaData(
        long_name="Width of velocity interval",
        units="m s-1",
        comment="Bin size of each velocity interval.",
        dimensions=("velocity",),
    ),
    "velocity_bnds": MetaData(
        long_name="Velocity bounds",
        units="m s-1",
        comment="Upper and lower bounds of velocity interval.",
        dimensions=("velocity", "nv"),
    ),
    "diameter": MetaData(
        long_name="Center diameter of precipitation particles",
        units="m",
        comment="Predefined diameter classes.",
        dimensions=("diameter",),
    ),
    "diameter_spread": MetaData(
        long_name="Width of diameter interval",
        units="m",
        comment="Bin size of each diameter interval.",
        dimensions=("diameter",),
    ),
    "diameter_bnds": MetaData(
        long_name="Diameter bounds",
        units="m",
        comment="Upper and lower bounds of diameter interval.",
        dimensions=("diameter", "nv"),
    ),
    "rainfall_rate": MetaData(
        long_name="Rainfall rate",
        units="m s-1",
        standard_name="rainfall_rate",
        dimensions=("time",),
    ),
    # "snowfall_rate": MetaData(
    #     long_name="Snowfall rate",
    #     units="m s-1",
    #     comment="Snow depth intensity (volume equivalent)",
    #     dimensions=("time",),
    # ),
    # "synop_WaWa": MetaData(
    #     long_name="Synop code WaWa", units="1", dimensions=("time",)
    # ),
    # "synop_WW": MetaData(long_name="Synop code WW", units="1", dimensions=("time",)),
    "radar_reflectivity": MetaData(
        long_name="Equivalent radar reflectivity factor",
        units="dBZ",
        standard_name="equivalent_reflectivity_factor",
        dimensions=("time",),
    ),
    # "visibility": MetaData(
    #     long_name="Meteorological optical range (MOR) visibility",
    #     units="m",
    #     standard_name="visibility_in_air",
    #     comment="Visibility estimation by the disdrometer is valid\n"
    #     "only during precipitation events.",
    #     dimensions=("time",),
    # ),
    "interval": MetaData(
        long_name="Length of measurement interval", units="s", dimensions=("time",)
    ),
    "n_particles": MetaData(
        long_name="Number of particles in time interval",
        units="1",
        dimensions=("time",),
    ),
    "number_concentration": MetaData(
        long_name="Number of particles per diameter class",
        units="m-3 mm-1",
        dimensions=("time", "diameter"),
    ),
    "fall_velocity": MetaData(
        long_name="Average velocity of each diameter class",
        units="m s-1",
        dimensions=("time", "diameter"),
    ),
    "data_raw": MetaData(
        long_name="Raw data as a function of particle diameter and velocity",
        units="1",
        dimensions=("time", "diameter", "velocity"),
    ),
    "kinetic_energy": MetaData(
        long_name="Kinetic energy of the hydrometeors",
        units="J m-2 h-1",
        dimensions=("time",),
    ),
    "sampling_area": MetaData(
        long_name="Instrument sampling area", units="m2", dimensions=None
    ),
}
