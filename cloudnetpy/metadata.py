"""Initial Metadata of Cloudnet variables for NetCDF file writing."""

from typing import NamedTuple


class MetaData(NamedTuple):
    long_name: str | None = None
    standard_name: str | None = None
    units: str | None = None
    comment: str | None = None
    definition: str | None = None
    references: str | None = None
    ancillary_variables: str | None = None
    positive: str | None = None
    axis: str | None = None
    calendar: str | None = None
    source: str | None = None


COMMON_ATTRIBUTES = {
    "time": MetaData(
        long_name="Time UTC",
        axis="T",
        standard_name="time",
        calendar="standard",
    ),
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "range": MetaData(
        long_name="Range from instrument",
        axis="Z",
        units="m",
        comment="Distance from instrument to centre of each range bin.",
    ),
    "latitude": MetaData(
        long_name="Latitude of site",
        units="degree_north",
        standard_name="latitude",
    ),
    "longitude": MetaData(
        long_name="Longitude of site",
        units="degree_east",
        standard_name="longitude",
    ),
    "altitude": MetaData(
        long_name="Altitude of site",
        standard_name="altitude",
        units="m",
    ),
    "Zh": MetaData(
        long_name="Radar reflectivity factor",
        units="dBZ",
        comment="Calibrated reflectivity. Calibration convention: in the absence\n"
        "of attenuation, a cloud at 273 K containing one million 100-micron droplets\n"
        "per cubic metre will have a reflectivity of 0 dBZ at all frequencies.",
    ),
    "width": MetaData(
        long_name="Spectral width",
        units="m s-1",
        comment=(
            "This parameter is the standard deviation of the reflectivity-weighted\n"
            "velocities in the radar pulse volume."
        ),
    ),
    "v": MetaData(
        long_name="Doppler velocity",
        units="m s-1",
        comment=(
            "This parameter is the radial component of the velocity, with positive\n"
            "velocities are away from the radar."
        ),
    ),
    "ldr": MetaData(
        long_name="Linear depolarisation ratio",
        units="dB",
        comment="This parameter is the ratio of cross-polar to co-polar reflectivity.",
    ),
    "sldr": MetaData(long_name="Slanted linear depolarisation ratio", units="dB"),
    "lwp": MetaData(
        long_name="Liquid water path",
        units="kg m-2",
        standard_name="atmosphere_cloud_liquid_water_content",
    ),
    "iwv": MetaData(
        long_name="Integrated water vapour",
        units="kg m-2",
        standard_name="atmosphere_mass_content_of_water_vapor",
    ),
    "kurtosis": MetaData(
        long_name="Kurtosis of spectra",
        units="1",
    ),
    "nyquist_velocity": MetaData(long_name="Nyquist velocity", units="m s-1"),
    "radar_frequency": MetaData(long_name="Radar transmit frequency", units="GHz"),
    "beta": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient.",
    ),
    "beta_raw": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="Non-screened attenuated backscatter coefficient.",
    ),
    "beta_smooth": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient.\n"
        "Weak background smoothed using Gaussian 2D-kernel.",
    ),
    "wavelength": MetaData(
        long_name="Laser wavelength",
        units="nm",
    ),
    "zenith_angle": MetaData(
        long_name="Zenith angle",
        units="degree",
        standard_name="zenith_angle",
        comment="Angle to the local vertical. A value of zero is directly overhead.",
    ),
    "azimuth_angle": MetaData(
        long_name="Azimuth angle",
        standard_name="sensor_azimuth_angle",
        units="degree",
        comment="Angle between North and the line of sight, measured clockwise.",
    ),
    "temperature": MetaData(
        long_name="Temperature",
        units="K",
    ),
    "pressure": MetaData(
        long_name="Pressure",
        units="Pa",
    ),
    "SNR": MetaData(
        long_name="Signal-to-noise ratio",
        units="dB",
    ),
    "relative_humidity": MetaData(
        long_name="Relative humidity",
        standard_name="relative_humidity",
        units="1",
    ),
    "absolute_humidity": MetaData(
        long_name="Absolute humidity",
        standard_name="mass_concentration_of_water_vapor_in_air",
        units="kg m-3",
    ),
    "wind_speed": MetaData(
        long_name="Wind speed",
        standard_name="wind_speed",
        units="m s-1",
    ),
    "wind_direction": MetaData(
        long_name="Wind direction",
        standard_name="wind_from_direction",
        units="degree",
    ),
    "rainfall_rate": MetaData(
        long_name="Rainfall rate",
        standard_name="rainfall_rate",
        units="m s-1",
    ),
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
}
