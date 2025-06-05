"""Initial Metadata of Cloudnet variables for NetCDF file writing."""

from typing import NamedTuple


class MetaData(NamedTuple):
    long_name: str
    dimensions: tuple[str, ...] | None
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
        dimensions=("time",),
    ),
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
        dimensions=("range",),
    ),
    "range": MetaData(
        long_name="Range from instrument",
        axis="Z",
        units="m",
        comment="Distance from instrument to centre of each range bin.",
        dimensions=("range",),
    ),
    "latitude": MetaData(
        long_name="Latitude of site",
        units="degree_north",
        standard_name="latitude",
        dimensions=("time",),
    ),
    "longitude": MetaData(
        long_name="Longitude of site",
        units="degree_east",
        standard_name="longitude",
        dimensions=("time",),
    ),
    "altitude": MetaData(
        long_name="Altitude of site",
        standard_name="altitude",
        units="m",
        dimensions=("time",),
    ),
    "Zh": MetaData(
        long_name="Radar reflectivity factor",
        units="dBZ",
        comment="Calibrated reflectivity. Calibration convention: in the absence\n"
        "of attenuation, a cloud at 273 K containing one million 100-micron droplets\n"
        "per cubic metre will have a reflectivity of 0 dBZ at all frequencies.",
        dimensions=("time", "range"),
    ),
    "width": MetaData(
        long_name="Spectral width",
        units="m s-1",
        comment=(
            "This parameter is the standard deviation of the reflectivity-weighted\n"
            "velocities in the radar pulse volume."
        ),
        dimensions=("time", "range"),
    ),
    "v": MetaData(
        long_name="Doppler velocity",
        units="m s-1",
        comment=(
            "This parameter is the radial component of the velocity, with positive\n"
            "velocities are away from the radar."
        ),
        dimensions=("time", "range"),
    ),
    "ldr": MetaData(
        long_name="Linear depolarisation ratio",
        units="dB",
        comment="This parameter is the ratio of cross-polar to co-polar reflectivity.",
        dimensions=("time", "range"),
    ),
    "sldr": MetaData(
        long_name="Slanted linear depolarisation ratio",
        units="dB",
        dimensions=("time", "range"),
    ),
    "lwp": MetaData(
        long_name="Liquid water path",
        units="kg m-2",
        standard_name="atmosphere_cloud_liquid_water_content",
        dimensions=("time",),
    ),
    "iwv": MetaData(
        long_name="Integrated water vapour",
        units="kg m-2",
        standard_name="atmosphere_mass_content_of_water_vapor",
        dimensions=("time",),
    ),
    "kurtosis": MetaData(
        long_name="Kurtosis of spectra",
        units="1",
        dimensions=("time", "range"),
    ),
    "skewness": MetaData(
        long_name="Skewness of spectra",
        units="1",
        dimensions=("time", "range"),
    ),
    "nyquist_velocity": MetaData(
        long_name="Nyquist velocity",
        units="m s-1",
        dimensions=("time", "range"),
    ),
    "radar_frequency": MetaData(
        long_name="Radar transmit frequency", units="GHz", dimensions=None
    ),
    "beta": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient.",
        dimensions=("time", "range"),
    ),
    "beta_raw": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="Non-screened attenuated backscatter coefficient.",
        dimensions=("time", "range"),
    ),
    "beta_smooth": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient.\n"
        "Weak background smoothed using Gaussian 2D-kernel.",
        dimensions=("time", "range"),
    ),
    "wavelength": MetaData(long_name="Laser wavelength", units="nm", dimensions=None),
    "zenith_angle": MetaData(
        long_name="Zenith angle",
        units="degree",
        standard_name="zenith_angle",
        comment="Angle to the local vertical. A value of zero is directly overhead.",
        dimensions=("time",),
    ),
    "ir_zenith_angle": MetaData(
        long_name="Infrared sensor zenith angle",
        units="degree",
        comment="90=horizon, 0=zenith",
        dimensions=("time",),
    ),
    "azimuth_angle": MetaData(
        long_name="Azimuth angle",
        standard_name="sensor_azimuth_angle",
        units="degree",
        comment="Angle between North and the line of sight, measured clockwise.",
        dimensions=("time",),
    ),
    "temperature": MetaData(long_name="Temperature", units="K", dimensions=("time",)),
    "pressure": MetaData(long_name="Pressure", units="Pa", dimensions=("time",)),
    "SNR": MetaData(
        long_name="Signal-to-noise ratio", units="dB", dimensions=("time", "range")
    ),
    "relative_humidity": MetaData(
        long_name="Relative humidity",
        standard_name="relative_humidity",
        units="1",
        dimensions=("time",),
    ),
    "absolute_humidity": MetaData(
        long_name="Absolute humidity",
        standard_name="mass_concentration_of_water_vapor_in_air",
        units="kg m-3",
        dimensions=("time", "range"),
    ),
    "wind_speed": MetaData(
        long_name="Wind speed",
        standard_name="wind_speed",
        units="m s-1",
        dimensions=("time",),
    ),
    "wind_direction": MetaData(
        long_name="Wind direction",
        standard_name="wind_from_direction",
        units="degree",
        dimensions=("time",),
    ),
    "rainfall_rate": MetaData(
        long_name="Rainfall rate",
        standard_name="rainfall_rate",
        units="m s-1",
        dimensions=("time",),
    ),
    "rainfall_amount": MetaData(
        long_name="Rainfall amount",
        standard_name="thickness_of_rainfall_amount",
        units="m",
        comment="Cumulated precipitation since 00:00 UTC",
        dimensions=("time",),
    ),
    "air_temperature": MetaData(
        long_name="Air temperature",
        standard_name="air_temperature",
        units="K",
        dimensions=("time",),
    ),
    "air_pressure": MetaData(
        long_name="Air pressure",
        standard_name="air_pressure",
        units="Pa",
        dimensions=("time",),
    ),
    "snr_limit": MetaData(
        long_name="SNR limit",
        units="dB",
        comment="SNR threshold used in data screening.",
        dimensions=None,
    ),
}
