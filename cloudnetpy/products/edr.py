"""Module for creating Cloudnet droplet effective radius
using the Frisch et al. 2002 method.
"""

import logging

import numpy as np
from numpy import ma
from scipy.interpolate import CubicSpline

from cloudnetpy import output  # , utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products import product_tools


def generate_edr(
    radar_file: str,
    categorize_file: str,
    output_file: str,
    uuid: str | None = None,
) -> str:
    """Generates Cloudnet eddy dissipation rates
    product according to Griesche 2020 / Borque 2016.

    This function calculates the eddy dissipation rate.
    The results are written in a netCDF file.

    Args:
        radar_file: Radar file name.
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_edr
        >>> generate_edr('radar_file.nc','categorize.nc', 'edr.nc')

    References:
        Griesche, H. J., Seifert, P., Ansmann, A., Baars, H., Barrientos Velasco,
        C., Bühl, J., Engelmann, R., Radenz, M., Zhenping, Y., and Macke, A. (2020):
        Application of the shipborne remote sensing supersite OCEANET for
        profiling of Arctic aerosols and clouds during Polarstern cruise PS106,
        Atmos. Meas. Tech., 13, 5335–5358,
        from https://doi.org/10.5194/amt-13-5335-2020.

        Borque, P., Luke, E., and Kollias, P. (2016):
        On the uniﬁed estimation of turbulence eddy dissipation rate using Doppler
        cloud radars and lidars, J. Geophys. Res.Atmos., 120, 5972–5989,
        from https://doi.org/10.1002/2015JD024543.

    """
    wind_source = WindSource(categorize_file)
    edr_source = EdrSource(radar_file)
    edr_source.append_edr(wind_source)
    edr_source.update_time_dependent_variables(wind_source)
    date = edr_source.get_date()
    attributes = output.add_time_attribute(EDR_ATTRIBUTES, date)
    output.update_attributes(edr_source.data, attributes)
    uuid = output.save_product_file("edr", edr_source, output_file, uuid)
    edr_source.close()
    return uuid


class WindSource(DataSource):
    """Data container for wind for eddy dissipation rate calculations."""

    #
    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)

    #
    def get_wind(self) -> np.ndarray:
        uwind = self.getvar("uwind")
        vwind = self.getvar("vwind")
        model_time = self.getvar("model_time")
        model_height = self.getvar("model_height")
        return product_tools.get_interpolated_horizontal_wind(
            uwind, vwind, model_time, model_height
        )

    #
    def get_cat_time(self) -> np.ndarray:
        return self.getvar("time")


class EdrSource(DataSource):
    """Data container for eddy dissipation rate calculations."""

    #
    def __init__(self, radar_file: str):
        super().__init__(radar_file)

    def update_time_dependent_variables(self, wind_source, copy_from_cat: tuple = ()):
        """Update the temporal high resolved variables that will be stored in
        the product file to the standard Cloudnet product time resolution of
        30s.
        """
        self.dataset.variables["time"] = wind_source.dataset.variables["time"]
        self.time = wind_source.getvar("time")
        vars_to_be_copied_from_source = (
            "altitude",
            "latitude",
            "longitude",
            "time",
            "height",
            *copy_from_cat,
        )
        for key in vars_to_be_copied_from_source:
            if self.dataset.variables[key].size > 0:
                self.dataset.variables[key] = wind_source.dataset.variables[key]

    def append_edr(self, wind_source):
        """Estimate Eddy Dissipation Rate (EDR) according to Griesche et al. 2020."""

        def interpolate_missing_values(
            data: np.ndarray, time_window: np.ndarray
        ) -> np.ndarray | None:
            """Interpolate missing Doppler velocity values using a cubic spline.

            Returns:
            - interpolated_data: np.ndarray
                Data with missing values interpolated, or None if insufficient data.
            """
            valid_indices = ma.where(data)[0]
            if (
                valid_indices.shape[0] < 0.9 * data.shape[0]
            ):  # Minimum threshold for valid data
                return None
            if ma.is_masked(data[0]) or ma.is_masked(
                data[-1]
            ):  # Avoid edge interpolation issues
                return None

            spline = CubicSpline(time_window[valid_indices], data[valid_indices])
            return spline(time_window)

        def compute_epsilon_from_spectrum(
            freq_sp: np.ndarray,
            power_sp: np.ndarray,
            freq_list: list[list[float]],
            constant: float,
        ) -> list[float]:
            """Compute epsilon values from power spectrum."""
            epsilon_values = []

            for freq_range in freq_list:
                mask = (freq_sp > freq_range[0]) & (freq_sp < freq_range[1])
                if freq_sp[mask].shape[0] > 1 and power_sp[mask].shape[0] > 1:
                    fit = product_tools.spec_fit(freq_sp, power_sp, freq_range)
                    slope = fit[0]
                    intercept = fit[1]
                    if np.isnan(slope) or not (
                        THRESHOLD_SLOPE_MIN < slope < THRESHOLD_SLOPE_MAX
                    ):
                        continue
                    epsilon_values.append((10**intercept / constant) ** (3.0 / 2.0))
            return epsilon_values

        # Predefinded Constants
        KOLMOGOROV_CONSTANT = 0.5
        THRESHOLD_SLOPE_MIN = -2  # based on Borque, 2016, JGR (-5/3 +-20%)
        THRESHOLD_SLOPE_MAX = -1.33  # based on Borque, 2016, JGR (-5/3 +-20%)
        AVERAGING_TIME_MIN = 5  # Averaging time in minutes
        H_to_S = 3600  # hours to seconds
        H_to_M = 60  # hours to minutes

        # Input data
        freq_list = standard_freq_list  # predefined list of frequency ranges to analyze
        radar_time = self.getvar("time")
        radar_time_resolution_sec = np.round(
            np.nanmin(np.diff(radar_time)) * H_to_S
        )  # in seconds
        height = self.getvar("height")
        vertical_velocity = self.getvar("v")

        horizontal_wind_function = wind_source.get_wind()
        horizontal_wind = horizontal_wind_function(radar_time, height)
        categorize_time = wind_source.get_cat_time()
        product_resolution = np.round(
            np.nanmin(np.diff(categorize_time)) * H_to_S
        )  # in seconds

        num_height_levels = height.shape[0]
        epsilon = (
            np.ones((int(np.ceil(categorize_time.shape[0])), num_height_levels))
            * -999.0
        )
        logging.debug(product_resolution, radar_time_resolution_sec)

        # Process each time interval
        for time_idx, t in enumerate(categorize_time):
            logging.debug(time_idx, t)
            if t + AVERAGING_TIME_MIN / H_to_M > radar_time[-1]:
                break
            #
            start_idx = np.where(radar_time > t)[0][0]
            end_idx = np.where(radar_time > t + AVERAGING_TIME_MIN / H_to_M)[0][0]

            # Process each height level
            for height_idx, _h in enumerate(height[:num_height_levels]):
                data_velocity = vertical_velocity[start_idx:end_idx, height_idx]
                time_window = radar_time[start_idx:end_idx]

                if ma.is_masked(data_velocity):
                    data_velocity = interpolate_missing_values(
                        data_velocity, time_window
                    )  # type: ignore[assignment]
                if data_velocity is None:
                    continue

                horizontal_wind_speed = horizontal_wind[
                    int((start_idx + end_idx) / 2), height_idx
                ]
                if np.isnan(horizontal_wind_speed):
                    continue

                # Compute frequency spectrum
                freq_sp, power_sp = product_tools.periodogram(
                    data_velocity, radar_time_resolution_sec, horizontal_wind_speed
                )
                if np.isnan(freq_sp).all():
                    continue
                # Compute epsilon for frequency ranges
                epsilon_values = compute_epsilon_from_spectrum(
                    freq_sp, power_sp, freq_list, KOLMOGOROV_CONSTANT
                )

                # Store average epsilon
                if epsilon_values:
                    epsilon[time_idx, height_idx] = np.mean(epsilon_values)

        self.append_data(ma.masked_where(epsilon == -999.0, epsilon), "edr")


COMMENTS = {
    "general": (
        "This dataset contains the eddy dissipation rate calculated\n"
        "according to the approach presented by Griesche et al. (2020). It is based\n"
        "on a relationship between the slope of the inertial subrange of the\n"
        "turbulent energy spectrum. If spectrum follows a -5/3 slope in a log-log\n"
        "the eddy dissipation rate can be calculated. The turbulent energy spectrum\n"
        "was derived as the power spectrum of cloud radar Doppler velocity averaged\n"
        "over 5 minutes.\n"
    ),
    "edr": (
        "This variable was calculated for the profiles where 5 minutes of\n"
        "continuous cloud radar Doppler velocity was available. In case of\n"
        "than lass than 10% missing data, the missing data points were interpolated\n"
        "using cubic spine interpolation.\n"
    ),
}


EDR_ATTRIBUTES = {
    "comment": COMMENTS["general"],
    "edr": MetaData(
        long_name="Eddy dissipation rate",
        units="m2 s-3",
        ancillary_variables="edr_error",
        comment=COMMENTS["edr"],
    ),
    "edr_error": MetaData(
        long_name="Absolute error in eddy dissipation rate",
        units="m2 s-3",
    ),
}


standard_freq_list = [
    [5e-3, 5e-2],
    [5e-3, 1e-1],
    [5e-3, 35e-2],
    [5e-3, 8e-1],
    [1e-2, 1e-1],
    [1e-2, 35e-2],
    [1e-2, 8e-1],
    [2e-2, 2e-1],
    [2e-2, 8e-1],
    [35e-3, 2e-1],
    [35e-3, 8e-1],
    [5e-2, 2e-1],
    [5e-2, 8e-1],
    [8e-2, 35e-2],
    [8e-2, 8e-1],
    [1e-1, 5e-1],
    [1e-1, 8e-1],
]  # adapted from Griesche et al. 2020
