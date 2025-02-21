"""Metadata for plotting module."""

from collections.abc import Sequence
from typing import NamedTuple


class PlotMeta(NamedTuple):
    """A class representing the metadata for plotting.

    Attributes:
        cmap: The colormap to be used for the plot.
        clabel: The label for the colorbar. It can be a single string, a sequence
            of tuples containing the label and units for each colorbar, or None
            if no colorbar is needed.
        plot_range: The range of values to be plotted. It can be a tuple
            containing the minimum and maximum values, or None if the range should
            be automatically determined.
        log_scale: Whether to plot data values in a logarithmic scale.
        moving_average: Whether to plot a moving average in a 1d plot.
        contour: Whether to plot contours on top of a filled colormap.
        zero_line: Whether to plot a zero line in a 1d plot.
        time_smoothing_duration: The duration of the time smoothing window
            (in 2d plots) in minutes.
    """

    cmap: str = "viridis"
    clabel: str | Sequence[tuple[str, str]] | None = None
    plot_range: tuple[float, float] | None = None
    log_scale: bool = False
    moving_average: bool = True
    contour: bool = False
    zero_line: bool = False
    time_smoothing_duration: int = 0


_COLORS = {
    "green": "#3cb371",
    "darkgreen": "#253A24",
    "lightgreen": "#70EB5D",
    "yellowgreen": "#C7FA3A",
    "yellow": "#FFE744",
    "orange": "#ffa500",
    "pink": "#B43757",
    "red": "#F57150",
    "shockred": "#E64A23",
    "seaweed": "#646F5E",
    "seaweed_roll": "#748269",
    "white": "#ffffff",
    "lightblue": "#6CFFEC",
    "blue": "#209FF3",
    "skyblue": "#CDF5F6",
    "darksky": "#76A9AB",
    "darkpurple": "#464AB9",
    "lightpurple": "#6A5ACD",
    "purple": "#BF9AFF",
    "darkgray": "#2f4f4f",
    "lightgray": "#ECECEC",
    "gray": "#d3d3d3",
    "lightbrown": "#CEBC89",
    "lightsteel": "#a0b0bb",
    "steelblue": "#4682b4",
    "mask": "#C8C8C8",
}

# Labels (and corresponding data) starting with an underscore are NOT shown:

_CLABEL = {
    "target_classification": (
        ("_Clear sky", _COLORS["white"]),
        ("Droplets", _COLORS["lightblue"]),
        ("Drizzle or rain", _COLORS["blue"]),
        ("Drizzle & droplets", _COLORS["purple"]),
        ("Ice", _COLORS["lightsteel"]),
        ("Ice & droplets", _COLORS["darkpurple"]),
        ("Melting ice", _COLORS["orange"]),
        ("Melting & droplets", _COLORS["yellowgreen"]),
        ("Aerosols", _COLORS["lightbrown"]),
        ("Insects", _COLORS["shockred"]),
        ("Aerosols & insects", _COLORS["pink"]),
    ),
    "detection_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Lidar only", _COLORS["yellow"]),
        ("Uncorrected atten.", _COLORS["seaweed_roll"]),
        ("Radar & lidar", _COLORS["green"]),
        ("_No radar but unknown atten.", _COLORS["purple"]),
        ("Radar only", _COLORS["lightgreen"]),
        ("_No radar but known atten.", _COLORS["orange"]),
        ("Corrected atten.", _COLORS["skyblue"]),
        ("Clutter", _COLORS["shockred"]),
        ("_Lidar molecular scattering", _COLORS["pink"]),
    ),
    "ice_retrieval_status": (
        ("_No ice", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Uncorrected", _COLORS["orange"]),
        ("Corrected", _COLORS["lightgreen"]),
        ("Ice from lidar", _COLORS["yellow"]),
        ("_Ice above rain", _COLORS["darksky"]),
        ("Clear above rain", _COLORS["skyblue"]),
        ("Positive temp.", _COLORS["seaweed"]),
    ),
    "lwc_retrieval_status": (
        ("No liquid", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Adjusted", _COLORS["lightgreen"]),
        ("New pixel", _COLORS["yellow"]),
        ("Invalid LWP", _COLORS["seaweed_roll"]),
        ("_Invalid LWP2", _COLORS["shockred"]),
        ("_Measured rain", _COLORS["orange"]),
    ),
    "drizzle_retrieval_status": (
        ("_No drizzle", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Below melting", _COLORS["lightgreen"]),
        ("Unfeasible", _COLORS["red"]),
        ("Drizzle-free", _COLORS["orange"]),
        ("Rain", _COLORS["seaweed"]),
    ),
    "der_retrieval_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Reliable", _COLORS["green"]),
        ("Mixed phase", _COLORS["lightgreen"]),
        ("Unfeasible", _COLORS["red"]),
        ("Surrounding ice", _COLORS["lightsteel"]),
    ),
}


_MWR_SINGLE_SMOOTHING = 10
_MWR_MULTI_SMOOTHING = 30

ATTRIBUTES = {
    "rain-radar": {
        "rainfall_rate": PlotMeta(
            cmap="Blues",
            plot_range=(0, 50 / 3600000),
        )
    },
    "mwr": {
        "temperature": PlotMeta(
            cmap="coolwarm",
            plot_range=(223.15, 323.15),
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
        "relative_humidity": PlotMeta(
            plot_range=(0, 120),
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
        "absolute_humidity": PlotMeta(
            plot_range=(1e-4, 1e-2),
            log_scale=True,
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
    },
    "mwr-single": {
        "temperature": PlotMeta(
            cmap="coolwarm",
            plot_range=(223.15, 323.15),
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
        "potential_temperature": PlotMeta(
            cmap="coolwarm",
            plot_range=(260, 320),
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
        "equivalent_potential_temperature": PlotMeta(
            cmap="coolwarm",
            plot_range=(260, 320),
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
        "relative_humidity": PlotMeta(
            plot_range=(0, 120),
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
        "absolute_humidity": PlotMeta(
            plot_range=(1e-4, 1e-2),
            log_scale=True,
            contour=True,
            time_smoothing_duration=_MWR_SINGLE_SMOOTHING,
        ),
    },
    "mwr-multi": {
        "temperature": PlotMeta(
            cmap="coolwarm",
            plot_range=(223.15, 323.15),
            contour=True,
            time_smoothing_duration=_MWR_MULTI_SMOOTHING,
        ),
        "potential_temperature": PlotMeta(
            cmap="coolwarm",
            plot_range=(260, 320),
            contour=True,
            time_smoothing_duration=_MWR_MULTI_SMOOTHING,
        ),
        "equivalent_potential_temperature": PlotMeta(
            cmap="coolwarm",
            plot_range=(260, 320),
            contour=True,
            time_smoothing_duration=_MWR_MULTI_SMOOTHING,
        ),
        "relative_humidity": PlotMeta(
            plot_range=(0, 120),
            contour=True,
            time_smoothing_duration=_MWR_MULTI_SMOOTHING,
        ),
    },
    "fallback": {
        "nubf": PlotMeta(plot_range=(0, 5)),
        "ze_sat": PlotMeta(
            plot_range=(-40, 15),
        ),
        "vm_sat": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-4, 4),
        ),
        "vm_sat_folded": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-4, 4),
        ),
        "vm_sat_noise": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-4, 4),
        ),
        "vm_sat_vel": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-4, 4),
        ),
        "ier": PlotMeta(
            plot_range=(2e-5, 6e-5),
        ),
        "ier_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(1e-5, 5e-5),
        ),
        "ier_inc_rain": PlotMeta(
            plot_range=(2e-5, 6e-5),
        ),
        "ier_retrieval_status": PlotMeta(
            clabel=_CLABEL["ice_retrieval_status"],
        ),
        "Do": PlotMeta(
            plot_range=(1e-6, 1e-3),
            log_scale=True,
        ),
        "Do_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0.1, 0.5),
        ),
        "der": PlotMeta(
            cmap="coolwarm",
            plot_range=(1.0e-6, 1.0e-4),
            log_scale=True,
        ),
        "N_scaled": PlotMeta(
            plot_range=(1.0e0, 1e3),
            log_scale=True,
        ),
        "der_error": PlotMeta(
            cmap="coolwarm",
            plot_range=(1.0e-6, 1.0e-4),
            log_scale=True,
        ),
        "der_scaled": PlotMeta(
            cmap="coolwarm",
            plot_range=(1.0e-6, 1.0e-4),
            log_scale=True,
        ),
        "der_scaled_error": PlotMeta(
            cmap="coolwarm",
            plot_range=(1.0e-6, 1.0e-4),
            log_scale=True,
        ),
        "der_retrieval_status": PlotMeta(
            clabel=_CLABEL["der_retrieval_status"],
        ),
        "mu": PlotMeta(
            plot_range=(0, 10),
        ),
        "S": PlotMeta(
            plot_range=(0, 25),
        ),
        "S_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0.1, 0.5),
        ),
        "drizzle_N": PlotMeta(
            plot_range=(1e4, 1e9),
            log_scale=True,
        ),
        "drizzle_N_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0.1, 0.5),
        ),
        "drizzle_lwc": PlotMeta(
            plot_range=(1e-8, 1e-3),
            log_scale=True,
        ),
        "drizzle_lwc_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0.3, 1),
        ),
        "drizzle_lwf": PlotMeta(
            plot_range=(1e-8, 1e-5),
            log_scale=True,
        ),
        "drizzle_lwf_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0.3, 1),
        ),
        "v_drizzle": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-2, 2),
        ),
        "v_drizzle_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0.3, 1),
        ),
        "drizzle_retrieval_status": PlotMeta(
            clabel=_CLABEL["drizzle_retrieval_status"],
        ),
        "v_air": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-2, 2),
        ),
        "uwind": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-50, 50),
        ),
        "uwind_raw": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-50, 50),
        ),
        "vwind": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-50, 50),
        ),
        "vwind_raw": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-50, 50),
        ),
        "temperature": PlotMeta(
            cmap="RdBu_r",
            plot_range=(223.15, 323.15),
        ),
        "cloud_fraction": PlotMeta(
            cmap="Blues",
            plot_range=(0, 1),
        ),
        "Tw": PlotMeta(
            cmap="RdBu_r",
            plot_range=(223.15, 323.15),
        ),
        "specific_humidity": PlotMeta(
            plot_range=(1e-5, 1e-2),
            log_scale=True,
        ),
        "q": PlotMeta(
            plot_range=(1e-5, 1e-2),
            log_scale=True,
        ),
        "number_concentration": PlotMeta(plot_range=(1e-2, 1e3), log_scale=True),
        "fall_velocity": PlotMeta(
            plot_range=(0, 10),
        ),
        "pressure": PlotMeta(
            plot_range=(1e4, 1.2e5),
            log_scale=True,
        ),
        "beta": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_raw": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_1064_raw": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_1064": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_532_raw": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_532": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_532_nr_raw": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_532_nr": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_355_raw": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_355": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "beta_355_nr_raw": PlotMeta(
            plot_range=(1e5, 1e8),
            log_scale=True,
        ),
        "beta_355_nr": PlotMeta(
            plot_range=(1e5, 1e8),
            log_scale=True,
        ),
        "beta_smooth": PlotMeta(
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "depolarisation_raw": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation_1064_raw": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation_1064": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation_532_raw": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation_532": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation_355_raw": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation_355": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "depolarisation_smooth": PlotMeta(
            plot_range=(1e-3, 1),
            log_scale=True,
        ),
        "Z": PlotMeta(
            plot_range=(-40, 15),
        ),
        "Z_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0, 3),
        ),
        "Zh": PlotMeta(
            plot_range=(-40, 15),
        ),
        "ldr": PlotMeta(
            plot_range=(-30, -5),
        ),
        "sldr": PlotMeta(
            plot_range=(-30, -5),
        ),
        "zdr": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-1, 1),
        ),
        "width": PlotMeta(
            plot_range=(1e-2, 1e0),
            log_scale=True,
        ),
        "v": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-4, 4),
        ),
        "skewness": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-1, 1),
        ),
        "kurtosis": PlotMeta(
            plot_range=(1, 5),
        ),
        "phi_cx": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-2, 2),
        ),
        "differential_attenuation": PlotMeta(
            plot_range=(0, 1),
        ),
        "rho_cx": PlotMeta(
            plot_range=(1e-2, 1e0),
            log_scale=True,
        ),
        "rho_hv": PlotMeta(
            plot_range=(0.8, 1),
        ),
        "srho_hv": PlotMeta(
            plot_range=(0, 0.5),
        ),
        "phi_dp": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-0.1, 0.1),
        ),
        "kdp": PlotMeta(
            cmap="RdBu_r",
            plot_range=(-0.1, 0.1),
        ),
        "v_sigma": PlotMeta(
            plot_range=(1e-2, 1e0),
            log_scale=True,
        ),
        "insect_prob": PlotMeta(
            plot_range=(0, 1),
        ),
        "liquid_prob": PlotMeta(
            plot_range=(0, 0.7),
        ),
        "radar_liquid_atten": PlotMeta(
            plot_range=(0, 5),
        ),
        "radar_gas_atten": PlotMeta(
            plot_range=(0, 5),
        ),
        "radar_rain_atten": PlotMeta(
            plot_range=(0, 5),
        ),
        "radar_melting_atten": PlotMeta(
            plot_range=(0, 5),
        ),
        "iwv": PlotMeta(
            cmap="Blues",
            plot_range=(0, 1),
        ),
        "target_classification": PlotMeta(
            clabel=_CLABEL["target_classification"],
        ),
        "detection_status": PlotMeta(
            clabel=_CLABEL["detection_status"],
        ),
        "iwc": PlotMeta(
            plot_range=(1e-7, 1e-3),
            log_scale=True,
        ),
        "iwc_inc_rain": PlotMeta(
            cmap="Blues",
            plot_range=(1e-7, 1e-4),
            log_scale=True,
        ),
        "iwc_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0, 5),
        ),
        "iwc_retrieval_status": PlotMeta(
            clabel=_CLABEL["ice_retrieval_status"],
        ),
        "lwc": PlotMeta(
            cmap="Blues",
            plot_range=(1e-5, 1e-2),
            log_scale=True,
        ),
        "lwc_error": PlotMeta(
            cmap="RdYlGn_r",
            plot_range=(0, 2),
        ),
        "lwc_retrieval_status": PlotMeta(
            clabel=_CLABEL["lwc_retrieval_status"],
        ),
        "pia": PlotMeta(
            plot_range=(0, 3),
        ),
        "lwp": PlotMeta(
            zero_line=True,
        ),
        "epsilon": PlotMeta(
            cmap="inferno",
            plot_range=(1e-7, 1e-1),
            log_scale=True,
        ),
    },
}
