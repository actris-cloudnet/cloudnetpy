"""Misc. plotting routines for Cloudnet products."""
import os.path
from dataclasses import dataclass
from datetime import date

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import Figure
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import Affine2D, Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma, ndarray

from cloudnetpy.plotting.plot_meta import ATTRIBUTES, PlotMeta


@dataclass
class PlotParameters:
    dpi: float = 120
    max_y: int = 12
    title: bool = True
    subtitle: bool = True
    mark_data_gaps: bool = True
    grid: bool = False
    edge_tick_labels: bool = False
    show_sources: bool = False
    footer_text: str | None = None
    plot_meta: PlotMeta | None = None


class Dimensions:
    """Dimensions of a generated figure in pixels."""

    width: int
    height: int
    margin_top: int
    margin_right: int
    margin_bottom: int
    margin_left: int

    def __init__(self, fig, axes, pad_inches: float | None = None):
        if pad_inches is None:
            pad_inches = rcParams["savefig.pad_inches"]

        tightbbox = (
            fig.get_tightbbox(fig.canvas.get_renderer())
            .padded(pad_inches)
            .transformed(Affine2D().scale(fig.dpi))
        )
        self.width = int(tightbbox.width)
        self.height = int(tightbbox.height)

        x0, y0, x1, y1 = (
            Bbox.union([ax.get_window_extent() for ax in axes])
            .translated(-tightbbox.x0, -tightbbox.y0)
            .extents
        )
        self.margin_top = int(self.height - round(y1))
        self.margin_right = int(self.width - round(x1) - 1)
        self.margin_bottom = int(round(y0) - 1)
        self.margin_left = int(round(x0))


class FigureData:
    def __init__(
        self,
        file: netCDF4.Dataset,
        requested_variables: list[str],
        options: PlotParameters,
    ):
        self.file = file
        self.variables, self.indices = self._get_valid_variables_and_indices(
            requested_variables
        )
        self.options = options
        self.height = self._get_height()
        self.time = self.file.variables["time"][:]
        self.time_including_gaps = np.array([])

    def initialize_figure(self) -> tuple[Figure, list[Axes]]:
        n_subplots = len(self)
        fig, axes = plt.subplots(
            n_subplots,
            1,
            figsize=(16, 4 + (n_subplots - 1) * 4.8),
            dpi=self.options.dpi,
        )
        fig.subplots_adjust(left=0.06, right=0.73)
        if n_subplots == 1:
            axes = [axes]
        return fig, axes

    def add_subtitle(self, fig: Figure) -> None:
        fig.suptitle(
            self._get_subtitle_text(),
            fontsize=13,
            y=0.885,
            x=0.07,
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    def _get_subtitle_text(self) -> str:
        measurement_date = date(
            int(self.file.year), int(self.file.month), int(self.file.day)
        )
        site_name = self.file.location.replace("-", " ")
        return f"{site_name}, {measurement_date.strftime('%d %b %Y').lstrip('0')}"

    def _get_valid_variables_and_indices(
        self, requested_variables: list[str]
    ) -> tuple[list[netCDF4.Variable], list[int | None]]:
        valid_variables = []
        variable_indices = []
        for variable_name in requested_variables:
            if variable_name.startswith("tb_"):
                extracted_name = "tb"
                extracted_ind = int(variable_name.split("_")[1])
            else:
                extracted_name = variable_name
                extracted_ind = None
            if extracted_name in self.file.variables:
                valid_variables.append(self.file.variables[extracted_name])
                variable_indices.append(extracted_ind)
        if not valid_variables:
            msg = f"None of the variables {requested_variables} found in the file."
            raise ValueError(msg)
        return valid_variables, variable_indices

    def _get_height(self) -> np.ndarray | None:
        m2km = 1e-3
        if self.file.cloudnet_file_type == "model":
            return ma.mean(self.file.variables["height"][:], axis=0) * m2km
        if "height" in self.file.variables:
            return self.file.variables["height"][:] * m2km
        if "range" in self.file.variables:
            return self.file.variables["range"][:] * m2km
        return None

    def __len__(self) -> int:
        return len(self.variables)


class SubPlot:
    def __init__(
        self,
        ax: Axes,
        variable: netCDF4.Variable,
        options: PlotParameters,
        file_type: str,
    ):
        self.ax = ax
        self.variable = variable
        self.options = options
        self.plot_meta = self._read_plot_meta(file_type)

    def set_xax(self) -> None:
        resolution = 4
        x_tick_labels = [
            f"{int(i):02d}:00"
            if (24 >= i >= 0 if self.options.edge_tick_labels else 24 > i > 0)
            else ""
            for i in np.arange(0, 24.01, resolution)
        ]
        self.ax.set_xticks(np.arange(0, 25, resolution, dtype=int))
        self.ax.set_xticklabels(x_tick_labels, fontsize=12)
        self.ax.set_xlim(0, 24)

    def set_yax(
        self,
        ylabel: str | None = None,
        y_limits: tuple[float, float] | None = None,
    ) -> None:
        label = ylabel or "Height (km)"
        self.ax.set_ylabel(label, fontsize=13)
        if y_limits is not None:
            self.ax.set_ylim(*y_limits)

    def add_title(self, ind: int | None) -> None:
        title = self.variable.long_name
        if self.variable.name == "tb" and ind is not None:
            title += f" (channel {ind + 1})"
        self.ax.set_title(title, fontsize=14)

    def add_grid(self) -> None:
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        self.ax.grid(which="major", axis="x", color="k", lw=0.1)
        self.ax.grid(which="minor", axis="x", lw=0.1, color="k", ls=":")
        self.ax.grid(which="major", axis="y", lw=0.1, color="k", ls=":")

    def add_sources(self, figure_data: FigureData) -> None:
        source = getattr(self.variable, "source", None) or (
            figure_data.file.source if "source" in figure_data.file.ncattrs() else None
        )
        if source is not None:
            source_word = "sources" if "\n" in source else "source"
            text = f"Data {source_word}:\n{source}"
            self.ax.text(
                0.012,
                0.96,
                text,
                ha="left",
                va="top",
                fontsize=7,
                transform=self.ax.transAxes,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "grey",
                    "boxstyle": "round",
                    "linewidth": 0.5,
                },
            )

    def set_xlabel(self) -> None:
        self.ax.set_xlabel("Time (UTC)", fontsize=13)

    def show_footer(self, fig: Figure):
        if isinstance(self.options.footer_text, str):
            fig.text(
                0.06,
                -0.05 + len(fig.get_axes()) / 50,
                self.options.footer_text,
                fontsize=11,
                ha="left",
                va="bottom",
            )

    def _read_plot_meta(self, file_type: str) -> PlotMeta:
        if self.options.plot_meta is not None:
            return self.options.plot_meta
        fallback = ATTRIBUTES["fallback"].get(self.variable.name, PlotMeta())
        file_attributes = ATTRIBUTES.get(file_type, {})
        plot_meta = file_attributes.get(self.variable.name, fallback)
        if plot_meta.clabel is None:
            plot_meta = plot_meta._replace(clabel=_reformat_units(self.variable.units))
        return plot_meta


class Plot:
    def __init__(self, sub_plot: SubPlot):
        self.sub_plot = sub_plot
        self._data = sub_plot.variable[:]
        self._plot_meta = sub_plot.plot_meta
        self._is_log = sub_plot.plot_meta.log_scale
        self._ax = sub_plot.ax

    def _get_y_limits(self) -> tuple[float, float]:
        return 0, self.sub_plot.options.max_y

    def _init_colorbar(self, plot) -> Colorbar:
        divider = make_axes_locatable(self._ax)
        cax = divider.append_axes("right", size="1%", pad=0.25)
        return plt.colorbar(plot, fraction=1.0, ax=self._ax, cax=cax)

    def _fill_between_data_gaps(self, figure_data: FigureData) -> None:
        gap_times = list(set(figure_data.time_including_gaps) - set(figure_data.time))
        gap_times.sort()
        batches = [gap_times[i : i + 2] for i in range(0, len(gap_times), 2)]
        for batch in batches:
            self._ax.fill_between(
                batch,
                *self._get_y_limits(),
                hatch="//",
                facecolor="grey",
                edgecolor="black",
                alpha=0.15,
            )

    def _mark_gaps(self, figure_data: FigureData) -> None:
        time = figure_data.time
        data = self._data
        if time[0] < 0 or time[-1] > 24:
            msg = "Time values outside the range 0-24."
            raise ValueError(msg)
        max_gap_fraction_hour = _get_max_gap_in_minutes(figure_data) / 60
        gap_indices = np.where(np.diff(time) > max_gap_fraction_hour)[0]
        if not ma.is_masked(data):
            mask_new = np.zeros(data.shape)
        elif ma.all(data.mask) is ma.masked:
            mask_new = np.ones(data.shape)
        else:
            mask_new = np.copy(data.mask)
        data_new = ma.copy(data)
        time_new = np.copy(time)
        if self._data.ndim == 2:
            temp_array = np.zeros((2, data.shape[1]))
            temp_mask = np.ones((2, data.shape[1]))
        else:
            temp_array = np.zeros(2)
            temp_mask = np.ones(2)
        time_delta = 0.001
        for ind in np.sort(gap_indices)[::-1]:
            ind_gap = ind + 1
            data_new = np.insert(data_new, ind_gap, temp_array, axis=0)
            mask_new = np.insert(mask_new, ind_gap, temp_mask, axis=0)
            time_new = np.insert(time_new, ind_gap, time[ind_gap] - time_delta)
            time_new = np.insert(time_new, ind_gap, time[ind_gap - 1] + time_delta)
        if (time[0] - 0) > max_gap_fraction_hour:
            data_new = np.insert(data_new, 0, temp_array, axis=0)
            mask_new = np.insert(mask_new, 0, temp_mask, axis=0)
            time_new = np.insert(time_new, 0, time[0] - time_delta)
            time_new = np.insert(time_new, 0, time_delta)
        if (24 - time[-1]) > max_gap_fraction_hour:
            ind_gap = mask_new.shape[0]
            data_new = np.insert(data_new, ind_gap, temp_array, axis=0)
            mask_new = np.insert(mask_new, ind_gap, temp_mask, axis=0)
            time_new = np.insert(time_new, ind_gap, 24 - time_delta)
            time_new = np.insert(time_new, ind_gap, time[-1] + time_delta)
        data_new.mask = mask_new
        self._data = data_new
        figure_data.time_including_gaps = time_new


class Plot2D(Plot):
    def plot(self, figure_data: FigureData):
        self._mark_gaps(figure_data)
        if self.sub_plot.variable.name == "cloud_fraction":
            self._data[self._data == 0] = ma.masked
        if any(
            key in self.sub_plot.variable.name for key in ("status", "classification")
        ):
            self._plot_segment_data(figure_data)
        else:
            self._plot_mesh_data(figure_data)

        if figure_data.options.mark_data_gaps:
            self._fill_between_data_gaps(figure_data)

    def _plot_segment_data(self, figure_data: FigureData) -> None:
        def _hide_segments(
            data_in: ma.MaskedArray,
        ) -> tuple[ma.MaskedArray, list, list]:
            if self._plot_meta.clabel is None:
                msg = f"No clabel defined for {self.sub_plot.variable.name}."
                raise ValueError(msg)
            labels = [x[0] for x in self._plot_meta.clabel]
            colors = [x[1] for x in self._plot_meta.clabel]
            segments_to_hide = np.char.startswith(labels, "_")
            indices = np.where(segments_to_hide)[0]
            for ind in np.flip(indices):
                del labels[ind], colors[ind]
                data_in[data_in == ind] = ma.masked
                data_in[data_in > ind] -= 1
            return data_in, colors, labels

        data, cbar, clabel = _hide_segments(self._data)
        alt = self._screen_data_by_max_y(figure_data)
        image = self._ax.pcolorfast(
            figure_data.time_including_gaps,
            alt,
            self._data.T[:-1, :-1],
            cmap=ListedColormap(cbar),
            vmin=-0.5,
            vmax=len(cbar) - 0.5,
        )
        colorbar = self._init_colorbar(image)
        colorbar.set_ticks(np.arange(len(clabel)).tolist())
        colorbar.ax.set_yticklabels(clabel, fontsize=13)

    def _plot_mesh_data(self, figure_data: FigureData) -> None:
        if self._plot_meta.plot_range is None:
            vmin, vmax = self._data.min(), self._data.max()
        else:
            vmin, vmax = self._plot_meta.plot_range
        if self._is_log:
            self._data, vmin, vmax = lin2log(self._data, vmin, vmax)

        alt = self._screen_data_by_max_y(figure_data)
        image = self._ax.pcolorfast(
            figure_data.time_including_gaps,
            alt,
            self._data.T[:-1, :-1],
            cmap=plt.get_cmap(str(self._plot_meta.cmap)),
            vmin=vmin,
            vmax=vmax,
        )
        cbar = self._init_colorbar(image)
        cbar.set_label(str(self._plot_meta.clabel), fontsize=13)
        if self._is_log:
            cbar.set_ticks(np.arange(vmin, vmax + 1).tolist())
            tick_labels = get_log_cbar_tick_labels(vmin, vmax)
            cbar.ax.set_yticklabels(tick_labels)

    def _screen_data_by_max_y(self, figure_data: FigureData) -> ndarray:
        if figure_data.height is None:
            msg = "No height information in the file."
            raise ValueError(msg)
        alt = figure_data.height
        if figure_data.options.max_y is None:
            return alt
        ind = int((np.argmax(alt > figure_data.options.max_y) or len(alt)) + 1)
        self._data = self._data[:, :ind]
        return alt[:ind]


class Plot1D(Plot):
    def plot_tb(self, figure_data: FigureData, ind: int):
        flagged_data = self._pointing_filter(figure_data, ind)
        self._ax.plot(
            figure_data.time,
            flagged_data,
            color="salmon",
            marker=".",
            lw=0,
            markersize=3,
        )
        self.plot(figure_data)
        self._ax.legend(
            ["Flagged data", "Valid data"],
            markerscale=3,
            numpoints=1,
            reverse=True,
            frameon=False,
        )

    def plot(self, figure_data: FigureData):
        units = self._convert_units()
        self._mark_gaps(figure_data)
        self._ax.plot(
            figure_data.time_including_gaps,
            self._data,
            **self._get_plot_options(),
        )
        if self._plot_meta.moving_average:
            self._plot_moving_average(figure_data)
        self._fill_between_data_gaps(figure_data)
        self.sub_plot.set_yax(ylabel=units, y_limits=self._get_y_limits())
        pos = self._ax.get_position()
        self._ax.set_position((pos.x0, pos.y0, pos.width * 0.965, pos.height))

    def _get_y_limits(self) -> tuple[float, float]:
        percent_gap = 0.05
        fallback = (-percent_gap, percent_gap)
        if ma.all(self._data.mask):
            return fallback
        min_data = self._data.min()
        max_data = self._data.max()
        range_val = max_data - min_data
        gap = percent_gap * range_val
        min_y = min_data - gap
        max_y = max_data + gap
        if min_y == 0 and max_y == 0:
            return fallback
        return min_y, max_y

    def _convert_units(self) -> str | None:
        multiply, add = "multiply", "add"
        units_conversion = {
            "rainfall_rate": (multiply, 360000, "mm h$^{-1}$"),
            "air_pressure": (multiply, 0.01, "hPa"),
            "relative_humidity": (multiply, 100, "%"),
            "rainfall_amount": (multiply, 1000, "mm"),
            "air_temperature": (add, -273.15, "\u00B0C"),
        }
        conversion_method, conversion, units = units_conversion.get(
            self.sub_plot.variable.name, (multiply, 1, None)
        )
        if conversion_method == multiply:
            self._data *= conversion
        elif conversion_method == add:
            self._data += conversion
        if units is not None:
            return units
        return _reformat_units(self.sub_plot.variable.units)

    def _get_plot_options(self) -> dict:
        default_options = {
            "color": "lightblue",
            "lw": 0,
            "marker": ".",
            "markersize": 3,
        }
        custom_options = {
            "tb": {
                "color": "lightblue",
            }
        }

        variable_name = self.sub_plot.variable.name
        if variable_name in custom_options:
            default_options.update(custom_options[variable_name])

        return default_options

    @staticmethod
    def _get_line_width(time: np.ndarray) -> float:
        line_width = np.median(np.diff(time)) * 1000
        return min(max(line_width, 0.25), 0.9)

    def _plot_moving_average(self, figure_data: FigureData):
        time = figure_data.time_including_gaps.copy()
        data, time = self._get_unmasked_values(self._data, time)
        sma = self._calculate_moving_average(data, time, window=5)
        gap_time = _get_max_gap_in_minutes(figure_data)
        gaps = self._find_time_gap_indices(time, max_gap_min=gap_time)
        sma[gaps] = np.nan
        self._ax.plot(time, sma, color="slateblue", lw=2)

    @staticmethod
    def _get_unmasked_values(
        data: ma.MaskedArray,
        time: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not ma.is_masked(data):
            return data, time
        good_values = ~data.mask
        return data[good_values], time[good_values]

    def _pointing_filter(self, figure_data: FigureData, ind: int) -> ndarray:
        zenith_limit = 5
        status = 0
        self._data = self._data[:, ind]
        flagged_data = ma.masked_all_like(figure_data.time)
        if "pointing_flag" in figure_data.file.variables:
            pointing_flag = figure_data.file.variables["pointing_flag"][:]
            zenith_angle = figure_data.file.variables["zenith_angle"][:]
            quality_flag = figure_data.file.variables["quality_flag"][:, ind]
            # First mask bad zenith angle points
            self._data[np.abs(zenith_angle) > zenith_limit] = ma.masked
            self._data[pointing_flag != status] = ma.masked
            # Store flagged data points for visualization
            valid_ind = np.where(quality_flag != status)[0]
            if len(valid_ind) > 0:
                flagged_data[valid_ind] = self._data[valid_ind]
            self._data[quality_flag != status] = ma.masked
        return flagged_data

    @staticmethod
    def _find_time_gap_indices(time: ndarray, max_gap_min: float) -> ndarray:
        gap_decimal_hour = max_gap_min / 60
        return np.where(np.diff(time) > gap_decimal_hour)[0]

    @staticmethod
    def _calculate_moving_average(
        data: np.ndarray, time: np.ndarray, window: float = 5
    ) -> np.ndarray:
        if len(data) == 0:
            return np.array([])
        time_delta_hours = np.median(np.diff(time))
        window_size = int(window / 60 / time_delta_hours)
        if window_size < 1:
            window_size = 1
        if (window_size % 2) != 0:
            window_size += 1
        weights = np.repeat(1.0, window_size) / window_size
        sma = np.convolve(data, weights, "valid")
        edge = window_size // 2
        return np.pad(sma, (edge, edge - 1), mode="constant", constant_values=np.nan)


def generate_figure(
    filename: os.PathLike | str,
    variables: list[str],
    *,
    show: bool = True,
    output_filename: os.PathLike | str | None = None,
    options: PlotParameters | None = None,
) -> Dimensions:
    if options is None:
        options = PlotParameters()

    with netCDF4.Dataset(filename) as file:
        figure_data = FigureData(file, variables, options)
        fig, axes = figure_data.initialize_figure()

        for ax, variable, ind in zip(
            axes, figure_data.variables, figure_data.indices, strict=True
        ):
            subplot = SubPlot(ax, variable, options, file.cloudnet_file_type)

            if variable.name == "tb" and ind is not None:
                Plot1D(subplot).plot_tb(figure_data, ind)
            elif variable.ndim == 1:
                Plot1D(subplot).plot(figure_data)
            else:
                Plot2D(subplot).plot(figure_data)
                subplot.set_yax(y_limits=(0, figure_data.options.max_y))

            subplot.set_xax()

            if options.title:
                subplot.add_title(ind)

            if options.grid:
                subplot.add_grid()

            if options.show_sources:
                subplot.add_sources(figure_data)

            if options.subtitle and variable == figure_data.variables[-1]:
                figure_data.add_subtitle(fig)

    subplot.set_xlabel()

    if options.footer_text is not None:
        subplot.show_footer(fig)

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

    return Dimensions(fig, axes)


def lin2log(*args) -> list:
    return [ma.log10(x) for x in args]


def get_log_cbar_tick_labels(value_min: float, value_max: float) -> list[str]:
    return [f"10$^{{{int(i)}}}$" for i in np.arange(value_min, value_max + 1)]


def _reformat_units(unit: str) -> str:
    unit_mapping = {
        "1": "",
        "mu m": "$\\mu$m",
        "m-3": "m$^{-3}$",
        "m s-1": "m s$^{-1}$",
        "sr-1 m-1": "sr$^{-1}$ m$^{-1}$",
        "kg m-2": "kg m$^{-2}$",
        "kg m-3": "kg m$^{-3}$",
        "kg m-2 s-1": "kg m$^{-2}$ s$^{-1}$",
        "dB km-1": "dB km$^{-1}$",
        "rad km-1": "rad km$^{-1}$",
    }
    if unit in unit_mapping:
        return unit_mapping[unit]
    return unit


def _get_max_gap_in_minutes(figure_data: FigureData) -> float:
    max_allowed_gap = {
        "model": 181 if "gdas1" in figure_data.file.source else 61,
        "mwr-multi": 21,
    }
    return max_allowed_gap.get(figure_data.file.cloudnet_file_type, 10)


def plot_2d(
    data: ma.MaskedArray,
    cmap: str = "viridis",
    ncolors: int = 50,
    clim: tuple | None = None,
    ylim: tuple | None = None,
    xlim: tuple | None = None,
    *,
    cbar: bool = True,
) -> None:
    """Simple plot of 2d variable."""
    plt.close()
    if cbar:
        color_map = plt.get_cmap(cmap, ncolors)
        plt.imshow(
            ma.masked_equal(data, 0).T,
            aspect="auto",
            origin="lower",
            cmap=color_map,
        )
        plt.colorbar()
    else:
        plt.imshow(ma.masked_equal(data, 0).T, aspect="auto", origin="lower")
    if clim:
        plt.clim(clim[0], clim[1])
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()
