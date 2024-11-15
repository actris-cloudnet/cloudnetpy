"""Misc. plotting routines for Cloudnet products."""

import os.path
import re
import textwrap
from dataclasses import dataclass
from datetime import date
from typing import Any

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
from scipy.ndimage import uniform_filter

from cloudnetpy.constants import T0
from cloudnetpy.exceptions import PlottingError
from cloudnetpy.instruments.ceilometer import calc_sigma_units
from cloudnetpy.plotting.plot_meta import ATTRIBUTES, PlotMeta


@dataclass
class PlotParameters:
    """Class representing the parameters for plotting.

    Attributes:
        dpi: The resolution of the plot in dots per inch.
        max_y: Maximum y-axis value (km) in 2D time / height plots.
        title: Whether to display the title of the plot.
        subtitle: Whether to display the subtitle of the plot.
        mark_data_gaps: Whether to mark data gaps in the plot.
        grid: Whether to display grid lines in the plot.
        edge_tick_labels: Whether to display tick labels on the edges of the plot.
        show_sources: Whether to display the sources of plotted data (i.e.
            instruments and model).
        footer_text: The text to display in the footer of the plot.
        plot_meta: Additional metadata for the plot.
        raise_on_empty: Whether to raise an error if no data is found for a
            plotted variable.
    """

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
    raise_on_empty: bool = False


class Dimensions:
    """Dimensions of a generated figure in pixels. Elements such as the figure
    title, labels, colorbar and legend are excluded from the margins.

    Attributes:
        width (int): Figure width in pixels.
        height (int): Figure height in pixels.
        margin_top (int): Space between top edge of image and plotted data in pixels.
        margin_right (int): Space between right edge of image and plotted data
            in pixels.
        margin_bottom (int): Space between bottom edge of image and plotted
            data in pixels.
        margin_left (int): Space between left edge of image and plotted data in pixels.
    """

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
            sharex=True,
        )
        fig.subplots_adjust(left=0.06, right=0.73)
        axes_list = [axes] if isinstance(axes, Axes) else axes.tolist()
        return fig, axes_list

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
            if variable_name.startswith(("tb_", "irt_")):
                parts = variable_name.split("_")
                extracted_name = parts[0]
                extracted_ind = int(parts[1])
            else:
                extracted_name = variable_name
                extracted_ind = None
            if extracted_name in self.file.variables:
                valid_variables.append(self.file.variables[extracted_name])
                variable_indices.append(extracted_ind)
        if not valid_variables:
            msg = f"None of the variables {requested_variables} found in the file."
            raise PlottingError(msg)
        return valid_variables, variable_indices

    def _get_height(self) -> ndarray | None:
        m2km = 1e-3
        m2mm = 1e3
        file_type = getattr(self.file, "cloudnet_file_type", "")
        if file_type == "model":
            return ma.mean(self.file.variables["height"][:], axis=0) * m2km
        if "height" in self.file.variables:
            return self.file.variables["height"][:] * m2km
        if "range" in self.file.variables:
            return self.file.variables["range"][:] * m2km
        if "diameter" in self.file.variables:
            return self.file.variables["diameter"][:] * m2mm
        return None

    def is_mwrpy_product(self) -> bool:
        cloudnet_file_type = getattr(self.file, "cloudnet_file_type", "")
        return cloudnet_file_type in ("mwr-single", "mwr-multi")

    def __len__(self) -> int:
        return len(self.variables)


class SubPlot:
    def __init__(
        self,
        ax: Axes,
        variable: netCDF4.Variable,
        options: PlotParameters,
        file_type: str | None,
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
        label = ylabel if ylabel is not None else "Height (km)"
        self.ax.set_ylabel(label, fontsize=13)
        if y_limits is not None:
            self.ax.set_ylim(*y_limits)

    def add_title(self, ind: int | None) -> None:
        title = self.variable.long_name
        if self.variable.name == "tb" and ind is not None:
            title += f" (channel {ind + 1})"
        self.ax.set_title(title, fontsize=14)

    def add_grid(self) -> None:
        zorder = _get_zorder("grid")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        self.ax.grid(which="major", axis="x", color="k", lw=0.1, zorder=zorder)
        self.ax.grid(which="minor", axis="x", lw=0.1, color="k", ls=":", zorder=zorder)
        self.ax.grid(which="major", axis="y", lw=0.1, color="k", ls=":", zorder=zorder)

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

    def show_footer(self, fig: Figure, ax: Axes) -> None:
        if isinstance(self.options.footer_text, str):
            n = 50
            if len(self.options.footer_text) > n:
                wrapped_text = textwrap.fill(self.options.footer_text, n)
                self.options.footer_text = "\n".join(wrapped_text.splitlines())

            n_lines = self.options.footer_text.count("\n") + 1
            y0 = ax.get_position().y0
            y1 = ax.get_position().y1
            y = (y1 - y0) * (n_lines * 0.06 + 0.1)
            fig.text(
                0.06,
                y0 - y,
                self.options.footer_text,
                fontsize=11,
                ha="left",
                va="bottom",
            )

    def _read_plot_meta(self, file_type: str | None) -> PlotMeta:
        if self.options.plot_meta is not None:
            plot_meta = self.options.plot_meta
        else:
            fallback = ATTRIBUTES["fallback"].get(self.variable.name, PlotMeta())
            file_attributes = ATTRIBUTES.get(file_type or "", {})
            plot_meta = file_attributes.get(self.variable.name, fallback)
        if plot_meta.clabel is None:
            plot_meta = plot_meta._replace(clabel=_reformat_units(self.variable.units))
        return plot_meta


class Plot:
    def __init__(self, sub_plot: SubPlot):
        self.sub_plot = sub_plot
        self._data = sub_plot.variable[:]
        self._data_orig = self._data.copy()
        self._plot_meta = sub_plot.plot_meta
        self._is_log = sub_plot.plot_meta.log_scale
        self._ax = sub_plot.ax

    def _convert_units(self) -> str:
        multiply, add = "multiply", "add"
        units_conversion = {
            "rainfall_rate": (multiply, 3600000, "mm h$^{-1}$"),
            "snowfall_rate": (multiply, 3600000, "mm h$^{-1}$"),
            "air_pressure": (multiply, 0.01, "hPa"),
            "relative_humidity": (multiply, 100, "%"),
            "rainfall_amount": (multiply, 1000, "mm"),
            "air_temperature": (add, -273.15, "\u00b0C"),
        }
        conversion_method, conversion, units = units_conversion.get(
            self.sub_plot.variable.name, (multiply, 1, None)
        )
        if conversion_method == multiply:
            self._data *= conversion
            self._data_orig *= conversion
        elif conversion_method == add:
            self._data += conversion
            self._data_orig += conversion
        if units is not None:
            self._plot_meta = self._plot_meta._replace(clabel=units)
            return units
        units = getattr(self.sub_plot.variable, "units", "")
        return _reformat_units(units)

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
                facecolor="whitesmoke",
                edgecolor="lightgrey",
                label="_nolegend_",
                zorder=_get_zorder("data_gap"),
            )

    def _mark_gaps(self, figure_data: FigureData) -> None:
        time = figure_data.time
        data = self._data
        if time[0] < 0 or time[-1] > 24:
            msg = "Time values outside the range 0-24."
            raise ValueError(msg)
        max_gap_fraction_hour = _get_max_gap_in_minutes(figure_data) / 60

        if getattr(figure_data.file, "cloudnet_file_type", "") == "model":
            time, data = screen_completely_masked_profiles(time, data)

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

    def _read_flagged_data(self, figure_data: FigureData) -> ndarray:
        flag_names = [
            f"{self.sub_plot.variable.name}_quality_flag",
            "temperature_quality_flag",
        ]
        if self.sub_plot.variable.name != "irt":
            flag_names.append("quality_flag")
        for flag_name in flag_names:
            if flag_name in figure_data.file.variables:
                return figure_data.file.variables[flag_name][:] > 0
        return np.array([])


class Plot2D(Plot):
    def plot(self, figure_data: FigureData):
        self._convert_units()
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

        if figure_data.is_mwrpy_product():
            self._fill_flagged_data(figure_data)

    def _fill_flagged_data(self, figure_data: FigureData) -> None:
        flags = self._read_flagged_data(figure_data)
        batches = find_batches_of_ones(flags)
        for batch in batches:
            if batch[0] == batch[1]:
                continue
            time_batch = figure_data.time[batch[0]], figure_data.time[batch[1]]
            self._ax.fill_between(
                time_batch,
                *self._get_y_limits(),
                facecolor="white",
                alpha=0.7,
                label="_nolegend_",
                zorder=_get_zorder("flags"),
            )

    def _plot_segment_data(self, figure_data: FigureData) -> None:
        def _hide_segments() -> tuple[list, list]:
            if self._plot_meta.clabel is None:
                msg = "Missing clabel"
                raise ValueError(msg)
            labels = [x[0] for x in self._plot_meta.clabel]
            colors = [x[1] for x in self._plot_meta.clabel]
            segments_to_hide = np.char.startswith(labels, "_")
            indices = np.where(segments_to_hide)[0]
            for ind in np.flip(indices):
                del labels[ind], colors[ind]
                self._data[self._data == ind] = ma.masked
                self._data[self._data > ind] -= 1
            return colors, labels

        cbar, clabel = _hide_segments()
        alt = self._screen_data_by_max_y(figure_data)
        image = self._ax.pcolorfast(
            figure_data.time_including_gaps,
            alt,
            self._data.T[:-1, :-1],
            cmap=ListedColormap(cbar),
            vmin=-0.5,
            vmax=len(cbar) - 0.5,
            zorder=_get_zorder("data"),
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

        if (duration := self._plot_meta.time_smoothing_duration) > 0:
            sigma_units = calc_sigma_units(
                figure_data.time, alt * 1e3, sigma_minutes=duration, sigma_metres=0
            )
            valid_time_ind = ~np.all(self._data.mask, axis=1)
            smoothed_data = uniform_filter(self._data[valid_time_ind, :], sigma_units)
            self._data[valid_time_ind, :] = smoothed_data

        if self._data.mask.all() and figure_data.options.raise_on_empty:
            msg = "All data is masked"
            raise PlottingError(msg)

        pcolor_kwargs = {
            "cmap": plt.get_cmap(str(self._plot_meta.cmap)),
            "vmin": vmin,
            "vmax": vmax,
            "zorder": _get_zorder("data"),
        }
        image: Any
        if getattr(figure_data.file, "cloudnet_file_type", "") == "model":
            image = self._ax.pcolor(
                figure_data.time_including_gaps,
                alt,
                self._data.T,
                **pcolor_kwargs,
                shading="nearest",
            )
        else:
            image = self._ax.pcolorfast(
                figure_data.time_including_gaps,
                alt,
                self._data.T[:-1, :-1],
                **pcolor_kwargs,
            )
        cbar = self._init_colorbar(image)
        cbar.set_label(str(self._plot_meta.clabel), fontsize=13)

        if self._is_log:
            cbar.set_ticks(np.arange(vmin, vmax + 1).tolist())
            tick_labels = get_log_cbar_tick_labels(vmin, vmax)
            cbar.ax.set_yticklabels(tick_labels)

        if self._plot_meta.contour:
            self._plot_contour(
                figure_data,
                alt,
                levels=np.linspace(vmin, vmax, num=10),
                colors="black",
                linewidths=0.5,
            )

        if self.sub_plot.variable.name == "Tw":
            self._plot_contour(
                figure_data,
                alt,
                levels=np.array([T0]),
                colors="gray",
                linewidths=1.25,
                linestyles="dashed",
            )

    def _plot_contour(self, figure_data: FigureData, alt: np.ndarray, **options):
        time_length = len(figure_data.time_including_gaps)
        step = max(1, time_length // 200)
        ind_time = np.arange(0, time_length, step)
        self._ax.contour(
            figure_data.time_including_gaps[ind_time],
            alt,
            self._data[ind_time, :].T,
            **options,
            zorder=_get_zorder("contour"),
        )

    def _screen_data_by_max_y(self, figure_data: FigureData) -> ndarray:
        if figure_data.height is None:
            msg = "No height information in the file."
            raise ValueError(msg)
        if self._data.ndim < 2:
            msg = "Data has to be 2D."
            raise PlottingError(msg)
        alt = figure_data.height
        if figure_data.options.max_y is None:
            return alt
        ind = int((np.argmax(alt > figure_data.options.max_y) or len(alt)) + 1)
        self._data = self._data[:, :ind]
        return alt[:ind]


class Plot1D(Plot):
    def plot(self, figure_data: FigureData, hacky_freq_ind: int | None = None) -> None:
        units = self._convert_units()
        self._mark_gaps(figure_data)
        self._ax.plot(
            figure_data.time_including_gaps,
            self._data,
            label="_nolegend_",
            **self._get_plot_options(),
            zorder=_get_zorder("data"),
        )
        if self._plot_meta.moving_average:
            self._plot_moving_average(figure_data, hacky_freq_ind)
        if self._plot_meta.zero_line:
            self._ax.axhline(0, color="black", alpha=0.5, label="_nolegend_")
        self._fill_between_data_gaps(figure_data)
        self.sub_plot.set_yax(ylabel=units, y_limits=self._get_y_limits())
        pos = self._ax.get_position()
        self._ax.set_position((pos.x0, pos.y0, pos.width * 0.965, pos.height))
        if figure_data.is_mwrpy_product():
            flags = self._read_flagged_data(figure_data)
            if np.any(flags):
                self._plot_flag_data(figure_data.time[flags], self._data_orig[flags])
                self._add_legend()

    def plot_tb(self, figure_data: FigureData, freq_ind: int) -> None:
        if len(self._data.shape) != 2 or freq_ind >= self._data.shape[1]:
            msg = "Frequency index not found in data"
            raise PlottingError(msg)
        self._data = self._data[:, freq_ind]
        self._data[np.isnan(self._data)] = ma.masked
        if self._data.mask.all() and figure_data.options.raise_on_empty:
            msg = "All data is masked"
            raise PlottingError(msg)
        self._data_orig = self._data_orig[:, freq_ind]
        if self.sub_plot.variable.name == "tb":
            is_bad_zenith = self._get_bad_zenith_profiles(figure_data)
            self._data[is_bad_zenith] = ma.masked
            self._data_orig[is_bad_zenith] = ma.masked
            flags = self._read_flagged_data(figure_data)[:, freq_ind]
            flags[is_bad_zenith] = False
            if np.any(flags):
                self._plot_flag_data(figure_data.time[flags], self._data_orig[flags])
                self._add_legend()
        self._show_frequency(figure_data, freq_ind)
        self.plot(figure_data, freq_ind)

    def _show_frequency(self, figure_data: FigureData, freq_ind: int) -> None:
        if self.sub_plot.variable.name == "tb":
            label = "Freq"
            value = figure_data.file.variables["frequency"][freq_ind]
            unit = "GHz"
        elif "ir_wavelength" in figure_data.file.variables:
            label = "WL"
            variable = figure_data.file.variables["ir_wavelength"]
            # `ir_wavelength` is scalar in old files
            value = variable[:] if len(variable.shape) == 0 else variable[freq_ind]
            value /= 1e-6  # m to μm
            unit = "μm"
        else:
            return
        self._ax.text(
            0.0,
            -0.13,
            f"{label}: {value:.2f} {unit}",
            transform=self._ax.transAxes,
            fontsize=12,
            color="dimgrey",
            bbox={
                "facecolor": "white",
                "linewidth": 0,
                "boxstyle": "round",
            },
        )

    def _plot_flag_data(self, time: ndarray, values: ndarray) -> None:
        self._ax.plot(
            time,
            values,
            color="salmon",
            marker=".",
            lw=0,
            markersize=3,
            zorder=_get_zorder("flags"),
        )

    def _add_legend(self) -> None:
        self._ax.legend(
            ["Flagged data"],
            markerscale=3,
            numpoints=1,
            frameon=False,
        )

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

    def _plot_moving_average(
        self, figure_data: FigureData, hacky_freq_ind: int | None = None
    ) -> None:
        time = figure_data.time.copy()
        data = self._data_orig.copy()

        if figure_data.is_mwrpy_product() or self.sub_plot.variable.name in (
            "tb",
            "irt",
        ):
            flags = self._read_flagged_data(figure_data)
        else:
            flags = np.array([])

        if hacky_freq_ind is not None and np.any(flags):
            flags = flags[:, hacky_freq_ind]
        is_invalid = ma.getmaskarray(data)
        if np.any(flags):
            is_invalid |= flags

        is_wind_direction = self.sub_plot.variable.name == "wind_direction"
        if is_wind_direction:
            wind_speed = figure_data.file["wind_speed"]
            data = np.stack([wind_speed, data], axis=1)

        block_ind = np.where(np.diff(is_invalid))[0] + 1
        valid_time_blocks = np.split(time, block_ind)[is_invalid[0] :: 2]
        valid_data_blocks = np.split(data, block_ind)[is_invalid[0] :: 2]

        for time1, data1 in zip(valid_time_blocks, valid_data_blocks, strict=False):
            if is_wind_direction:
                sma = self._calculate_average_wind_direction(
                    data1[:, 0], data1[:, 1], time1, window=15
                )
            else:
                sma = self._calculate_moving_average(data1, time1, window=5)
            gap_time = _get_max_gap_in_minutes(figure_data)
            gaps = self._find_time_gap_indices(time1, max_gap_min=gap_time) + 1

            for time2, data2 in zip(
                np.split(time1, gaps), np.split(sma, gaps), strict=False
            ):
                self._ax.plot(
                    time2,
                    data2,
                    color="slateblue",
                    lw=2,
                    label="_nolegend_",
                    zorder=_get_zorder("mean_curve"),
                )

    @staticmethod
    def _get_line_width(time: ndarray) -> float:
        line_width = np.median(np.diff(time)) * 1000
        return min(max(line_width, 0.25), 0.9)

    @staticmethod
    def _get_bad_zenith_profiles(figure_data: FigureData) -> ndarray:
        zenith_limit = 5
        valid_pointing_status = 0
        if "pointing_flag" in figure_data.file.variables:
            pointing_flag = figure_data.file.variables["pointing_flag"][:]
            zenith_angle = figure_data.file.variables["zenith_angle"][:]
            is_bad_zenith = np.abs(zenith_angle) > zenith_limit
            is_bad_pointing = pointing_flag != valid_pointing_status
            return is_bad_zenith | is_bad_pointing
        return np.zeros_like(figure_data.time, dtype=bool)

    @staticmethod
    def _find_time_gap_indices(time: ndarray, max_gap_min: float) -> ndarray:
        gap_decimal_hour = max_gap_min / 60
        return np.where(np.diff(time) > gap_decimal_hour)[0]

    @staticmethod
    def _calculate_moving_average(
        data: ndarray, time: ndarray, window: float = 5
    ) -> ndarray:
        if len(data) == 0:
            return np.array([])
        if len(data) == 1:
            return data
        time_delta_hours = np.median(np.diff(time))
        window_size = int(window / 60 / time_delta_hours)
        if window_size < 1:
            window_size = 1
        if window_size % 2 == 0:
            window_size += 1
        weights = np.repeat(1 / window_size, window_size)
        padded_data = np.pad(data, window_size // 2, mode="edge")
        return np.convolve(padded_data, weights, "valid")

    @classmethod
    def _calculate_average_wind_direction(
        cls,
        wind_speed: ndarray,
        wind_direction: ndarray,
        time: ndarray,
        window: float = 5,
    ) -> ndarray:
        angle = np.deg2rad(wind_direction)
        u = wind_speed * np.cos(angle)
        v = wind_speed * np.sin(angle)
        avg_u = cls._calculate_moving_average(u, time, window)
        avg_v = cls._calculate_moving_average(v, time, window)
        data = np.rad2deg(np.arctan2(avg_v, avg_u)) % 360
        wrap = np.where(np.abs(np.diff(data)) > 300)[0]
        data[wrap] = np.nan
        return data


def generate_figure(
    filename: os.PathLike | str,
    variables: list[str],
    *,
    show: bool = True,
    output_filename: os.PathLike | str | None = None,
    options: PlotParameters | None = None,
) -> Dimensions:
    """Generate a figure based on the given filename and variables.

    Args:
        filename: The path to the input file.
        variables: A list of variable names to plot.
        show: Whether to display the figure. Defaults to True.
        output_filename: The path to save the figure. Defaults to None.
        options: Additional plot parameters. Defaults to None.

    Returns:
        Dimensions: Dimensions of a generated figure in pixels.

    """
    if options is None:
        options = PlotParameters()

    with netCDF4.Dataset(filename) as file:
        figure_data = FigureData(file, variables, options)
        fig, axes = figure_data.initialize_figure()

        for ax, variable, ind in zip(
            axes, figure_data.variables, figure_data.indices, strict=True
        ):
            file_type = getattr(file, "cloudnet_file_type", None)
            subplot = SubPlot(ax, variable, options, file_type)

            if variable.name in ("tb", "irt") and ind is not None:
                Plot1D(subplot).plot_tb(figure_data, ind)
            elif variable.ndim == 1:
                Plot1D(subplot).plot(figure_data)
            elif variable.name in ("number_concentration", "fall_velocity"):
                Plot2D(subplot).plot(figure_data)
                subplot.set_yax(ylabel="Diameter (mm)", y_limits=(0, 10))
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
        subplot.show_footer(fig, ax)

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
    if unit == "1":
        return ""
    return re.sub(r"(-\d+)", r"$^{\1}$", unit).replace("mu ", "$\\mu$")


def _get_max_gap_in_minutes(figure_data: FigureData) -> float:
    source = getattr(figure_data.file, "source", "").lower()
    file_type = getattr(figure_data.file, "cloudnet_file_type", "")
    max_allowed_gap = {
        "model": 181 if "gdas1" in source or "ecmwf open" in source else 61,
        "mwr-multi": 35,
        "weather-station": 12,
        "doppler-lidar-wind": 75,
    }
    return max_allowed_gap.get(file_type, 10)


def _get_zorder(name: str) -> int:
    zorder = {
        "contour": 2,
        "data_gap": 2,
        "flags": 2,
    }
    return zorder.get(name, -1)


def find_batches_of_ones(array: ndarray) -> list[tuple[int, int]]:
    """Find batches of ones in a binary array."""
    starts = np.where(np.diff(np.hstack(([0], array))) == 1)[0]
    stops = np.where(np.diff(np.hstack((array, [0]))) == -1)[0]
    return list(zip(starts, stops, strict=True))


def screen_completely_masked_profiles(time: ndarray, data: ma.MaskedArray) -> tuple:
    if not ma.is_masked(data):
        return time, data
    good_ind = np.where(np.any(~data.mask, axis=1))[0]
    if len(good_ind) == 0:
        msg = "All values masked in the file."
        raise PlottingError(msg)
    good_ind = np.append(good_ind, good_ind[-1] + 1)
    good_ind = np.clip(good_ind, 0, len(time) - 1)
    return time[good_ind], data[good_ind, :]


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
