"""Misc. plotting routines for Cloudnet products."""
import os.path
from datetime import date, datetime, timezone

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import rcParams
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import Affine2D, Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma, ndarray
from scipy.signal import filtfilt

import cloudnetpy.products.product_tools as ptools
from cloudnetpy import utils
from cloudnetpy.plotting.plot_meta import _MS1, ATTRIBUTES, PlotMeta, Scale
from cloudnetpy.products.product_tools import CategorizeBits

_ZORDER = 42


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


def generate_figure(
    nc_file: str,
    field_names: list,
    *,
    save_path: str | None = None,
    max_y: int = 12,
    dpi: int = 120,
    image_name: str | None = None,
    sub_title: bool = True,
    title: bool = True,
    show: bool = True,
    add_grid: bool = False,
    include_xlimits: bool = False,
    add_sources: bool = False,
    add_serial_number: bool = False,
    add_copyright: bool = False,
    copyright_text: str = "\u00A9 CLOUDNET, cloudnet.fmi.fi",
    add_creation_time: bool = True,
) -> Dimensions:
    """Generates a Cloudnet figure.

    Args:
    ----
        nc_file (str): Input file.
        field_names (list): Variable names to be plotted.
        show (bool, optional): If True, shows the figure. Default is True.
        save_path (str, optional): Setting this path will save the figure (in the
            given path). Default is None, when the figure is not saved.
        max_y (int, optional): Upper limit in the plots (km). Default is 12.
        dpi (int, optional): Figure quality (if saved). Higher value means
            more pixels, i.e., better image quality. Default is 120.
        image_name (str, optional): Name (and full path) of the output image.
            Overrides the *save_path* option. Default is None.
        sub_title (bool, optional): Add subtitle to image. Default is True.
        title (bool, optional): Add title to image. Default is True.
        add_grid (bool, optional): Whether to include a grid in the figure
            to facilitate orientation of the viewer. Designed to be non-intrusive
        include_xlimits (bool, optional): Whether the xticklabels should include
            00:00 and 24:00 for completeness. Default is False.
        add_sources (bool, optional): Add the data source to the image if
            available, otherwise global source. Default is False.
        add_serial_number (bool, optional): Add the serial number for the plot if
            available. If the source attr is available for a field_name
            the source_serial_number attr is checked for the serial number
            (which may not exist, leading to no serial number). If no
            source attr is available, the global source attr and global
            source_serial_number is checked. In either case, the length
            of both source and source_serial_number has to agree.
        add_copyright (bool, optional): Add watermark to image,
            putting the copyright_text in the bottom left. Default is False.
        copyright_text (bool, optional): The text that will be added if
            add_copyright is True. Should be adjusted for any non-ACTRIS/FMI site
            Default is '\u00A9 CLOUDNET, cloudnet.fmi.fi' (copyright symbol).
        add_creation_time (bool, optional): Add the creation time
            after the copyright_text (datetime.datetime.utcnow()

    Returns:
    -------
        Dimensions of the generated figure in pixels.

    Examples:
    --------
        >>> from cloudnetpy.plotting import generate_figure
        >>> generate_figure('categorize_file.nc', ['Z', 'v', 'width', 'ldr',
        'beta', 'lwp'])
        >>> generate_figure('iwc_file.nc', ['iwc', 'iwc_error',
        'iwc_retrieval_status'])
        >>> generate_figure('lwc_file.nc', ['lwc', 'lwc_error',
        'lwc_retrieval_status'], max_y=4)
        >>> generate_figure('classification_file.nc', ['target_classification',
        'detection_status'])
        >>> generate_figure('drizzle_file.nc', ['Do', 'mu', 'S'], max_y=3)
        >>> generate_figure('ier.nc', ['ier', 'ier_error', 'ier_retrieval_status'],
        max_y=3)
        >>> generate_figure('der.nc', ['der', 'der_scaled'], max_y=12)
    """
    indices = [name.split("_")[-1] for name in field_names]
    with netCDF4.Dataset(nc_file) as nc:
        cloudnet_file_type = nc.cloudnet_file_type
    if cloudnet_file_type == "mwr-l1c":
        field_names = [name.split("_")[0] for name in field_names]
    valid_fields, valid_names = _find_valid_fields(nc_file, field_names)
    is_height = _is_height_dimension(nc_file)
    fig, axes = _initialize_figure(len(valid_fields), dpi)

    for ax, field, name, tb_ind in zip(
        axes,
        valid_fields,
        valid_names,
        indices,
        strict=True,
    ):
        original_attrib = None  # monkey patch
        if cloudnet_file_type == "rain-radar" and name == "rainfall_rate":
            original_attrib = ATTRIBUTES[name]
            ATTRIBUTES[name] = PlotMeta(
                name="Rainfall rate",
                cbar="Blues",
                clabel=_MS1,
                plot_range=(0, 50 / 3600000),
                plot_type="mesh",
            )
        plot_type = ATTRIBUTES[name].plot_type

        set_xax(ax, include_xlimits=include_xlimits)

        if title:
            _set_title(ax, name, "")

        if add_grid:
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.grid(which="major", axis="x", color="k", lw=0.2, zorder=0)
            ax.grid(which="minor", axis="x", lw=0.1, color="k", ls=":", zorder=0)
            ax.grid(which="major", axis="y", lw=0.1, color="k", ls=":", zorder=0)

        if add_sources:
            sources = read_source(nc_file, name, add_serial_number=add_serial_number)
            display_datasources(ax, sources)

        # 1D plots
        if (
            not is_height
            or (cloudnet_file_type == "mwr-single" and name in ("lwp", "iwv"))
            or (cloudnet_file_type == "radar" and name == "lwp")
        ):
            unit = _get_variable_unit(nc_file, name)
            source = ATTRIBUTES[name].source
            time = _read_time_vector(nc_file)
            try:
                tb_index = int(tb_ind)
            except ValueError:
                tb_index = None
            _plot_instrument_data(
                ax,
                field,
                name,
                source,
                time,
                unit,
                nc_file,
                tb_index,
            )
            continue
        ax_value = _read_ax_values(nc_file)

        if plot_type not in ("bar", "model"):
            time_new, field_with_gaps = _mark_gaps(ax_value[0], field)
            ax_value = (time_new, ax_value[1])
        else:
            field_with_gaps = field

        field_screened, ax_value = _screen_high_altitudes(
            field_with_gaps,
            ax_value,
            max_y,
        )
        set_yax(ax, max_y, ylabel=None)
        if plot_type == "bar":
            unit = _get_variable_unit(nc_file, name)
            _plot_bar_data(ax, field_screened, ax_value[0], unit)
            set_yax(ax, 2, ATTRIBUTES[name].ylabel)

        elif plot_type == "segment":
            _plot_segment_data(ax, field_screened, name, ax_value)

        else:
            _plot_colormesh_data(ax, field_screened, name, ax_value)
        if original_attrib is not None:
            ATTRIBUTES[name] = original_attrib
    case_date = set_labels(fig, axes[-1], nc_file, sub_title=sub_title)

    if add_copyright:
        display_watermark(fig, copyright_text, add_creation_time)
    handle_saving(image_name, save_path, case_date, valid_names, show=show)
    return Dimensions(fig, axes)


def _mark_gaps(
    time: np.ndarray,
    data: ma.MaskedArray,
    max_allowed_gap: float = 1,
) -> tuple:
    if time[0] < 0:
        msg = "Negative time values in the file."
        raise ValueError(msg)
    if time[-1] > 24:
        msg = "Time values exceed 24 hours."
        raise ValueError(msg)
    max_gap = max_allowed_gap / 60
    if not ma.is_masked(data):
        mask_new = np.zeros(data.shape)
    elif ma.all(data.mask) is ma.masked:
        mask_new = np.ones(data.shape)
    else:
        mask_new = np.copy(data.mask)
    data_new = ma.copy(data)
    time_new = np.copy(time)
    gap_indices = np.where(np.diff(time) > max_gap)[0]
    temp_array = np.zeros((2, data.shape[1]))
    temp_mask = np.ones((2, data.shape[1]))
    time_delta = 0.001
    for ind in np.sort(gap_indices)[::-1]:
        ind_gap = ind + 1
        data_new = np.insert(data_new, ind_gap, temp_array, axis=0)
        mask_new = np.insert(mask_new, ind_gap, temp_mask, axis=0)
        time_new = np.insert(time_new, ind_gap, time[ind_gap] - time_delta)
        time_new = np.insert(time_new, ind_gap, time[ind_gap - 1] + time_delta)
    if (time[0] - 0) > max_gap:
        data_new = np.insert(data_new, 0, temp_array, axis=0)
        mask_new = np.insert(mask_new, 0, temp_mask, axis=0)
        time_new = np.insert(time_new, 0, time[0] - time_delta)
        time_new = np.insert(time_new, 0, time_delta)
    if (24 - time[-1]) > max_gap:
        ind_gap = mask_new.shape[0]
        data_new = np.insert(data_new, ind_gap, temp_array, axis=0)
        mask_new = np.insert(mask_new, ind_gap, temp_mask, axis=0)
        time_new = np.insert(time_new, ind_gap, 24 - time_delta)
        time_new = np.insert(time_new, ind_gap, time[-1] + time_delta)
    data_new.mask = mask_new
    return time_new, data_new


def handle_saving(
    image_name: str | None,
    save_path: str | None,
    case_date: date,
    field_names: list,
    fix: str = "",
    *,
    show: bool = False,
) -> None:
    if image_name:
        plt.savefig(image_name, bbox_inches="tight")
    elif save_path:
        file_name = _create_save_name(save_path, case_date, field_names, fix)
        plt.savefig(file_name, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _get_relative_error(fields: list, ax_values: list, max_y: int) -> tuple:
    x, y = ax_values[0]
    x_new, y_new = ax_values[1]
    old_data_interp = utils.interpolate_2d_mask(x, y, fields[0], x_new, y_new)
    error = utils.calc_relative_error(old_data_interp, fields[1])
    return _screen_high_altitudes(error, ax_values[1], max_y)


def set_labels(fig, ax, nc_file: str, *, sub_title: bool = True) -> date:
    ax.set_xlabel("Time (UTC)", fontsize=13)
    case_date = read_date(nc_file)
    site_name = read_location(nc_file)
    if sub_title:
        add_subtitle(fig, case_date, site_name)
    return case_date


def display_watermark(
    fig,
    copyright_text,
    add_creation_time,
    ypos: float = -0.05,
    fontsize: int = 7,
) -> None:
    if add_creation_time:
        now = datetime.now(tz=timezone.utc).isoformat().split(".")[0].split("T")
        copyright_text += " / Created on " + " ".join(now) + " UTC"
    # similar to add_subtitle
    fig.text(
        0.05,
        ypos + len(fig.get_axes()) / 50,
        copyright_text,
        fontsize=fontsize,
        ha="left",
        va="bottom",
    )


def display_datasources(
    ax,
    source: str,
    xpos: float = 0.01,
    ypos: float = 0.99,
    fontsize: int = 7,
    **kwargs,
) -> None:
    _ = "s" if "\n" in source else ""
    ax.text(
        xpos,
        ypos,
        "Instrument" + _ + ":\n" + source,
        ha="left",
        va="top",
        fontsize=fontsize,
        transform=ax.transAxes,
        **kwargs,
    )


def _set_title(ax, field_name: str, identifier: str = " from CloudnetPy") -> None:
    ax.set_title(f"{ATTRIBUTES[field_name].name}{identifier}", fontsize=14)


def _find_valid_fields(nc_file: str, names: list) -> tuple[list, list]:
    """Returns valid field names and corresponding data."""
    valid_names, valid_data = names[:], []
    try:
        bits = CategorizeBits(nc_file)
    except KeyError:
        bits = None
    with netCDF4.Dataset(nc_file) as nc:
        for name in names:
            if name in nc.variables:
                valid_data.append(nc.variables[name][:])
            elif bits and name in CategorizeBits.category_keys:
                valid_data.append(bits.category_bits[name])
            elif bits and name in CategorizeBits.quality_keys:
                valid_data.append(bits.quality_bits[name])
            else:
                valid_names.remove(name)
    if not valid_names:
        msg = "No valid fields to be plotted."
        raise ValueError(msg)
    return valid_data, valid_names


def _is_height_dimension(full_path: str) -> bool:
    with netCDF4.Dataset(full_path) as nc:
        return any(key in nc.variables for key in ("height", "range"))


def _get_variable_unit(full_path: str, name: str) -> str:
    with netCDF4.Dataset(full_path) as nc:
        return nc.variables[name].units


def _initialize_figure(n_subplots: int, dpi) -> tuple:
    """Creates an empty figure according to the number of subplots."""
    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=(16, 4 + (n_subplots - 1) * 4.8),
        dpi=dpi,
    )
    fig.subplots_adjust(left=0.06, right=0.73)
    if n_subplots == 1:
        axes = [axes]
    return fig, axes


def _read_ax_values(full_path: str) -> tuple[ndarray, ndarray]:
    """Returns time and height arrays."""
    file_type = utils.get_file_type(full_path)
    with netCDF4.Dataset(full_path) as nc:
        is_height = "height" in nc.variables
    fields = ["time", "range"] if is_height is not True else ["time", "height"]
    time, height = ptools.read_nc_fields(full_path, fields)
    if file_type == "model":
        height = ma.mean(height, axis=0)
    height_km = height / 1000
    return time, height_km


def _read_time_vector(nc_file: str) -> ndarray:
    """Converts time vector to fraction hour."""
    with netCDF4.Dataset(nc_file) as nc:
        time = nc.variables["time"][:]
    if max(time) < 24:
        return time
    return utils.seconds2hours(time)


def _screen_high_altitudes(data_field: ndarray, ax_values: tuple, max_y: int) -> tuple:
    """Removes altitudes from 2D data that are not visible in the figure.

    Bug in pcolorfast causing effect to axis not noticing limitation while
    saving fig. This fixes that bug till pcolorfast does fixing themselves.

    Args:
    ----
        data_field (ndarray): 2D data array.
        ax_values (tuple): Time and height 1D arrays.
        max_y (int): Upper limit in the plots (km).

    """
    alt = ax_values[-1]
    if data_field.ndim > 1:
        ind = int((np.argmax(alt > max_y) or len(alt)) + 1)
        data_field = data_field[:, :ind]
        alt = alt[:ind]
    return data_field, (ax_values[0], alt)


def set_xax(ax, *, include_xlimits: bool = False) -> None:
    """Sets xticks and xtick labels for plt.imshow()."""
    ticks_x_labels = _get_standard_time_ticks(include_xlimits=include_xlimits)
    ax.set_xticks(np.arange(0, 25, 4, dtype=int))
    ax.set_xticklabels(ticks_x_labels, fontsize=12)
    ax.set_xlim(0, 24)


def set_yax(ax, max_y: float, ylabel: str | None, min_y: float = 0.0) -> None:
    """Sets yticks, ylim and ylabel for yaxis of axis."""
    ax.set_ylim(min_y, max_y)
    ax.set_ylabel("Height (km)", fontsize=13)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13)


def _get_standard_time_ticks(
    resolution: int = 4,
    *,
    include_xlimits: bool = False,
) -> list:
    """Returns typical ticks / labels for a time vector between 0-24h."""
    if include_xlimits:
        return [
            f"{int(i):02d}:00" if 24 >= i >= 0 else ""
            for i in np.arange(0, 24.01, resolution)
        ]
    return [
        f"{int(i):02d}:00" if 24 > i > 0 else ""
        for i in np.arange(0, 24.01, resolution)
    ]


def _plot_bar_data(ax, data: np.ndarray, time: ndarray, unit: str) -> None:
    """Plots 1D variable as bar plot.

    Args:
    ----
        ax (obj): Axes object.
        data (maskedArray): 1D data array.
        time (ndarray): 1D time array.

    """
    data = _convert_to_kg(data, unit)
    ax.plot(time, data, color="navy", zorder=_ZORDER)
    data_filled = data.filled(0) if isinstance(data, ma.MaskedArray) else data

    ax.bar(
        time,
        data_filled,
        width=1 / 120,
        align="center",
        alpha=0.5,
        color="royalblue",
        zorder=_ZORDER,
    )
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])


def _plot_segment_data(ax, data: ma.MaskedArray, name: str, axes: tuple) -> None:
    """Plots categorical 2D variable.

    Args:
    ----
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.

    """

    def _hide_segments(
        data_in: ma.MaskedArray,
    ) -> tuple[ma.MaskedArray, list, list]:
        if variables.clabel is None:
            msg = f"Labels not defined for {name}."
            raise ValueError(msg)
        labels = [x[0] for x in variables.clabel]
        colors = [x[1] for x in variables.clabel]
        segments_to_hide = np.char.startswith(labels, "_")
        indices = np.where(segments_to_hide)[0]
        for ind in np.flip(indices):
            del labels[ind], colors[ind]
            data_in[data_in == ind] = ma.masked
            data_in[data_in > ind] -= 1
        return data_in, colors, labels

    variables = ATTRIBUTES[name]
    original_mask = np.copy(data.mask)
    data, cbar, clabel = _hide_segments(data)
    cmap = ListedColormap(cbar)
    data[original_mask] = 99
    pl = ax.pcolorfast(
        *axes,
        data[:-1, :-1].T,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(cbar) - 0.5,
        zorder=_ZORDER,
    )
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_ticks(np.arange(len(clabel)).tolist())
    colorbar.ax.set_yticklabels(clabel, fontsize=13)


def _plot_colormesh_data(ax, data: ndarray, name: str, axes: tuple) -> None:
    """Plots continuous 2D variable.

    Creates only one plot, so can be used both one plot and subplot type of figs.

    Args:
    ----
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.
    """
    variables = ATTRIBUTES[name]
    if variables.plot_range is None:
        msg = f"Plot range not defined for {name}."
        raise ValueError(msg)

    if name == "cloud_fraction":
        data[data < 0.1] = ma.masked

    if variables.plot_type == "bit":
        color_map: Colormap = ListedColormap(str(variables.cbar))
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])
    else:
        color_map = plt.get_cmap(str(variables.cbar), 22)

    vmin, vmax = variables.plot_range

    if variables.plot_scale == Scale.LOGARITHMIC:
        data, vmin, vmax = lin2log(data, vmin, vmax)

    pl = ax.pcolorfast(
        *axes,
        data[:-1, :-1].T,
        vmin=vmin,
        vmax=vmax,
        cmap=color_map,
        zorder=_ZORDER,
    )

    if variables.plot_type != "bit":
        colorbar = _init_colorbar(pl, ax)
        colorbar.set_label(str(variables.clabel), fontsize=13)

    if variables.plot_scale == Scale.LOGARITHMIC:
        tick_labels = generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax + 1).tolist())
        colorbar.ax.set_yticklabels(tick_labels)


def _plot_instrument_data(
    ax,
    data: ma.MaskedArray,
    name: str,
    product: str | None,
    time: ndarray,
    unit: str,
    full_path: str | None = None,
    tb_ind: int | None = None,
) -> None:
    if product in ("mwr", "mwr-single"):
        _plot_mwr(ax, data, name, time, unit)
    if product == "disdrometer":
        _plot_disdrometer(ax, data, time, name, unit)
    if product == "weather-station":
        _plot_weather_station(ax, data, time, name)
    if full_path is not None and tb_ind is not None:
        quality_flag_array = ptools.read_nc_fields(full_path, "quality_flag")
        quality_flag_array_ma = ma.array(quality_flag_array)
        quality_flag = quality_flag_array_ma[:, tb_ind]
        data = data[:, tb_ind]
        data_dict = {"tb": data, "quality_flag": quality_flag, "time": time}
        _plot_hatpro(ax, data_dict, full_path)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])


def _plot_disdrometer(ax, data: ndarray, time: ndarray, name: str, unit: str) -> None:
    if name == "rainfall_rate":
        if unit == "m s-1":
            data *= 1000 * 3600
        ax.plot(time, data, color="royalblue", zorder=_ZORDER)
        ylim = max((np.max(data) * 1.05, 0.1))
        set_yax(ax, ylim, "mm h$^{-1}$")
    if name == "n_particles":
        ax.plot(time, data, color="royalblue", zorder=_ZORDER)
        ylim = max((np.max(data) * 1.05, 1))
        set_yax(ax, ylim, "")


def _plot_hatpro(ax, data: dict, full_path: str) -> None:
    tb = _pointing_filter(full_path, data["tb"])
    ax.plot(
        data["time"],
        tb,
        color="royalblue",
        linestyle="-",
        linewidth=1,
        zorder=_ZORDER,
    )
    set_yax(
        ax,
        max_y=np.max(tb) + 0.5,
        min_y=np.min(tb) - 0.5,
        ylabel="Brightness temperature [K]",
    )


def _pointing_filter(
    full_path: str,
    data: ndarray,
    zenith_limit=5,
    status: int = 0,
) -> ndarray:
    """Filters data according to pointing flag and zenith angle."""
    with netCDF4.Dataset(full_path) as nc:
        if "pointing_flag" in nc.variables:
            pointing = ptools.read_nc_fields(full_path, "pointing_flag")
            zenith_angle = ptools.read_nc_fields(full_path, "zenith_angle")
            if data.ndim > 1:
                data[np.abs(zenith_angle) > zenith_limit, :] = ma.masked
                data[pointing != status, :] = ma.masked
            else:
                data[np.abs(zenith_angle) > zenith_limit] = ma.masked
                data[pointing != status] = ma.masked
    return data


def _plot_weather_station(ax, data: ndarray, time: ndarray, name: str) -> None:
    match name:
        case "air_temperature":
            unit = "K"
            min_y = np.min(data) - 1
            max_y = np.max(data) + 1
            ax.plot(time, data, color="royalblue", zorder=_ZORDER)
            set_yax(ax, min_y=min_y, max_y=max_y, ylabel=unit)
        case "wind_speed":
            unit = "m s$^{-1}$"
            min_y = np.min(data) - 1
            max_y = np.max(data) + 1
            ax.plot(time, data, color="royalblue", zorder=_ZORDER)
            set_yax(ax, min_y=min_y, max_y=max_y, ylabel=unit)
        case "wind_direction":
            unit = "degree"
            ax.plot(
                time,
                data,
                color="royalblue",
                marker=".",
                linewidth=0,
                markersize=3,
                zorder=_ZORDER,
            )
            set_yax(ax, min_y=0, max_y=360, ylabel=unit)
        case "relative_humidity":
            data *= 100
            unit = "%"
            min_y = np.min(data) - 1
            max_y = np.max(data) + 1
            ax.plot(time, data, color="royalblue", zorder=_ZORDER)
            set_yax(ax, min_y=min_y, max_y=max_y, ylabel=unit)
        case "air_pressure":
            data /= 100
            unit = "hPa"
            min_y = np.min(data) - 1
            max_y = np.max(data) + 1
            ax.plot(time, data, color="royalblue", zorder=_ZORDER)
            set_yax(ax, min_y=min_y, max_y=max_y, ylabel=unit)
        case "rainfall_amount":
            data *= 1000
            unit = "mm"
            min_y = 0
            max_y = np.max(data) + 1
            ax.plot(time, data, color="royalblue", zorder=_ZORDER)
            set_yax(ax, min_y=min_y, max_y=max_y, ylabel=unit)
        case unknown:
            msg = f"Not implemented for {unknown}"
            raise NotImplementedError(msg)


def _plot_mwr(ax, data_in: ma.MaskedArray, name: str, time: ndarray, unit: str) -> None:
    data, time = _get_unmasked_values(data_in, time)
    data = _convert_to_kg(data, unit)
    rolling_mean, width = _calculate_rolling_mean(time, data)
    gaps = _find_time_gap_indices(time)
    n, line_width = _get_plot_parameters(data)
    data_filtered = _filter_noise(data, n)
    time[gaps] = np.nan
    ax.plot(time, data_filtered, color="royalblue", lw=line_width, zorder=_ZORDER)
    ax.axhline(linewidth=0.8, color="k")
    ax.plot(
        time[int(width / 2 - 1) : int(-width / 2)],
        rolling_mean,
        color="sienna",
        linewidth=2.0,
        zorder=_ZORDER,
    )
    ax.plot(
        time[int(width / 2 - 1) : int(-width / 2)],
        rolling_mean,
        color="wheat",
        linewidth=0.6,
        zorder=_ZORDER,
    )
    set_yax(
        ax,
        round(np.max(data), 3) + 0.0005,
        ATTRIBUTES[name].ylabel,
        min_y=round(np.min(data), 3) - 0.0005,
    )


def _get_unmasked_values(
    data: ma.MaskedArray,
    time: ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if ma.is_masked(data) is False:
        return data, time
    good_values = ~data.mask
    return data[good_values], time[good_values]


def _convert_to_kg(data: np.ndarray, unit: str) -> np.ndarray:
    if "kg" in unit:
        return data
    return data / 1000


def _find_time_gap_indices(time: ndarray) -> ndarray:
    """Finds time gaps bigger than 5min."""
    time_diff = np.diff(time)
    dec_hour_5min = 0.085
    return np.where(time_diff > dec_hour_5min)[0]


def _get_plot_parameters(data: ndarray) -> tuple[int, float]:
    length = len(data)
    n = np.rint(np.nextafter((length / 10000), (length / 10000) + 1))
    if length < 10000:
        line_width = 0.9
    elif 10000 <= length < 38000:
        line_width = 0.7
    elif 38000 <= length < 55000:
        line_width = 0.3
    else:
        line_width = 0.25
    return int(n), line_width


def _calculate_rolling_mean(time: ndarray, data: ndarray) -> tuple[ndarray, int]:
    width = len(time[time <= time[0] + 0.3])
    if (width % 2) != 0:
        width = width + 1
    rolling_window = np.blackman(width)
    rolling_mean = np.convolve(data, rolling_window, "valid")
    rolling_mean = rolling_mean / np.sum(rolling_window)
    return rolling_mean, width


def _filter_noise(data: ndarray, n: int) -> ndarray:
    """IIR filter"""
    if n <= 1:
        n = 2
    b = [1.0 / n] * n
    a = 1
    return filtfilt(b, a, data)


def _init_colorbar(plot, axis) -> Colorbar:
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def generate_log_cbar_ticklabel_list(vmin: float, vmax: float) -> list:
    """Create list of log format colorbar label ticks as string"""
    return [f"10$^{{{int(i)}}}$" for i in np.arange(vmin, vmax + 1)]


def read_location(nc_file: str) -> str:
    """Returns site name."""
    with netCDF4.Dataset(nc_file) as nc:
        return nc.location


def read_date(nc_file: str) -> date:
    """Returns measurement date."""
    with netCDF4.Dataset(nc_file) as nc:
        return date(int(nc.year), int(nc.month), int(nc.day))


def read_source(nc_file: str, name: str, *, add_serial_number: bool = True) -> str:
    """Returns source attr of field name or global one and maybe serial number ."""
    with netCDF4.Dataset(nc_file) as nc:
        if name in nc.variables and "source" in nc.variables[name].ncattrs():
            # single device has available src attr and maybe SN
            source = nc.variables[name].source
            # even if the attr is source_serial_number, it is possible that
            # the variable comes from more than one device, e.g. Do for drizzle
            # for which we need to account
            if (
                add_serial_number
                and "source_serial_number" in nc.variables[name].ncattrs()
            ):
                sno = nc.variables[name].source_serial_number
                source, sno = source.split("\n"), sno.split("\n")
                source = [
                    f"{_source} (SN: {_sno})" if _sno else f"{_source}"
                    for _source, _sno in zip(source, sno, strict=True)
                ]
                source = "\n".join(source)
        else:
            # global src, a \n sep string-list
            source = nc.source if "source" in nc.ncattrs() else []
            # empty list means that the zip below runs for 0 times as
            # the assumption is if we do not have any sources we can't
            # have any serial numbers, i.e. no instrument type means
            # to instrument serial number. If this is the case
            # something somewhere else is wrong and should not be
            # fixed here.
            # who knows whether the cloudnet nc file actually has the SNs
            # so better check beforehand
            if add_serial_number and "source_serial_numbers" in nc.ncattrs():
                sno = nc.source_serial_numbers.split("\n")
                sno = [i if i else "" for i in sno]
                if source:
                    source = source.split("\n")
                source = [
                    f"{_source} (SN: {_sno})" if _sno else f"{_source}"
                    for _source, _sno in zip(source, sno, strict=True)
                ]
                source = "\n".join(source)
    return source.rstrip("\n")


def add_subtitle(fig, case_date: date, site_name: str) -> None:
    """Adds subtitle into figure."""
    text = _get_subtitle_text(case_date, site_name)
    fig.suptitle(
        text,
        fontsize=13,
        y=0.885,
        x=0.07,
        horizontalalignment="left",
        verticalalignment="bottom",
    )


def _get_subtitle_text(case_date: date, site_name: str) -> str:
    site_name = site_name.replace("-", " ")
    return f"{site_name}, {case_date.strftime('%d %b %Y').lstrip('0')}"


def _create_save_name(
    save_path: str,
    case_date: date,
    field_names: list,
    fix: str = "",
) -> str:
    """Creates file name for saved images."""
    date_string = case_date.strftime("%Y%m%d")
    return f"{save_path}{date_string}_{'_'.join(field_names)}{fix}.png"


def _plot_relative_error(ax, error: ma.MaskedArray, ax_values: tuple) -> None:
    pl = ax.pcolorfast(
        *ax_values,
        error[:-1, :-1].T,
        cmap="RdBu",
        vmin=-30,
        vmax=30,
        zorder=_ZORDER,
    )
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_label("%", fontsize=13)
    median_error = ma.median(error.compressed())
    median_error = np.round(median_error, 3)
    ax.set_title(f"Median relative error: {median_error} %", fontsize=14)


def lin2log(*args) -> list:
    return [ma.log10(x) for x in args]


# Misc plotting routines:


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


def compare_files(
    nc_files: tuple[str, str],
    field_name: str,
    save_path: str | None = None,
    max_y: int = 12,
    dpi: int = 120,
    image_name: str | None = None,
    *,
    show: bool = True,
    relative_err: bool = False,
) -> Dimensions:
    """Plots one particular field from two Cloudnet files.

    Args:
    ----
        nc_files (tuple): Filenames of the two files to be compared.
        field_name (str): Name of variable to be plotted.
        show (bool, optional): If True, shows the plot.
        relative_err (bool, optional): If True, plots also relative error. Makes
            sense only for continuous variables. Default is False.
        save_path (str, optional): If defined, saves the image to this path.
            Default is None.
        max_y (int, optional): Upper limit of images (km). Default is 12.
        dpi (int, optional): Quality of plots. Default is 120.
        image_name (str, optional): Name (and full path) of the output image.
            Overrides the *save_path* option. Default is None.

    Returns:
    -------
        Dimensions of the generated figure in pixels.

    """
    plot_type = ATTRIBUTES[field_name].plot_type
    fields = [_find_valid_fields(file, [field_name])[0][0] for file in nc_files]
    nc = netCDF4.Dataset(nc_files[0])
    nc.close()
    ax_values = [_read_ax_values(nc_file) for nc_file in nc_files]
    subtitle = (
        f" - {os.path.basename(nc_files[0])}",
        f" - {os.path.basename(nc_files[1])}",
    )
    n_subs = 3 if relative_err is True else 2
    fig, axes = _initialize_figure(n_subs, dpi)

    for ii, ax in enumerate(axes[:2]):
        field, ax_value = _screen_high_altitudes(fields[ii], ax_values[ii], max_y)
        set_yax(ax, max_y, ylabel=None)
        _set_title(ax, field_name, subtitle[ii])

        if plot_type == "model":
            _plot_colormesh_data(ax, field, field_name, ax_value)
        elif plot_type == "bar":
            unit = _get_variable_unit(nc_files[ii], field_name)
            _plot_bar_data(ax, field, ax_value[0], unit)
            set_yax(ax, 2, ATTRIBUTES[field_name].ylabel)
        elif plot_type == "segment":
            _plot_segment_data(ax, field, field_name, ax_value)
        else:
            _plot_colormesh_data(ax, field, field_name, ax_value)
            if relative_err is True and ii == 1:
                set_yax(axes[-1], max_y, ylabel=None)
                error, ax_value = _get_relative_error(fields, ax_values, max_y)
                _plot_relative_error(axes[-1], error, ax_value)

    case_date = set_labels(fig, axes[-1], nc_files[0], sub_title=False)
    handle_saving(
        image_name,
        save_path,
        case_date,
        [field_name],
        "_comparison",
        show=show,
    )
    return Dimensions(fig, axes)
