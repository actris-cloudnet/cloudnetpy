"""Misc. plotting routines for Cloudnet products."""
import os.path
from datetime import date
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Affine2D, Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma, ndarray
from scipy.signal import filtfilt

import cloudnetpy.products.product_tools as ptools
from cloudnetpy import utils
from cloudnetpy.plotting.plot_meta import ATTRIBUTES, Scale
from cloudnetpy.products.product_tools import CategorizeBits


class Dimensions:
    """Dimensions of a generated figure in pixels."""

    width: int
    height: int
    margin_top: int
    margin_right: int
    margin_bottom: int
    margin_left: int

    def __init__(self, fig, axes, pad_inches: Optional[float] = None):
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
            .extents.round()
        )
        self.margin_top = int(self.height - y1)
        self.margin_right = int(self.width - x1 - 1)
        self.margin_bottom = int(y0 - 1)
        self.margin_left = int(x0)


def generate_figure(
    nc_file: str,
    field_names: list,
    show: bool = True,
    save_path: str = None,
    max_y: int = 12,
    dpi: int = 120,
    image_name: Optional[str] = None,
    sub_title: bool = True,
    title: bool = True,
) -> Dimensions:
    """Generates a Cloudnet figure.

    Args:
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

    Returns:
        Dimensions of the generated figure in pixels.

    Examples:
        >>> from cloudnetpy.plotting import generate_figure
        >>> generate_figure('categorize_file.nc', ['Z', 'v', 'width', 'ldr', 'beta', 'lwp'])
        >>> generate_figure('iwc_file.nc', ['iwc', 'iwc_error', 'iwc_retrieval_status'])
        >>> generate_figure('lwc_file.nc', ['lwc', 'lwc_error', 'lwc_retrieval_status'], max_y=4)
        >>> generate_figure('classification_file.nc', ['target_classification', 'detection_status'])
        >>> generate_figure('drizzle_file.nc', ['Do', 'mu', 'S'], max_y=3)
        >>> generate_figure('ier.nc', ['ier', 'ier_error', 'ier_retrieval_status'], max_y=3)
        >>> generate_figure('der.nc', ['der', 'der_scaled'], max_y=12)
    """
    valid_fields, valid_names = _find_valid_fields(nc_file, field_names)
    is_height = _is_height_dimension(nc_file)
    fig, axes = _initialize_figure(len(valid_fields), dpi)

    for ax, field, name in zip(axes, valid_fields, valid_names):
        plot_type = ATTRIBUTES[name].plot_type
        if title:
            _set_title(ax, name, "")
        if not is_height:
            unit = _get_variable_unit(nc_file, name)
            source = ATTRIBUTES[name].source
            time = _read_time_vector(nc_file)
            _plot_instrument_data(ax, field, name, source, time, unit)
            continue
        ax_value = _read_ax_values(nc_file)

        if plot_type not in ("bar", "model"):
            time_new, field = _mark_gaps(ax_value[0], field)
            ax_value = (time_new, ax_value[1])

        field, ax_value = _screen_high_altitudes(field, ax_value, max_y)
        _set_ax(ax, max_y)
        if plot_type == "bar":
            _plot_bar_data(ax, field, ax_value[0])
            _set_ax(ax, 2, ATTRIBUTES[name].ylabel)

        elif plot_type == "segment":
            _plot_segment_data(ax, field, name, ax_value)

        else:
            _plot_colormesh_data(ax, field, name, ax_value)
    case_date = _set_labels(fig, axes[-1], nc_file, sub_title)
    _handle_saving(image_name, save_path, show, case_date, valid_names)
    return Dimensions(fig, axes)


def _mark_gaps(time: np.ndarray, data: ma.MaskedArray, max_allowed_gap: float = 1) -> tuple:
    assert time[0] >= 0
    assert time[-1] <= 24
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
        ind += 1
        data_new = np.insert(data_new, ind, temp_array, axis=0)
        mask_new = np.insert(mask_new, ind, temp_mask, axis=0)
        time_new = np.insert(time_new, ind, time[ind] - time_delta)
        time_new = np.insert(time_new, ind, time[ind - 1] + time_delta)
    if (time[0] - 0) > max_gap:
        data_new = np.insert(data_new, 0, temp_array, axis=0)
        mask_new = np.insert(mask_new, 0, temp_mask, axis=0)
        time_new = np.insert(time_new, 0, time[0] - time_delta)
        time_new = np.insert(time_new, 0, time_delta)
    if (24 - time[-1]) > max_gap:
        ind = mask_new.shape[0]
        data_new = np.insert(data_new, ind, temp_array, axis=0)
        mask_new = np.insert(mask_new, ind, temp_mask, axis=0)
        time_new = np.insert(time_new, ind, 24 - time_delta)
        time_new = np.insert(time_new, ind, time[-1] + time_delta)
    data_new.mask = mask_new
    return time_new, data_new


def _handle_saving(
    image_name: Optional[str],
    save_path: Optional[str],
    show: bool,
    case_date: date,
    field_names: list,
    fix: str = "",
):
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


def _set_labels(fig, ax, nc_file: str, sub_title: bool = True) -> date:
    ax.set_xlabel("Time (UTC)", fontsize=13)
    case_date = _read_date(nc_file)
    site_name = _read_location(nc_file)
    if sub_title:
        _add_subtitle(fig, case_date, site_name)
    return case_date


def _set_title(ax, field_name: str, identifier: str = " from CloudnetPy"):
    ax.set_title(f"{ATTRIBUTES[field_name].name}{identifier}", fontsize=14)


def _find_valid_fields(nc_file: str, names: list) -> Tuple[list, list]:
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
        raise ValueError("No fields to be plotted")
    return valid_data, valid_names


def _is_height_dimension(full_path: str) -> bool:
    with netCDF4.Dataset(full_path) as nc:
        is_height = any(key in nc.variables for key in ("height", "range"))
    return is_height


def _get_variable_unit(full_path: str, name: str) -> str:
    with netCDF4.Dataset(full_path) as nc:
        var = nc.variables[name]
        unit = var.units
    return unit


def _initialize_figure(n_subplots: int, dpi) -> tuple:
    """Creates an empty figure according to the number of subplots."""
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 4 + (n_subplots - 1) * 4.8), dpi=dpi)
    fig.subplots_adjust(left=0.06, right=0.73)
    if n_subplots == 1:
        axes = [axes]
    return fig, axes


def _read_ax_values(full_path: str) -> Tuple[ndarray, ndarray]:
    """Returns time and height arrays."""
    file_type = utils.get_file_type(full_path)
    with netCDF4.Dataset(full_path) as nc:
        is_height = "height" in nc.variables
    if is_height is not True:
        fields = ["time", "range"]
    else:
        fields = ["time", "height"]
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


def _set_ax(ax, max_y: float, ylabel: str = None, min_y: float = 0.0):
    """Sets ticks and tick labels for plt.imshow()."""
    ticks_x_labels = _get_standard_time_ticks()
    ax.set_ylim(min_y, max_y)
    ax.set_xticks(np.arange(0, 25, 4, dtype=int))
    ax.set_xticklabels(ticks_x_labels, fontsize=12)
    ax.set_ylabel("Height (km)", fontsize=13)
    ax.set_xlim(0, 24)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13)


def _get_standard_time_ticks(resolution: int = 4) -> list:
    """Returns typical ticks / labels for a time vector between 0-24h."""
    return [f"{int(i):02d}:00" if 24 > i > 0 else "" for i in np.arange(0, 24.01, resolution)]


def _plot_bar_data(ax, data: ma.MaskedArray, time: ndarray):
    """Plots 1D variable as bar plot.

    Args:
        ax (obj): Axes object.
        data (maskedArray): 1D data array.
        time (ndarray): 1D time array.

    """
    # TODO: unit change somewhere else
    ax.plot(time, data / 1000, color="navy")
    ax.bar(time, data.filled(0) / 1000, width=1 / 120, align="center", alpha=0.5, color="royalblue")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])


def _plot_segment_data(ax, data: ma.MaskedArray, name: str, axes: tuple):
    """Plots categorical 2D variable.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.

    """

    def _hide_segments(data_in: ma.MaskedArray) -> Tuple[ma.MaskedArray, list, list]:
        assert variables.clabel is not None
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
    pl = ax.pcolorfast(*axes, data[:-1, :-1].T, cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_ticks(np.arange(len(clabel)))
    colorbar.ax.set_yticklabels(clabel, fontsize=13)


def _plot_colormesh_data(ax, data: ndarray, name: str, axes: tuple):
    """Plots continuous 2D variable.

    Creates only one plot, so can be used both one plot and subplot type of figs.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.
    """
    variables = ATTRIBUTES[name]
    assert variables.plot_range is not None

    if name == "cloud_fraction":
        data[data < 0.1] = ma.masked

    if variables.plot_type == "bit":
        cmap = ListedColormap(variables.cbar)
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])
    else:
        cmap = plt.get_cmap(variables.cbar, 22)

    vmin, vmax = variables.plot_range

    if variables.plot_scale == Scale.LOGARITHMIC:
        data, vmin, vmax = _lin2log(data, vmin, vmax)

    pl = ax.pcolorfast(*axes, data[:-1, :-1].T, vmin=vmin, vmax=vmax, cmap=cmap)

    if variables.plot_type != "bit":
        colorbar = _init_colorbar(pl, ax)
        colorbar.set_label(variables.clabel, fontsize=13)

    if variables.plot_scale == Scale.LOGARITHMIC:
        tick_labels = _generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax + 1))
        colorbar.ax.set_yticklabels(tick_labels)


def _plot_instrument_data(
    ax, data: ma.MaskedArray, name: str, product: Optional[str], time: ndarray, unit: str
):
    if product == "mwr":
        _plot_mwr(ax, data, name, time, unit)
    if product == "disdrometer":
        _plot_disdrometer(ax, data, time, name, unit)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])


def _plot_disdrometer(ax, data: ndarray, time: ndarray, name: str, unit: str):
    if name == "rainfall_rate":
        if unit == "m s-1":
            data *= 1000 * 3600
        ax.plot(time, data, color="royalblue")
        ylim = max((np.max(data) * 1.05, 0.1))
        _set_ax(ax, ylim, "mm h-1")
    if name == "n_particles":
        ax.plot(time, data, color="royalblue")
        ylim = max((np.max(data) * 1.05, 1))
        _set_ax(ax, ylim, "")


def _plot_mwr(ax, data_in: ma.MaskedArray, name: str, time: ndarray, unit: str):
    data, time = _get_unmasked_values(data_in, time)
    data = _g_to_kg(data, unit)
    rolling_mean, width = _calculate_rolling_mean(time, data)
    gaps = _find_time_gap_indices(time)
    n, line_width = _get_plot_parameters(data)
    data_filtered = _filter_noise(data, n)
    time[gaps] = np.nan
    ax.plot(time, data_filtered, color="royalblue", lw=line_width)
    ax.axhline(linewidth=0.8, color="k")
    ax.plot(time[int(width / 2 - 1) : int(-width / 2)], rolling_mean, color="sienna", linewidth=2.0)
    ax.plot(time[int(width / 2 - 1) : int(-width / 2)], rolling_mean, color="wheat", linewidth=0.6)
    _set_ax(
        ax,
        round(np.max(data), 3) + 0.0005,
        ATTRIBUTES[name].ylabel,
        min_y=round(np.min(data), 3) - 0.0005,
    )


def _get_unmasked_values(data: ma.MaskedArray, time: ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if ma.is_masked(data) is False:
        return data, time
    good_values = ~data.mask
    return data[good_values], time[good_values]


def _g_to_kg(data: np.ndarray, unit: str) -> np.ndarray:
    if "kg" in unit:
        return data
    return data / 1000


def _find_time_gap_indices(time: ndarray) -> ndarray:
    """Finds time gaps bigger than 5min."""
    time_diff = np.diff(time)
    dec_hour_5min = 0.085
    gaps = np.where(time_diff > dec_hour_5min)[0]
    return gaps


def _get_plot_parameters(data: ndarray) -> Tuple[int, float]:
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


def _calculate_rolling_mean(time: ndarray, data: ndarray) -> Tuple[ndarray, int]:
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


def _init_colorbar(plot, axis):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def _generate_log_cbar_ticklabel_list(vmin: float, vmax: float) -> list:
    """Create list of log format colorbar label ticks as string"""
    return ["10$^{%s}$" % int(i) for i in np.arange(vmin, vmax + 1)]  # pylint: disable=C0209


def _read_location(nc_file: str) -> str:
    """Returns site name."""
    with netCDF4.Dataset(nc_file) as nc:
        site_name = nc.location
    return site_name


def _read_date(nc_file: str) -> date:
    """Returns measurement date."""
    with netCDF4.Dataset(nc_file) as nc:
        case_date = date(int(nc.year), int(nc.month), int(nc.day))
    return case_date


def _add_subtitle(fig, case_date: date, site_name: str):
    """Adds subtitle into figure."""
    text = _get_subtitle_text(case_date, site_name)
    fig.suptitle(
        text, fontsize=13, y=0.885, x=0.07, horizontalalignment="left", verticalalignment="bottom"
    )


def _get_subtitle_text(case_date: date, site_name: str) -> str:
    site_name = site_name.replace("-", " ")
    return f"{site_name}, {case_date.strftime('%d %b %Y').lstrip('0')}"


def _create_save_name(save_path: str, case_date: date, field_names: list, fix: str = "") -> str:
    """Creates file name for saved images."""
    date_string = case_date.strftime("%Y%m%d")
    return f"{save_path}{date_string}_{'_'.join(field_names)}{fix}.png"


def _plot_relative_error(ax, error: ma.MaskedArray, ax_values: tuple):
    pl = ax.pcolorfast(*ax_values, error[:-1, :-1].T, cmap="RdBu", vmin=-30, vmax=30)
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_label("%", fontsize=13)
    median_error = ma.median(error.compressed())
    median_error = np.round(median_error, 3)
    ax.set_title(f"Median relative error: {median_error} %", fontsize=14)


def _lin2log(*args) -> list:
    return [ma.log10(x) for x in args]


# Misc plotting routines:


def plot_2d(
    data: ma.MaskedArray,
    cbar: bool = True,
    cmap: str = "viridis",
    ncolors: int = 50,
    clim: tuple = None,
    ylim: tuple = None,
    xlim: tuple = None,
):
    """Simple plot of 2d variable."""
    plt.close()
    if cbar:
        cmap = plt.get_cmap(cmap, ncolors)
        plt.imshow(ma.masked_equal(data, 0).T, aspect="auto", origin="lower", cmap=cmap)
        plt.colorbar()
    else:
        plt.imshow(ma.masked_equal(data, 0).T, aspect="auto", origin="lower")
    if clim:
        plt.clim(clim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()


def compare_files(
    nc_files: list,
    field_name: str,
    show: bool = True,
    relative_err: bool = False,
    save_path: str = None,
    max_y: int = 12,
    dpi: int = 120,
    image_name: str = None,
) -> Dimensions:
    """Plots one particular field from two Cloudnet files.

    Args:
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
        Dimensions of the generated figure in pixels.

    """
    plot_type = ATTRIBUTES[field_name].plot_type
    fields = [_find_valid_fields(file, [field_name])[0][0] for file in nc_files]
    nc = netCDF4.Dataset(nc_files[0])
    nc.close()
    ax_values = [_read_ax_values(nc_file) for nc_file in nc_files]
    subtitle = (f" - {os.path.basename(nc_files[0])}", f" - {os.path.basename(nc_files[0])}")
    n_subs = 3 if relative_err is True else 2
    fig, axes = _initialize_figure(n_subs, dpi)

    for ii, ax in enumerate(axes[:2]):
        field, ax_value = _screen_high_altitudes(fields[ii], ax_values[ii], max_y)
        _set_ax(ax, max_y)
        _set_title(ax, field_name, subtitle[ii])

        if plot_type == "model":
            _plot_colormesh_data(ax, field, field_name, ax_value)
        elif plot_type == "bar":
            _plot_bar_data(ax, field, ax_value[0])
            _set_ax(ax, 2, ATTRIBUTES[field_name].ylabel)
        elif plot_type == "segment":
            _plot_segment_data(ax, field, field_name, ax_value)
        else:
            _plot_colormesh_data(ax, field, field_name, ax_value)
            if relative_err is True and ii == 1:
                _set_ax(axes[-1], max_y)
                error, ax_value = _get_relative_error(fields, ax_values, max_y)
                _plot_relative_error(axes[-1], error, ax_value)

    case_date = _set_labels(fig, axes[-1], nc_files[0], sub_title=False)
    _handle_saving(image_name, save_path, show, case_date, [field_name], "_comparison")
    return Dimensions(fig, axes)
