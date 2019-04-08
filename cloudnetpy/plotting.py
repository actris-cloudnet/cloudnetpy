"""Misc. plotting routines for Cloudnet products."""

from datetime import date
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4
import seaborn
import cloudnetpy.products.product_tools as ptools
from .plot_meta import ATTRIBUTES


def plot_2d(data, cbar=True, cmap='viridis', ncolors=50, clim=None):
    """Simple plot of 2d variable."""
    if cbar:
        cmap = plt.get_cmap(cmap, ncolors)
        plt.imshow(ma.masked_equal(data, 0).T, aspect='auto', origin='lower', cmap=cmap)
        plt.colorbar()
    else:
        plt.imshow(ma.masked_equal(data, 0).T, aspect='auto', origin='lower')
    if clim:
        plt.clim(clim)
    plt.show()


#IDENTIFIER = " from CloudnetPy"
IDENTIFIER = ""


def _plot_segment_data(ax, data, axes, name):
    """ Plotting data with segments as 2d variable.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        axes (tuple): Tuple containing time (datetime format) and height (km).
        name (string): name of plotted data

    """
    variables = ATTRIBUTES[name]
    n = len(variables.cbar)
    cmap = _colors_to_colormap(variables.cbar)
    pl = ax.pcolormesh(*axes, data.T, cmap=cmap, vmin=-0.5, vmax=n-0.5)
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_ticks(np.arange(0, n + 1, 1))
    colorbar.ax.set_yticklabels(variables.clabel, fontsize=13)
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)


def _plot_colormesh_data(ax, data, axes, name):
    """ Plot data with range of variability.

    Creates only one plot, so can be used both one plot and subplot type of figs

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): Figure object
        axes (tuple): Tuple containing time (datetime format) and height (km).
        name (string): name of plotted data
    """
    variables = ATTRIBUTES[name]
    cmap = plt.get_cmap(variables.cbar, 10)
    vmin, vmax = variables.plot_range
    if variables.plot_scale == 'logarithmic':
        data = np.log10(data)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    pl = ax.pcolormesh(*axes, data.T, cmap=cmap, vmin=vmin, vmax=vmax)
    colorbar = _init_colorbar(pl, ax)

    if variables.plot_scale == 'logarithmic':
        tick_labels = _generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax+1))
        colorbar.ax.set_yticklabels(tick_labels)

    colorbar.set_label(variables.clabel, fontsize=13)
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)


def _init_colorbar(plot, axis):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def _parse_field_names(nc_file, field_names):
    """Returns field names that actually exist in the nc-file."""
    variables = netCDF4.Dataset(nc_file).variables
    return [field for field in field_names if field in variables]


def generate_figure(nc_file, field_names, show=True, save_path=None,
                    max_y=12, dpi=200):
    """Generates a Cloudnet figure.

    Args:
        nc_file (str): Input file.
        field_names (list): Variable names to be plotted.
        show (bool, optional): If True, shows the figure. Default is True.
        save_path (str, optional): Setting this path will save the figure (in the
            given path). Default is None, when the figure is not saved.
        max_y (int, optional): Upper limit in the plots (km). Default is 12.
        dpi (int, optional): Figure quality (if saved). Higher value means
            more pixels, i.e., better image quality. Default is 200.

    """
    field_names = _parse_field_names(nc_file, field_names)
    data_fields = ptools.read_nc_fields(nc_file, field_names)
    n_fields = len(data_fields)
    case_date = _read_case_date(nc_file)
    axes = _read_axes(nc_file, case_date)
    fig, ax = _initialize_figure(n_fields)

    for i, name in enumerate(field_names):
        ax[i] = _initialize_time_height_axes(ax[i], n_fields, i, max_y)
        if ATTRIBUTES[name].plot_type == 'segment':
            plotting_func = _plot_segment_data
        else:
            plotting_func = _plot_colormesh_data
        plotting_func(ax[i], data_fields[i], axes, name)

    _add_subtitle(fig, n_fields, case_date)

    if save_path:
        file_name = _create_save_name(save_path, case_date, field_names)
        plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()


def _create_save_name(save_path, case_date, field_names):
    """Creates file name for saved images."""
    date_string = case_date.strftime("%Y%m%d")
    return f"{save_path}{date_string}_{'_'.join(field_names)}.png"


def _add_subtitle(fig, n_fields, case_date):
    """Adds subtitle into figure."""
    y = _calc_subtitle_y(n_fields)
    fig.suptitle(case_date.strftime("%-d %b %Y"), fontsize=13, y=y, x=0.11,
                 fontweight='bold')


def _calc_subtitle_y(n_fields):
    """Returns the correct y-position of subtitle. """
    pos = 0.903
    step = 0.008
    for _ in range(2, n_fields):
        pos -= step
        step /= 2
    return 0.93 if n_fields == 1 else pos


def _read_case_date(nc_file):
    """Returns measurement date string."""
    obj = netCDF4.Dataset(nc_file)
    return date(int(obj.year), int(obj.month), int(obj.day))


def _read_axes(nc_file, case_date):
    """Returns time (datetime format) and height (km)."""
    decimal_hour, height = ptools.read_nc_fields(nc_file, ('time', 'height'))
    datetime_time = ptools.convert_dtime_to_datetime(case_date, decimal_hour)
    height_km = height / 1000
    return datetime_time, height_km


def _generate_log_cbar_ticklabel_list(vmin, vmax):
    """Create list of log format colorbar label ticks as string"""
    return ['10$^{%s}$' % int(i) for i in np.arange(vmin, vmax+1)]


def _initialize_figure(n_subplots):
    """Creates an empty figure according to the number of subplots."""
    fig, ax = plt.subplots(n_subplots, 1, figsize=(16, 4 + (n_subplots-1)*4.8))
    fig.subplots_adjust(left=0.06, right=0.73)
    if n_subplots == 1:
        ax = [ax]
    return fig, ax


def _colors_to_colormap(color_list):
    """Transforms list of colors to colormap"""
    return ListedColormap(seaborn.color_palette(color_list).as_hex())


def _initialize_time_height_axes(ax, n_subplots, current_subplot, max_y):
    xlabel = 'Time ' + r'(UTC)'
    ylabel = 'Height (km)'
    date_format = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.tick_params(axis='x', labelsize=12)
    if current_subplot == n_subplots - 1:
        ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, max_y)
    ax.set_ylabel(ylabel, fontsize=13)
    return ax
