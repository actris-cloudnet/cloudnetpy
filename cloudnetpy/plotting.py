"""Misc. plotting routines for Cloudnet products."""

from datetime import date
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4
import cloudnetpy.products.product_tools as ptools
from .plot_meta import ATTRIBUTES
from cloudnetpy import utils


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


def _plot_segment_data(ax, data, name):
    """ Plotting data with segments as 2d variable.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        axes (tuple): Tuple containing time (datetime format) and height (km).
        name (string): name of plotted data

    """
    variables = ATTRIBUTES[name]
    n_fields = len(variables.cbar)
    cmap = ListedColormap(variables.cbar)
    pl = ax.imshow(data.T, cmap=cmap, origin='lower', vmin=0, vmax=n_fields,
                   aspect='auto')
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_ticks(np.arange(n_fields+1)+0.5)
    colorbar.ax.set_yticklabels(variables.clabel, fontsize=13)
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)


def _plot_colormesh_data(ax, data, name):
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

    pl = ax.imshow(data.T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                   aspect='auto')
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
    fig, ax = _initialize_figure(n_fields)

    for axis, field, name in zip(ax, data_fields, field_names):
        if ATTRIBUTES[name].plot_type == 'segment':
            _plot_segment_data(axis, field, name)
        else:
            _plot_colormesh_data(axis, field, name)

    axes = _read_axes(nc_file)
    _set_axes(ax, axes, max_y)
    _add_subtitle(fig, n_fields, case_date)

    if save_path:
        file_name = _create_save_name(save_path, case_date, field_names)
        plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()


def _set_axes(ax, axes, max_y):
    time, alt = axes
    ticks_y, ticks_y_labels, n_max_y = _get_ticks(alt, max_y, 2)
    ticks_x, _, n_max_x = _get_ticks(time, 24, 4)
    ticks_x_labels = ['', '04:00', '08:00', '12:00', '16:00', '20:00', '']
    for axis in ax:
        axis.set_yticks(ticks_y)
        axis.set_yticklabels(ticks_y_labels, fontsize=12)
        axis.set_ylim(0, n_max_y)
        axis.set_xticks(ticks_x)
        axis.set_xticklabels(ticks_x_labels, fontsize=12)
        axis.set_xlim(0, n_max_x)
        axis.set_ylabel('Height (km)', fontsize=13)
    ax[-1].set_xlabel('Time (UTC)', fontsize=13)


def _get_ticks(x, x_max, tick_step):
    step = utils.mdiff(x)
    n_steps_to_reach_max = round(x_max/step)
    n_steps_in_one_tick = round(tick_step/step)
    max_value = np.round(x[-1])
    ticks = np.linspace(0, max_value*n_steps_in_one_tick, max_value+1)
    ticks_labels = (np.arange(max_value+1)*tick_step).astype(int).astype(str)
    return ticks, ticks_labels, n_steps_to_reach_max


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


def _read_axes(nc_file):
    """Returns time (datetime format) and height (km)."""
    decimal_hour, height = ptools.read_nc_fields(nc_file, ('time', 'height'))
    height_km = height / 1000
    return decimal_hour, height_km


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
