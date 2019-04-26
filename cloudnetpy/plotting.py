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


# IDENTIFIER = " from CloudnetPy"
IDENTIFIER = ""


def _plot_bar_data(ax, data, name, time):
    """Plots 1D variable as bar plot.

    Args:
        ax (obj): Axes object.
        data (ndarray): 1D data array.
        name (string): Name of plotted data.
        time (ndarray): 1D time array.

    """
    variables = ATTRIBUTES[name]
    width = 1/120
    ax.plot(time, data/1000, color='navy')
    data[data < np.min(data)] = 0
    ax.bar(time, data/1000, width, align='center', alpha=0.5, color='royalblue')
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.965, pos.height])


def _plot_segment_data(ax, data, name, axes):
    """Plots categorical 2D variable.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.

    """
    variables = ATTRIBUTES[name]
    n_fields = len(variables.cbar)
    cmap = ListedColormap(variables.cbar)
    pl = ax.pcolorfast(*axes, data[:-1, :-1].T, cmap=cmap, vmin=-0.5,
                       vmax=n_fields - 0.5)
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_ticks(np.arange(n_fields+1))
    colorbar.ax.set_yticklabels(variables.clabel, fontsize=13)
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)


def _plot_colormesh_data(ax, data, name, axes):
    """Plots continuous 2D variable.

    Creates only one plot, so can be used both one plot and subplot type of figs.

    Args:
        ax (obj): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        data (ndarray): 2D data array.
        name (string): Name of plotted data.
        axes (tuple): Time and height 1D arrays.
    """
    variables = ATTRIBUTES[name]
    cmap = plt.get_cmap(variables.cbar, 22)
    vmin, vmax = variables.plot_range
    if variables.plot_scale == 'logarithmic':
        data, vmin, vmax = _lin2log(data, vmin, vmax)

    pl = ax.pcolorfast(*axes, data[:-1, :-1].T, vmin=vmin, vmax=vmax, cmap=cmap)
    colorbar = _init_colorbar(pl, ax)

    if variables.plot_scale == 'logarithmic':
        tick_labels = _generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax+1))
        colorbar.ax.set_yticklabels(tick_labels)

    colorbar.set_label(variables.clabel, fontsize=13)
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)


def _lin2log(*args):
    return [ma.log10(x) for x in args]


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
    fig, axes = _initialize_figure(n_fields)

    for axis, field, name in zip(axes, data_fields, field_names):
        plot_type = ATTRIBUTES[name].plot_type
        axes_data = _read_axes(nc_file, plot_type)
        field, axes_data = _fix_data_limitation(field, axes_data, max_y)
        _set_axes(axis, max_y)

        if plot_type == 'model':
            _plot_colormesh_data(axis, field, name, axes_data)

        elif plot_type == 'bar':
            _plot_bar_data(axis, field, name, axes_data[0])
            _set_axes(axis, 1, ATTRIBUTES[name].ylabel)

        elif plot_type == 'segment':
            _plot_segment_data(axis, field, name, axes_data)

        else:
            _plot_colormesh_data(axis, field, name, axes_data)

    axes[-1].set_xlabel('Time (UTC)', fontsize=13)
    case_date = _read_case_date(nc_file)
    _add_subtitle(fig, n_fields, case_date)

    if save_path:
        file_name = _create_save_name(save_path, case_date, max_y, field_names)
        plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()


def _fix_data_limitation(data_field, axes, max_y):
    """Removes altitudes from 2D data that are not visible in the figure.

    Bug in pcolorfast causing effect to axis not noticing limitation while
    saving fig. This fixes that bug till pcolorfast does fixing themselves.

    Args:
        data_field (ndarray): 2D data array.
        axes (tuple): Time and height 1D arrays.
        max_y (int): Upper limit in the plots (km).

    """
    alt = axes[-1]
    if data_field.ndim > 1:
        ind = (np.argmax(alt > max_y) or len(alt)) + 1
        data_field = data_field[:, :ind]
        alt = alt[:ind]
    return data_field, (axes[0], alt)


def _set_axes(axis, max_y, ylabel=None):
    """Sets ticks and tick labels for plt.imshow()."""
    ticks_x_labels = _get_standard_time_ticks()
    axis.set_ylim(0, max_y)
    axis.set_xticks(np.arange(0, 25, 4, dtype=int))
    axis.set_xticklabels(ticks_x_labels, fontsize=12)
    axis.set_ylabel('Height (km)', fontsize=13)
    axis.set_xlim(0, 24)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=13)


def _get_standard_time_ticks(resolution=4):
    """Returns typical ticks / labels for a time vector between 0-24h."""
    labels = [f"{int(i):02d}:00" if 24 > i > 0 else ''
              for i in np.arange(0, 24.01, resolution)]
    return labels


def _create_save_name(save_path, case_date, max_y, field_names):
    """Creates file name for saved images."""
    date_string = case_date.strftime("%Y%m%d")
    return f"{save_path}{date_string}_{max_y}km_{'_'.join(field_names)}.png"


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


def _read_axes(nc_file, axes_type=None):
    """Returns time and height arrays."""
    if axes_type == 'model':
        fields = ['model_time', 'model_height']
        fields = ptools.get_correct_dimensions(nc_file, fields)
    else:
        fields = ['time', 'height']
    time, height = ptools.read_nc_fields(nc_file, fields)
    height_km = height / 1000
    return time, height_km


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
