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
from cloudnetpy import utils
from .plot_meta import ATTRIBUTES

# Plot range, colormap, is log plot
PARAMS = {
    'beta': [np.array((1e-8, 1e-4))*10, 'jet', True],
    'beta_raw': [np.array((1e-8, 1e-4))*1e8, 'jet', True],
    'Z': [(-40, 20), 'jet'],
    'Ze': [(-40, 20), 'jet'],
    'ldr': [(-35, -10), 'viridis'],
    'width': [(0, 1), 'viridis'],
    'v': [(-4, 2), 'RdBu_r'],
    'insect_prob': [(0, 1), 'viridis'],
    'radar_liquid_atten': [(0, 10), 'viridis'],
    'radar_gas_atten': [(0, 1), 'viridis'],
}


def plot_overview(file1, dvec, ylim=(0, 500),
                  savefig=False, savepath='', grid=False,
                  data_fields=('Z', 'v', 'ldr', 'width')):
    """Plots general image of data in categorize file."""
    nfields = len(data_fields)
    nsubs = (nfields, 1)
    plt.figure()
    for n, field in enumerate(data_fields, 1):
        _plot_data(nsubs, n, file1, field, ylim, grid, *PARAMS[field])
    _showpic(nsubs, dvec, savefig, savepath, 'overview')


def plot_variable(file1, file2, name, dvec, ylim=(0, 500),
                  savefig=False, savepath='', grid=False):
    """Plot relevant data for a Cloudnet variable."""
    if name == 'liquid':
        data_fields = ('Z', 'beta')
        bitno = 0
    elif name == 'melting':
        data_fields = ('Z', 'ldr', 'v')
        bitno = 3
    else:
        name = 'insects'
        data_fields = ('Z', 'ldr', 'width', 'insect_prob')
        bitno = 5

    nfields = len(data_fields)
    nsubs = (nfields+2, 1)
    plt.figure()
    for n, field in enumerate(data_fields, 1):
        _plot_data(nsubs, n, file1, field, ylim, grid, *PARAMS[field])
    _plot_bit(nsubs, nfields+1, file1, bitno, ylim)
    _plot_bit(nsubs, nfields+2, file2, bitno, ylim)
    _showpic(nsubs, dvec, savefig, savepath, name)


def _plot_data(nsubs, idx, filename, field, ylim, grid,
               clim, cmap='jet', log=False):
    """Plots 2-D data field."""
    plt.subplot(nsubs[0], nsubs[1], idx)
    ncv = netCDF4.Dataset(filename).variables
    data = ncv[field][:]
    if log:
        data = np.log(data)
        clim = np.log(clim)
    plt.imshow(data.T, aspect='auto', origin='lower', cmap=cmap)
    plt.clim(clim)
    _set_axes(ylim, data.shape, grid)
    plt.text(20, max(ylim)*0.8, field, fontsize=8)


def _plot_bit(nsubs, idx, filename, bitno, ylim, field='category_bits'):
    """Plots a bitfield."""
    plt.subplot(nsubs[0], nsubs[1], idx)
    ncv = netCDF4.Dataset(filename).variables
    data = utils.isbit(ncv[field][:], bitno)
    plt.imshow(ma.masked_equal(data, 0).T, aspect='auto', origin='lower')
    _set_axes(ylim, data.shape, grid=None)
    plt.text(20, max(ylim)*0.8, f"bit: {bitno}", fontsize=8)


def _set_axes(ylim, shape, grid):
    plt.xticks(np.linspace(0, shape[0], 13), [], length=20)
    plt.yticks(np.linspace(0, shape[1], 4), [])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.tick_params(length=0)
    plt.ylim(ylim)
    if grid:
        plt.grid(color=(.8, .8, .8), linestyle=':')


def _showpic(nsubs, dvec, savefig, imagepath, name):
    """Adjusts layout etc. and shows the actual figure."""
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00, hspace=0.0)
    plt.subplot(nsubs[0], nsubs[1], 1)
    plt.title(dvec, fontsize=8)
    if savefig:
        print('saving..')
        plt.savefig(f"{imagepath}{dvec}_{name}", dpi=200)
        plt.close()
        print('..done')
    else:
        plt.show()


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


IDENTIFIER = " from CloudnetPy"


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
    cmap = variables.cbar
    vmin, vmax = variables.plot_range
    if variables.plot_scale == 'logarithmic':
        data = np.log10(data)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    pl = ax.pcolormesh(*axes, data.T, cmap=cmap, vmin=vmin, vmax=vmax)
    colorbar = _init_colorbar(pl, ax)

    if variables.plot_scale == 'logarithmic':
        tick_labels = _generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax + 1, 1))
        colorbar.ax.set_yticklabels(tick_labels)

    colorbar.set_label(variables.clabel, fontsize=13)
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)


def _init_colorbar(plot, axis):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def generate_figure(nc_file, field_names, show=True, save_path=None, max_y=12):
    """ Usage to generate figure and plot wanted fig.
        Can be used for plotting both one fig and subplots.
        data_names is list of product names on select nc-file.
    """
    n_fields = len(field_names)
    case_date = _read_case_date(nc_file)
    axes = _read_axes(nc_file, case_date)
    data_fields = ptools.read_nc_fields(nc_file, field_names)

    fig, ax = _initialize_figure(n_fields)

    saving_name = ""
    for i, name in enumerate(field_names):
        ax[i] = _initialize_time_height_axes(ax[i], n_fields, i, max_y)
        if ATTRIBUTES[name].plot_type == 'segment':
            plotting_func = _plot_segment_data
        else:
            plotting_func = _plot_colormesh_data
        plotting_func(ax[i], data_fields[i], axes, name)
        saving_name += ("_" + name)

    _add_subtitle(fig, n_fields, case_date)

    if save_path:
        plt.savefig(save_path+case_date.strftime("%Y%m%d")+saving_name+".png",
                    bbox_inches='tight')
    if show:
        plt.show()


def _add_subtitle(fig, n_fields, case_date):
    """Adds subtitle into figure."""
    y = _calc_subtitle_y(n_fields)
    fig.suptitle(case_date.strftime("%-d %b %Y"), fontsize=13, y=y, x=0.11,
                 fontweight='bold')


def _calc_subtitle_y(n_fields):
    """Returns the correct y-position of subtitle. """
    return 0.92 - (n_fields - 1)*0.01


def _read_case_date(nc_file):
    """Returns measurement date string."""
    return date(int(netCDF4.Dataset(nc_file).year),
                int(netCDF4.Dataset(nc_file).month),
                int(netCDF4.Dataset(nc_file).day))


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
    ylabel = 'Height ' + '$(km)$'
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
