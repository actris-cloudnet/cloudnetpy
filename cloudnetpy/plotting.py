"""Misc. plotting routines for Cloudnet products."""

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
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


def plot_2d(data, cbar=True, cmap='viridis', ncolors=50, clim=None, color=None):
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


def plot_segment_data(ax, fig, data, xaxes, yaxes, name, subtit):
    """
    Plotting data with segments as 2d variable
    Args:
        ax(array): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        fig(object): Figure object
        xaxes(array): time in datetime format
        yaxes(array): height
        name(string): name of plotted data
        date(date): date of case date object
    """
    variables = ATTRIBUTES[name]
    n = len(variables.cbar)
    cmap = ptools.colors_to_colormap(variables.cbar)

    ax = ptools.initialize_time_height_axes(ax)

    pl = ax.pcolormesh(xaxes, yaxes, data.T, cmap=cmap,
                       vmin=-0.5, vmax=n-0.5)

    cbaxes = fig.add_axes([0.75, 0.11, 0.01, 0.77])
    cb = plt.colorbar(pl, ax=ax, cax=cbaxes)
    cb.set_ticks(np.arange(0, n + 1, 1))
    cb.ax.set_yticklabels(variables.clabel, fontsize= 13)

    ax.set_title(variables.name + subtit, fontsize=14)


def plot_colormesh_data(ax, fig, data, xaxes, yaxes, name, subtit):
    """
    Plot data with range of variability.

    Args:
        ax(array): Axes object of subplot (1,2,3,.. [1,1,],[1,2]... etc.)
        fig(object): Figure object
        xaxes(array): time in datetime format
        yaxes(array): height
        name(string): name of plotted data
        date(date): date of case date object
        subtit(string): title of fig
    """
    variables = ATTRIBUTES[name]
    cmap = variables.cbar

    ax = ptools.initialize_time_height_axes(ax)

    if variables.plot_scale == 'logarithmic':
        data = np.log10(data)
        vmin = np.log10(variables.plot_range[0])
        vmax = np.log10(variables.plot_range[-1])
        logs = ptools.generate_log_cbar_ticklabel_list(vmin,vmax)
    else:
        vmin = variables.plot_range[0]
        vmax = variables.plot_range[-1]

    pl = ax.pcolormesh(xaxes, yaxes, data.T, cmap=cmap, vmin=vmin,
                       vmax=vmax)

    cbaxes = fig.add_axes([0.75, 0.11, 0.01, 0.77])
    cb = plt.colorbar(pl, ax=ax, cax=cbaxes)
    # TODO: Jos ei logaritminen, mikä tulee rangeksi 1 paikalle?
    cb.set_ticks(np.arange(vmin,vmax+1,1))

    if variables.plot_scale == 'logarithmic':
        cb.ax.set_yticklabels(logs)
    cb.set_label(variables.clabel, fontsize = 13)

    ax.set_title(variables.name + subtit, fontsize=14)


def generate_one_figure(data_name, nc_file, save_path, show=False, save=False):
    """ Usage to generate figure and plot wanted fig """
    data, time_array, height, case_date = \
        ptools.read_variables_and_date(data_name, nc_file)
    time_array = ptools.convert_dtime_to_datetime(case_date, time_array)
    subtit = " from CloudnetPy"

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    fig.subplots_adjust(left=0.06, right=0.73)

    # Change to wanted one fig plot type
    plot_colormesh_data(ax, fig, data, time_array, height, data_name, subtit)

    fig.suptitle(case_date.strftime("%-d %b %Y"),
                 fontsize=13, y=0.93, x=0.1)

    if bool(save) is True:
        plt.savefig(save_path+case_date.strftime("%Y%m%d")+"_"+data_name+".png",
                    bbox_inches='tight')
    if bool(show) is True:
        plt.show()
