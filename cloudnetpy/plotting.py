"""Misc. plotting routines for Cloudnet products."""

import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
from cloudnetpy import utils

# Plot range, colormap, is log plot
PARAMS = {
    'beta': [np.array((1e-8, 1e-4))*10, 'jet', True],
    'beta_raw': [np.array((1e-8, 1e-4))*1e8, 'jet', True],
    'Z': [(-40, 20), 'jet'],
    'ldr': [(-35, -10), 'viridis'],
    'width': [(0, 1), 'viridis'],
    'v': [(-4, 2), 'RdBu_r'],
    'insect_prob': [(0, 1), 'viridis'],
    'radar_liquid_atten': [(0, 10), 'viridis'],
    'radar_gas_atten': [(0, 1), 'viridis'],
}


def plot_overview(file1, dvec, ylim=(0, 500), savefig=False,
                  savepath='', grid=False):
    """Plots general image of data in categorize file."""
    data_fields = ('Z', 'v', 'ldr', 'width', 'beta')
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
        data_fields = ('Z', 'beta', 'beta_raw')
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
        plt.savefig(f"{imagepath}{dvec}_{name}", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_2d(data, cbar=True, cmap='viridis', ncolors=50, clim=None):
    """Simple plot of 2d variable."""
    if cbar:
        cmap = plt.get_cmap(cmap, ncolors)
        plt.pcolormesh(ma.masked_equal(data, 0).T, cmap=cmap)
        plt.colorbar()
    else:
        plt.imshow(ma.masked_equal(data, 0).T, aspect='auto', origin='lower')
        plt.pcolormesh(ma.masked_equal(data, 0).T)
    if clim:
        plt.clim(clim)
    plt.show()
