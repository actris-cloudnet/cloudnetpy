"""Misc. plotting routines for Cloudnet products."""

from datetime import date
import numpy as np
import numpy.ma as ma
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cloudnetpy.products.product_tools as ptools
from cloudnetpy.plot_meta import ATTRIBUTES
from cloudnetpy.products.product_tools import CategorizeBits

# IDENTIFIER = " from CloudnetPy"
IDENTIFIER = ""


def generate_figure(nc_file, field_names, show=True, save_path=None,
                    max_y=12, dpi=200, image_name=None):
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
        image_name (str, optional): Name (and full path) of the output image.
            Overrides the *save_path* option. Default is None.

    Examples:
        >>> from cloudnetpy.plotting import generate_figure
        >>> generate_figure('categorize_file.nc', ['Z', 'v', 'width', 'ldr', 'beta', 'lwp'])
        >>> generate_figure('iwc_file.nc', ['iwc', 'iwc_error', 'iwc_retrieval_status'])
        >>> generate_figure('lwc_file.nc', ['lwc', 'lwc_error', 'lwc_retrieval_status'], max_y=4)
        >>> generate_figure('classification_file.nc', ['target_classification', 'detection_status'])
        >>> generate_figure('drizzle_file.nc', ['Do', 'mu', 'S'], max_y=3)
    """
    valid_fields, valid_names = _find_valid_fields(nc_file, field_names)
    n_fields = len(valid_fields)
    fig, axes = _initialize_figure(n_fields)

    for axis, field, name in zip(axes, valid_fields, valid_names):
        plot_type = ATTRIBUTES[name].plot_type
        axes_data = _read_axes(nc_file, plot_type)
        field, axes_data = _fix_data_limitation(field, axes_data, max_y)
        _set_axes(axis, max_y)

        if plot_type == 'bar':
            _plot_bar_data(axis, field, name, axes_data[0])
            _set_axes(axis, 2, ATTRIBUTES[name].ylabel)

        elif plot_type == 'segment':
            _plot_segment_data(axis, field, name, axes_data)

        else:
            _plot_colormesh_data(axis, field, name, axes_data)

    axes[-1].set_xlabel('Time (UTC)', fontsize=13)
    case_date, site_name = _read_case_date(nc_file)
    _add_subtitle(fig, case_date, site_name)

    if image_name:
        plt.savefig(image_name, bbox_inches='tight', dpi=dpi)
    elif save_path:
        file_name = _create_save_name(save_path, case_date, max_y, valid_names)
        plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()

    plt.close()


def _find_valid_fields(nc_file, names):
    """Returns valid field names and corresponding data."""
    valid_names, valid_data = names[:], []
    nc_variables = netCDF4.Dataset(nc_file).variables
    try:
        bits = CategorizeBits(nc_file)
    except KeyError:
        bits = None
    for name in names:
        if name in nc_variables:
            valid_data.append(nc_variables[name][:])
        elif bits and name in CategorizeBits.category_keys:
            valid_data.append(bits.category_bits[name])
        elif bits and name in CategorizeBits.quality_keys:
            valid_data.append(bits.quality_bits[name])
        else:
            valid_names.remove(name)
    return valid_data, valid_names


def _initialize_figure(n_subplots):
    """Creates an empty figure according to the number of subplots."""
    fig, ax = plt.subplots(n_subplots, 1, figsize=(16, 4 + (n_subplots-1)*4.8))
    fig.subplots_adjust(left=0.06, right=0.73)
    if n_subplots == 1:
        ax = [ax]
    return fig, ax


def _read_axes(nc_file, axes_type=None):
    """Returns time and height arrays."""

    def _get_correct_dimension(field_names):
        """Model dimensions are different in old/new files."""
        variables = netCDF4.Dataset(nc_file).variables
        for name in field_names:
            yield name.split('_')[-1] if name not in variables else name

    if axes_type == 'model':
        fields = _get_correct_dimension(['model_time', 'model_height'])
    else:
        fields = ['time', 'height']
    time, height = ptools.read_nc_fields(nc_file, fields)
    height_km = height / 1000
    return time, height_km


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
    return [f"{int(i):02d}:00" if 24 > i > 0 else ''
            for i in np.arange(0, 24.01, resolution)]


def _plot_bar_data(ax, data, name, time):
    """Plots 1D variable as bar plot.

    Args:
        ax (obj): Axes object.
        data (ndarray): 1D data array.
        name (string): Name of plotted data.
        time (ndarray): 1D time array.

    """
    # TODO: unit change somewhere else
    variables = ATTRIBUTES[name]
    width = 1/120
    ax.plot(time, data/1000, color='navy')
    ax.bar(time, data.filled(0)/1000, width, align='center', alpha=0.5, color='royalblue')
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
    def _hide_segments(data_in):
        labels = [x[0] for x in variables.clabel]
        colors = [x[1] for x in variables.clabel]
        segments_to_hide = np.char.startswith(labels, '_')
        indices = np.where(segments_to_hide)[0]
        for ind in np.flip(indices):
            del labels[ind], colors[ind]
            data_in[data_in == ind] = ma.masked
            data_in[data_in > ind] -= 1
        return data_in, colors, labels

    variables = ATTRIBUTES[name]
    data, cbar, clabel = _hide_segments(data)
    cmap = ListedColormap(cbar)
    pl = ax.pcolorfast(*axes, data[:-1, :-1].T, cmap=cmap, vmin=-0.5,
                       vmax=len(cbar) - 0.5)
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_ticks(np.arange(len(clabel) + 1))
    colorbar.ax.set_yticklabels(clabel, fontsize=13)
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

    if variables.plot_type == 'bit':
        cmap = ListedColormap(variables.cbar)
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])
    else:
        cmap = plt.get_cmap(variables.cbar, 22)

    vmin, vmax = variables.plot_range

    if variables.plot_scale == 'logarithmic':
        data, vmin, vmax = _lin2log(data, vmin, vmax)

    pl = ax.pcolorfast(*axes, data[:-1, :-1].T, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(variables.name + IDENTIFIER, fontsize=14)

    if variables.plot_type != 'bit':
        colorbar = _init_colorbar(pl, ax)
        colorbar.set_label(variables.clabel, fontsize=13)

    if variables.plot_scale == 'logarithmic':
        tick_labels = _generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax+1))
        colorbar.ax.set_yticklabels(tick_labels)


def _init_colorbar(plot, axis):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def _generate_log_cbar_ticklabel_list(vmin, vmax):
    """Create list of log format colorbar label ticks as string"""
    return ['10$^{%s}$' % int(i) for i in np.arange(vmin, vmax+1)]


def _read_case_date(nc_file):
    """Returns measurement date string."""
    obj = netCDF4.Dataset(nc_file)
    case_date = date(int(obj.year), int(obj.month), int(obj.day))
    site_name = obj.location
    return case_date, site_name


def _add_subtitle(fig, case_date, site_name):
    """Adds subtitle into figure."""
    site_name = site_name.replace('-', ' ')
    text = f"{site_name}, {case_date.strftime('%-d %b %Y')}"
    fig.suptitle(text, fontsize=13, y=0.885, x=0.07, horizontalalignment='left',
                 verticalalignment='bottom', fontweight='bold')


def _create_save_name(save_path, case_date, max_y, field_names):
    """Creates file name for saved images."""
    date_string = case_date.strftime("%Y%m%d")
    return f"{save_path}{date_string}_{max_y}km_{'_'.join(field_names)}.png"


def _lin2log(*args):
    return [ma.log10(x) for x in args]


def plot_2d(data, cbar=True, cmap='viridis', ncolors=50, clim=None):
    """Simple plot of 2d variable."""
    plt.close()
    if cbar:
        cmap = plt.get_cmap(cmap, ncolors)
        plt.imshow(ma.masked_equal(data, 0).T, aspect='auto', origin='lower', cmap=cmap)
        plt.colorbar()
    else:
        plt.imshow(ma.masked_equal(data, 0).T, aspect='auto', origin='lower')
    if clim:
        plt.clim(clim)
    plt.show()


"""
def _swap_segments(data):
    def _swap_data(arr):
            ind_a = np.where(data == a)
            ind_b = np.where(data == b)
            arr[ind_a], arr[ind_b] = b, a

        def _swap_elements(lst):
            lst[a], lst[b] = lst[b], lst[a]

        for a, b in variables.swap_labels:
            _swap_data(data)
            _swap_elements(cbar)
            _swap_elements(clabel)
        return data, cbar, clabel
"""