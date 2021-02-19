"""Misc. plotting routines for Cloudnet products."""

from datetime import date
import numpy as np
import numpy.ma as ma
from numpy import ndarray
import netCDF4
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cloudnetpy import utils
from cloudnetpy.plotting import legacy_meta
import cloudnetpy.products.product_tools as ptools
from cloudnetpy.plotting.plot_meta import ATTRIBUTES
from cloudnetpy.products.product_tools import CategorizeBits


def generate_figure(nc_file, field_names, show=True, save_path=None,
                    max_y=12, dpi=200, image_name=None, sub_title=True, title=True):
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
        sub_title (bool, optional): Add subtitle to image. Default is True.
        title (bool, optional): Add title to image. Default is True.

    Examples:
        >>> from cloudnetpy.plotting import generate_figure
        >>> generate_figure('categorize_file.nc', ['Z', 'v', 'width', 'ldr', 'beta', 'lwp'])
        >>> generate_figure('iwc_file.nc', ['iwc', 'iwc_error', 'iwc_retrieval_status'])
        >>> generate_figure('lwc_file.nc', ['lwc', 'lwc_error', 'lwc_retrieval_status'], max_y=4)
        >>> generate_figure('classification_file.nc', ['target_classification', 'detection_status'])
        >>> generate_figure('drizzle_file.nc', ['Do', 'mu', 'S'], max_y=3)
    """
    valid_fields, valid_names = _find_valid_fields(nc_file, field_names)
    is_height = _is_height_dimension(nc_file)
    fig, axes = _initialize_figure(len(valid_fields))

    for ax, field, name in zip(axes, valid_fields, valid_names):
        plot_type = ATTRIBUTES[name].plot_type
        if title:
            _set_title(ax, name, '')
        if not is_height:
            source = ATTRIBUTES[name].source
            time = _read_time_vector(nc_file)
            _plot_instrument_data(ax, field, name, source, time)
            continue
        ax_value = _read_ax_values(nc_file)
        field, ax_value = _screen_high_altitudes(field, ax_value, max_y)
        _set_ax(ax, max_y)
        if plot_type == 'bar':
            _plot_bar_data(ax, field, ax_value[0])
            _set_ax(ax, 2, ATTRIBUTES[name].ylabel)

        elif plot_type == 'segment':
            _plot_segment_data(ax, field, name, ax_value)

        else:
            _plot_colormesh_data(ax, field, name, ax_value)
    case_date = _set_labels(fig, axes[-1], nc_file, sub_title)
    _handle_saving(image_name, save_path, show, dpi, case_date, valid_names)


def _handle_saving(image_name, save_path, show, dpi, case_date, field_names,
                   fix=""):
    if image_name:
        plt.savefig(image_name, bbox_inches='tight', dpi=dpi)
    elif save_path:
        file_name = _create_save_name(save_path, case_date, field_names, fix)
        plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.close()


def _get_relative_error(fields, ax_values, max_y):
    old_data_interp = utils.interpolate_2d_masked(fields[0], *ax_values)
    error = utils.calc_relative_error(old_data_interp, fields[1])
    return _screen_high_altitudes(error, ax_values[1], max_y)


def _set_labels(fig, ax, nc_file, sub_title=True):
    ax.set_xlabel('Time (UTC)', fontsize=13)
    case_date = _read_date(nc_file)
    site_name = _read_location(nc_file)
    if sub_title:
        _add_subtitle(fig, case_date, site_name)
    return case_date


def _set_title(ax, field_name, identifier=" from CloudnetPy"):
    ax.set_title(f"{ATTRIBUTES[field_name].name}{identifier}", fontsize=14)


def _find_valid_fields(nc_file, names):
    """Returns valid field names and corresponding data."""
    valid_names, valid_data = names[:], []
    try:
        bits = CategorizeBits(nc_file)
    except KeyError:
        bits = None
    nc = netCDF4.Dataset(nc_file)
    for name in names:
        if name in nc.variables:
            valid_data.append(nc.variables[name][:])
        elif bits and name in CategorizeBits.category_keys:
            valid_data.append(bits.category_bits[name])
        elif bits and name in CategorizeBits.quality_keys:
            valid_data.append(bits.quality_bits[name])
        else:
            valid_names.remove(name)
    nc.close()
    if not valid_names:
        raise ValueError('No fields to be plotted')
    return valid_data, valid_names


def _is_height_dimension(full_path: str) -> bool:
    nc = netCDF4.Dataset(full_path)
    is_height = any([key in nc.variables for key in ('height', 'range')])
    nc.close()
    return is_height


def _initialize_figure(n_subplots: int):
    """Creates an empty figure according to the number of subplots."""
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 4 + (n_subplots-1)*4.8))
    fig.subplots_adjust(left=0.06, right=0.73)
    if n_subplots == 1:
        axes = [axes]
    return fig, axes


def _read_ax_values(full_path: str, file_type: str = None):
    """Returns time and height arrays."""
    if not file_type:
        nc = netCDF4.Dataset(full_path)
        file_type = nc.cloudnet_file_type
        nc.close()
    if file_type == 'radar':
        fields = ['time', 'range']
    else:
        fields = ['time', 'height']
    time, height = ptools.read_nc_fields(full_path, fields)
    if file_type == 'model':
        height = ma.mean(height, axis=0)
    height_km = height / 1000
    return time, height_km


def _read_time_vector(nc_file: str) -> ndarray:
    """Converts time vector to fraction hour."""
    nc = netCDF4.Dataset(nc_file)
    time = nc.variables['time']
    fraction_time = utils.seconds2hours(time[:])
    nc.close()
    return fraction_time


def _screen_high_altitudes(data_field, ax_values, max_y):
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
        ind = (np.argmax(alt > max_y) or len(alt)) + 1
        data_field = data_field[:, :ind]
        alt = alt[:ind]
    return data_field, (ax_values[0], alt)


def _set_ax(ax, max_y, ylabel=None, min_y=0.0):
    """Sets ticks and tick labels for plt.imshow()."""
    ticks_x_labels = _get_standard_time_ticks()
    ax.set_ylim(min_y, max_y)
    ax.set_xticks(np.arange(0, 25, 4, dtype=int))
    ax.set_xticklabels(ticks_x_labels, fontsize=12)
    ax.set_ylabel('Height (km)', fontsize=13)
    ax.set_xlim(0, 24)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)


def _get_standard_time_ticks(resolution=4):
    """Returns typical ticks / labels for a time vector between 0-24h."""
    return [f"{int(i):02d}:00" if 24 > i > 0 else ''
            for i in np.arange(0, 24.01, resolution)]


def _plot_bar_data(ax, data, time):
    """Plots 1D variable as bar plot.

    Args:
        ax (obj): Axes object.
        data (ndarray): 1D data array.
        time (ndarray): 1D time array.

    """
    # TODO: unit change somewhere else
    ax.plot(time, data/1000, color='navy')
    ax.bar(time, data.filled(0)/1000, width=1/120, align='center', alpha=0.5,
           color='royalblue')
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
    colorbar.set_ticks(np.arange(len(clabel)))
    colorbar.ax.set_yticklabels(clabel, fontsize=13)


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

    if name == 'cloud_fraction':
        data[data < 0.1] = ma.masked

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

    if variables.plot_type != 'bit':
        colorbar = _init_colorbar(pl, ax)
        colorbar.set_label(variables.clabel, fontsize=13)

    if variables.plot_scale == 'logarithmic':
        tick_labels = _generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax+1))
        colorbar.ax.set_yticklabels(tick_labels)


def _plot_instrument_data(ax, data, name, product, time):
    if product == 'mwr':
        _plot_mwr(ax, data, name, time)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.965, pos.height])


def _plot_mwr(ax, data, name, time):
    time, data = _remove_timestamps_of_next_date(time, data)
    rolling_mean, width = _calculate_rolling_mean(time, data/1000)
    data_filter = _filter_noise(data/1000, 10)
    ax.plot(time, data_filter, color='royalblue', linewidth=0.1)
    ax.plot(time, np.zeros(data.shape), color='k', linewidth=0.8)
    ax.plot(time[int(width / 2 - 1):int(-width / 2)], rolling_mean,
            color='sienna', linewidth=2.0)
    ax.plot(time[int(width / 2 - 1):int(-width / 2)], rolling_mean,
            color='wheat', linewidth=0.6)
    _set_ax(ax, round(np.max(data / 1000), 3) + 0.0005, ATTRIBUTES[name].ylabel,
            min_y=round(np.min(data / 1000), 3) - 0.0005)


def _remove_timestamps_of_next_date(time, data, n=100):
    """Check if timestamps of a next date is in a time-array. If so, remove
        extra timestamps which should not be included in a current date."""
    for i, t in enumerate(time[-n:-1]):
        if t > time[-n:][i+1]:
            nextday = i - n + 1
            return time[:nextday], data[:nextday]
    return time, data


def _calculate_rolling_mean(time, data):
    width = len(time[time <= time[0] + 0.3])
    if (width % 2) != 0:
        width = width + 1
    rolling_window = np.blackman(width)
    rolling_mean = np.convolve(data, rolling_window, 'valid')
    rolling_mean = rolling_mean / np.sum(rolling_window)
    return rolling_mean, width


def _filter_noise(data, n=5):
    """IIR filter"""
    b = [1.0 / n] * n
    a = 1
    return lfilter(b, a, data)


def _init_colorbar(plot, axis):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def _generate_log_cbar_ticklabel_list(vmin, vmax):
    """Create list of log format colorbar label ticks as string"""
    return ['10$^{%s}$' % int(i) for i in np.arange(vmin, vmax+1)]


def _read_location(nc_file):
    """Returns measurement date string."""
    nc = netCDF4.Dataset(nc_file)
    site_name = nc.location
    nc.close()
    return site_name


def _read_date(nc_file):
    """Returns measurement date string."""
    nc = netCDF4.Dataset(nc_file)
    case_date = date(int(nc.year), int(nc.month), int(nc.day))
    nc.close()
    return case_date


def _add_subtitle(fig, case_date, site_name):
    """Adds subtitle into figure."""
    text = _get_subtitle_text(case_date, site_name)
    fig.suptitle(text, fontsize=13, y=0.885, x=0.07, horizontalalignment='left',
                 verticalalignment='bottom')


def _get_subtitle_text(case_date, site_name):
    site_name = site_name.replace('-', ' ')
    return f"{site_name}, {case_date.strftime('%-d %b %Y')}"


def _create_save_name(save_path, case_date, field_names, fix=''):
    """Creates file name for saved images."""
    date_string = case_date.strftime("%Y%m%d")
    return f"{save_path}{date_string}_{'_'.join(field_names)}{fix}.png"


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


def _plot_relative_error(ax, error, ax_values):
    pl = ax.pcolorfast(*ax_values, error[:-1, :-1].T, cmap='RdBu', vmin=-30,
                       vmax=30)
    colorbar = _init_colorbar(pl, ax)
    colorbar.set_label("%", fontsize=13)
    median_error = ma.median(error.compressed())
    median_error = "%.3f" % median_error
    ax.set_title(f"Median relative error: {median_error} %", fontsize=14)


def _lin2log(*args):
    return [ma.log10(x) for x in args]

# LEGACY DATA PLOTTING ROUTINES:


def generate_legacy_figure(full_path: str,
                           product: str,
                           field_name: str,
                           show: bool = False,
                           save_path: str = None,
                           max_y: int = 12,
                           dpi: int = 200,
                           image_name: str = None) -> None:
    """ Plots one particular field from Cloudnet legacy product file.

    Args:
        full_path (str): Full path of a legacy Cloudnet file.
        product (str): Cloudnet file type.
        field_name (str): Name of variable to be plotted.
        show (bool, optional): If True, shows the plot. Default is False.
        save_path (str, optional): If defined, saves the image to this path.
            Default is None.
        max_y (int, optional): Upper limit of images (km). Default is 12.
        dpi (int, optional): Quality of plots. Default is 200.
        image_name (str, optional): Name (and full path) of the output image.
            Overrides the *save_path* option. Default is None.

    """
    plot_type = ATTRIBUTES[field_name].plot_type
    field = _find_valid_fields(full_path, [field_name])[0][0]
    fig, ax = _initialize_figure(1)
    ax = ax[0]
    ax_values = _read_ax_values(full_path, product)
    field, ax_value = _screen_high_altitudes(field, ax_values, max_y)
    _set_ax(ax, max_y)
    if plot_type == 'bar':
        _plot_bar_data(ax, field, ax_value[0])
        _set_ax(ax, 2, ATTRIBUTES[field_name].ylabel)
    elif plot_type == 'segment':
        field, field_name = legacy_meta.fix_legacy_data(field, field_name)
        _plot_segment_data(ax, field, field_name, ax_value)
    else:
        _plot_colormesh_data(ax, field, field_name, ax_value)
    case_date = _set_labels(fig, ax, full_path, False)
    _handle_saving(image_name, save_path, show, dpi, case_date, [field_name])


def compare_files(nc_files, field_name, show=True, relative_err=False,
                  save_path=None, max_y=12, dpi=200, image_name=None):
    """ Plots one particular field from old and new cloudnet files.

    Args:
        nc_files (tuple): Tuple of strings of the two files to compare
                         [0] = CloudnetPy file
                         [1] = legacy Cloudnet file
        field_name (str): Name of variable to be plotted.
        show (bool, optional): If True, shows the plot.
        relative_err (bool, optional): If True, plots also relative error. Makes
            sense only for continuous variables. Default is False.
        save_path (str, optional): If defined, saves the image to this path.
            Default is None.
        max_y (int, optional): Upper limit of images (km). Default is 12.
        dpi (int, optional): Quality of plots. Default is 200.
        image_name (str, optional): Name (and full path) of the output image.
            Overrides the *save_path* option. Default is None.

    """

    def _init_figure():
        n_subs = 3 if relative_err else 2
        return _initialize_figure(n_subs)

    plot_type = ATTRIBUTES[field_name].plot_type
    fields = [_find_valid_fields(file, [field_name])[0][0] for file in nc_files]
    ax_values = [_read_ax_values(nc_file) for nc_file in nc_files]
    subtitle = (" from CloudnetPy", " from Cloudnet")
    fig, axes = _init_figure()

    for ii, ax in enumerate(axes[:2]):
        field, ax_value = _screen_high_altitudes(fields[ii], ax_values[ii],
                                                 max_y)
        _set_ax(ax, max_y)
        _set_title(ax, field_name, subtitle[ii])

        if plot_type == 'model':
            _plot_colormesh_data(ax, field, field_name, ax_value)
        elif plot_type == 'bar':
            if field_name == 'lwp' and ii == 1:
                field *= 1000
            _plot_bar_data(ax, field, ax_value[0])
            _set_ax(ax, 2, ATTRIBUTES[field_name].ylabel)
        elif plot_type == 'segment':
            if ii == 1:
                field, field_name = legacy_meta.fix_legacy_data(field, field_name)
            _plot_segment_data(ax, field, field_name, ax_value)
        else:
            _plot_colormesh_data(ax, field, field_name, ax_value)
            if relative_err and ii == 1:
                _set_ax(axes[-1], max_y)
                error, ax_value = _get_relative_error(fields, ax_values, max_y)
                _plot_relative_error(axes[-1], error, ax_value)

    case_date = _set_labels(fig, axes[-1], nc_files[0])
    _handle_saving(image_name, save_path, show, dpi, case_date, [field_name],
                   '_comparison')
