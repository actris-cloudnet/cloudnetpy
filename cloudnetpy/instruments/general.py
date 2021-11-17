"""General helper functions for instrument processing."""
import logging
import numpy as np
import numpy.ma as ma
from cloudnetpy import CloudnetArray
from cloudnetpy import utils
import netCDF4


def add_site_geolocation(obj: any):
    """Adds site geolocation."""
    for key in ('latitude', 'longitude', 'altitude'):
        value = None
        # User-supplied:
        if key in obj.site_meta:
            value = obj.site_meta[key]
        # From source global attributes (MIRA):
        elif hasattr(obj, 'dataset') and hasattr(obj.dataset, key.capitalize()):
            value = float(getattr(obj.dataset, key.capitalize()).split()[0])
        # From source data (BASTA / RPG):
        elif hasattr(obj, 'dataset') and key in obj.dataset.variables:
            value = obj.dataset.variables[key][:]
        if value is not None:
            value = float(ma.mean(value))
            obj.data[key] = CloudnetArray(value, key)


def add_radar_specific_variables(obj: any):
    """Adds radar specific variables."""
    key = 'radar_frequency'
    obj.data[key] = CloudnetArray(obj.instrument.frequency, key)
    possible_nyquist_names = ('ambiguous_velocity', 'NyquistVelocity')
    data = obj.getvar(*possible_nyquist_names)
    key = 'nyquist_velocity'
    obj.data[key] = CloudnetArray(np.array(data), key)


def add_zenith_angle(obj: any) -> list:
    """Adds solar zenith angle and returns valid time indices."""
    key = 'elevation'
    try:
        elevation = obj.data[key].data
    except KeyError:
        elevation = obj.getvar(key)
    zenith = 90 - elevation
    if utils.isscalar(zenith):
        ind = np.arange(len(obj.time))
    else:
        median_value = ma.median(zenith)
        tolerance = 0.1
        ind = np.isclose(zenith, median_value, atol=tolerance)
        n_removed = len(ind) - np.count_nonzero(ind)
        if n_removed > 0:
            logging.warning(f'Removed {n_removed} time steps due to varying zenith angle.')
    obj.data['zenith_angle'] = CloudnetArray(zenith, 'zenith_angle')
    obj.data.pop(key, None)
    return list(ind)


def add_height(obj: any):
    try:
        zenith_angle = ma.median(obj.data['zenith_angle'].data)
    except RuntimeError:
        logging.warning('Assuming 0 deg zenith_angle')
        zenith_angle = 0
    height = utils.range_to_height(obj.data['range'].data, zenith_angle)
    height += obj.data['altitude'].data
    height = np.array(height)
    obj.data['height'] = CloudnetArray(height, 'height')


def linear_to_db(obj, variables_to_log: tuple) -> None:
    """Changes linear units to logarithmic."""
    for name in variables_to_log:
        obj.data[name].lin2db()


def get_files_with_common_range(files: list) -> list:
    """Returns files with the same (most common) number of range gates."""
    n_range = []
    for file in files:
        nc = netCDF4.Dataset(file)
        n_range.append(len(nc.variables['range']))
        nc.close()
    most_common = np.bincount(n_range).argmax()
    n_removed = len(n_range != most_common)
    if n_removed > 0:
        logging.warning(f'Removed {n_removed} MIRA files due to inconsistent height vector')
    ind = np.where(n_range == most_common)[0]
    return [file for i, file in enumerate(files) if i in ind]


def screen_time_indices(obj: any, valid_indices: list) -> None:
    n_time = len(obj.time)
    for key, cloudnet_array in obj.data.items():
        array = cloudnet_array.data
        if not utils.isscalar(array) and array.shape[0] == n_time:
            if array.ndim == 1:
                cloudnet_array.data = array[valid_indices]
            elif array.ndim == 2:
                cloudnet_array.data = array[valid_indices, :]
    obj.time = obj.time[valid_indices]
