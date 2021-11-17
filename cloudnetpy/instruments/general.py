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


def add_zenith_angle(obj: any) -> None:
    """Adds solar zenith angle."""
    key = 'elevation'
    try:
        elevation = obj.data[key].data
    except KeyError:
        elevation = obj.getvar(key)
    zenith = 90 - elevation
    if not utils.isscalar(zenith):
        tolerance = 0.5
        difference = np.diff(zenith)
        if np.any(difference > tolerance):
            logging.warning(f'Varying zenith angle. Maximum difference: {max(difference)}')
    obj.data['zenith_angle'] = CloudnetArray(zenith, 'zenith_angle')
    obj.data.pop(key, None)


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
