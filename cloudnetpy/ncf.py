""" This modules contains netCDF related functions. The functions
in this module typically have a pointer to netCDF variable(s) as
an argument."""

import numpy as np
import numpy.ma as ma
import netCDF4
import sys


def load_nc(file_in):
    """ Returns instance of netCDF Dataset variables. """
    return netCDF4.Dataset(file_in).variables


def fetch_radar_meta(radar_file):
    """Returns some global metadata from radar nc-file.

    Args:
        radar_file (str): Full path of the cloud radar netCDF file.

    Returns:
        Dict containing radar frequency, measurement date, and radar
        (i.e. site) location: {'freq', 'date', 'location'}.

    Raises:
        KeyError: No frequency in the radar file.
        ValueError: Invalid frequency value, only 35 and 94 Ghz
            radars supported.

    """
    nc = netCDF4.Dataset(radar_file)
    try:
        location = nc.location
    except AttributeError:
        location = 'Unknown location'        
    try:
        freq = get_radar_freq(nc.variables)
    except (ValueError, KeyError) as error:
        raise error
    dvec = '-'.join([str(nc.year), str(nc.month).zfill(2),
                     str(nc.day).zfill(2)])
    return {'freq': freq, 'date': dvec, 'location': location}


def get_radar_freq(vrs):
    """ Returns frequency of radar.

    Args:
        vrs: A netCDF instance.

    Returns:
        Frequency or radar.

    Raises:
        KeyError: No frequency in the radar file.
        ValueError: Invalid frequency value.

    """
    possible_fields = ('radar_frequency', 'frequency')
    freq = [vrs[field][:] for field in vrs if field in possible_fields]
    if not freq:
        raise KeyError('Missing frequency. Not a radar file??')
    freq = freq[0]  # actual data of the masked data
    assert ma.count(freq) == 1, 'Multiple frequencies. Not a radar file??'
    try:
        get_wl_band(freq)
    except ValueError as error:
        raise ValueError('Only 35 and 94 GHz radars supported.')
    return float(freq)


def get_wl_band(freq):
    """ Returns integer that corresponds to the radar wavelength.

    Args:
        freq: Radar frequency.

    Returns:
        Integer corresponding to freqeuency. Possible return
        values are 0 (35.5 GHz) and 1 (~94 GHz).

    Raises:
        ValueError: Not supported frequency.

    """
    if 30 < freq < 40:
        wl_band = 0
    elif 90 < freq < 100:
        wl_band = 1
    else:
        raise ValueError
    return wl_band


def fetch_instrument_models(radar_file, lidar_file, mwr_file):
    """Returns models of the three Cloudnet instruments."""
    
    def _get_lidar_model(lidar_file):
        """Returns model of the lidar."""
        try:
            return netCDF4.Dataset(lidar_file).system
        except AttributeError:
            return 'Unknown lidar'
        
    def _get_mwr_model(mwr_file):
        """Returns model of the microwave radiometer."""
        try:
            return netCDF4.Dataset(mwr_file).radiometer_system
        except AttributeError:
            return 'Unknown radiometer'

    def _get_radar_model(radar_file):
        """Returns model of the cloud radar."""
        try:
            return netCDF4.Dataset(radar_file).title.split()[0]
        except AttributeError:
            return 'Unknown cloud radar'

    radar_model = _get_radar_model(radar_file)
    lidar_model = _get_lidar_model(lidar_file)
    mwr_model = _get_mwr_model(mwr_file)
    return {'radar': radar_model, 'lidar': lidar_model, 'mwr': mwr_model}


def km2m(var):
    """ Converts km to m.

    Read input and convert it to from km to m. The input must
    have 'units' attribute set to 'km' to trigger the conversion.

    Args:
        var: A netCDF variable.

    Returns:
        Altitude (scalar or array) converted to km.

    """
    alt = var[:]
    if var.units == 'km':
        alt = alt*1000
    return alt


def m2km(var):
    """ Converts m to km.

    Read Input and convert it to from m -> km. The input must
    have 'units' attribute set to 'm' to trigger the conversion.

    Args:
        var: A netCDF variable.

    Returns:
        Altitude (scalar or array)  converted to m.

    """
    alt = var[:]
    if var.units == 'm':
        alt = alt/1000
    return alt


def get_site_alt(*vrs):
    """ Returns altitude of the measurement site above mean sea level in [m].

    Site altitude is defined as the lowermost value of
    the investigated values.

    Args:
       *vrs: Files (Dataset variables) to be investigated.

    Returns:
        Altitude (m) of the measurement site.

    Raises:
        KeyError: If no 'altitude' field is found from any of
                  the input files.

    """
    field = 'altitude'
    alts = [km2m(var[field]) for var in vrs if field in var]
    if not alts:
        raise KeyError("Can't determine site altitude.")
    return min(alts)
