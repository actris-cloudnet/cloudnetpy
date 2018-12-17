""" This modules contains netCDF related functions. The functions
in this module typically have a pointer to netCDF variable(s) as
an argument."""

import netCDF4
import numpy as np
import numpy.ma as ma


def load_nc(file_in):
    """ Return instance of netCDF Dataset variables. """
    return netCDF4.Dataset(file_in).variables


def km2m(var):
    """ Convert km to m.

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
    """ Convert m to km.

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


def get_radar_freq(vrs):
    """ Return frequency of radar.

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
        raise KeyError('Missing frequency in the radar file.')
    freq = freq[0]  # actual data of the masked data
    assert ma.count(freq) == 1, 'Multiple frequencies. Not a radar file??'
    try:
        get_wl_band(freq)
    except ValueError as error:
        raise ValueError('Only 35 and 94 GHz radars supported.')
    return float(freq)


def get_wl_band(freq):
    """ Return integer that corresponds to the radar wavelength.

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


def get_site_alt(*vrs):
    """ Return altitude of the measurement site above mean sea level in [m].

    Site altitude is defined as the lowermost value of
    the investigated values.

    Args:
       *vrs: Files (Dataset variables) to be investigated.

    Returns:
        Altitude of the measurement site.

    Raises:
        KeyError: If no 'altitude' field is found from any of
                  the input files.

    """
    field = 'altitude'
    alts = [km2m(var[field]) for var in vrs if field in var]
    if not alts:
        raise KeyError("Can't determine site altitude.")
    return min(alts)
