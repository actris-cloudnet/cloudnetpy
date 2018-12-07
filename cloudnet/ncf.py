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
    """ Convert m to km.

    Read Input and convert it to from km -> m (if needed). The input must
    have 'units' attribute set to 'km' to trigger the conversion.

    Args:
        var: A netCDF variable.

    Returns:
        Altitude (scalar or array)  converted to km. 

    """
    y = var[:]
    if var.units == 'km':
        y = y*1000
    return y


def get_radar_freq(vrs):
    """ Return frequency of radar.

    Args:
        vrs: Pointer to radar variables.

    Returns:
        Frequency or radar.

    Raises:
        KeyError: No frequency in the radar file.
        ValueError: Invalid frequency value.

    """
    possible_fields = ('radar_frequency', 'frequency')  # Several possible
    freq = [vrs[field][:] for field in vrs if field in possible_fields]
    if not freq:
        raise KeyError('Missing frequency in the radar file.')
    freq = freq[0]  # actual data of the masked data
    assert ma.count(freq) == 1, 'Multiple frequencies. Not a radar file??'
    range_1 = 30 < freq < 40
    range_2 = 90 < freq < 100
    if not (range_1 or range_2):
        raise ValueError('Only 35 and 94 GHz radars supported.')
    return float(freq)


def get_site_alt(*vrs):
    """ Return altitude of the measurement site above mean sea level.

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
    alts = [var[field][:] for var in vrs if field in var]
    if not alts:
        raise KeyError("Can't determine site altitude.")
    return min(alts)
