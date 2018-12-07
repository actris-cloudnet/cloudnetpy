""" This modules contains netCDF related functions. The functions 
in this module typically have a pointer to netCDF variable(s) as 
an argument."""

import netCDF4
import numpy as np
import numpy.ma as ma


def load_nc(file_in):
    """ Return pointer to netCDF file and its variables.

    Args:
        file_in (str): File name.

    Returns:
        Tuple containing

        - Pointer to file.
        - Pointer to file variables.

    """
    file_pointer = netCDF4.Dataset(file_in)
    return file_pointer, file_pointer.variables


def km2m(var):
    """ Convert m to km.

    Read Input and convert it to from km -> m (if needed). The input must
    have 'units' attribute set to 'km' to trigger the conversion.

    Args:
        vrs: A netCDF variable.

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
