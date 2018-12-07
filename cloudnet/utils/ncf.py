""" This modules contains netCDF related functions """

import netCDF4


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
