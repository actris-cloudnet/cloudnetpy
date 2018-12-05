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
