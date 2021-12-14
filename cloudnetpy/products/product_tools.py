"""General helper classes and functions for all products."""
from typing import Union
import numpy as np
import numpy.ma as ma
import netCDF4
import cloudnetpy.utils as utils


class CategorizeBits:
    """Class holding information about category and quality bits.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        category_bits (dict): Dictionary containing boolean fields for `droplet`,
            `falling`, `cold`, `melting`, `aerosol`, `insect`.

        quality_bits (dict): Dictionary containing boolean fields for `radar`,
            `lidar`, `clutter`, `molecular`, `attenuated` and `corrected`.

    """
    category_keys = ('droplet', 'falling', 'cold', 'melting', 'aerosol',
                     'insect')

    quality_keys = ('radar', 'lidar', 'clutter', 'molecular', 'attenuated',
                    'corrected')

    def __init__(self, categorize_file: str):
        self._categorize_file = categorize_file
        self.category_bits = self._read_bits('category')
        self.quality_bits = self._read_bits('quality')

    def _read_bits(self, bit_type: str) -> dict:
        """ Converts bitfield into dictionary."""
        nc = netCDF4.Dataset(self._categorize_file)
        try:
            bitfield = nc.variables[f"{bit_type}_bits"][:]
        except KeyError:
            nc.close()
            raise KeyError
        keys = getattr(CategorizeBits, f"{bit_type}_keys")
        bits = {key: utils.isbit(bitfield, i) for i, key in enumerate(keys)}
        nc.close()
        return bits


class ProductClassification(CategorizeBits):
    """Base class for creating different classifications in the child classes
    of various Cloudnet products. Child of CategorizeBits class.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        is_rain (ndarray): 1D array denoting rainy profiles.

    """
    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.is_rain = get_is_rain(categorize_file)


def get_is_rain(filename: str) -> np.ndarray:
    rain_rate = read_nc_fields(filename, 'rain_rate')
    is_rain = rain_rate != 0
    is_rain[is_rain.mask] = True
    return np.array(is_rain)


def read_nc_fields(nc_file: str, names: Union[str, list]) -> Union[ma.MaskedArray, list]:
    """Reads selected variables from a netCDF file.

    Args:
        nc_file: netCDF file name.
        names: Variables to be read, e.g. 'temperature' or ['ldr', 'lwp'].

    Returns:
        ndarray/list: Array in case of one variable passed as a string.
        List of arrays otherwise.

    """
    names = [names] if isinstance(names, str) else names
    nc = netCDF4.Dataset(nc_file)
    data = [nc.variables[name][:] for name in names]
    nc.close()
    return data[0] if len(data) == 1 else data


def interpolate_model(cat_file: str, names: Union[str, list]) -> Union[list, np.ndarray]:
    """Interpolates 2D model field into dense Cloudnet grid.

    Args:
        cat_file: Categorize file name.
        names: Model variable to be interpolated, e.g. 'temperature' or ['temperature', 'pressure'].

    Returns:
        ndarray/list: Array in case of one variable passed as a string. List of arrays otherwise.

    """
    def _interp_field(var_name):
        values = read_nc_fields(cat_file, ['model_time', 'model_height', var_name,
                                           'time', 'height'])
        return utils.interpolate_2d(*values)

    names = [names] if isinstance(names, str) else names
    data = [_interp_field(name) for name in names]
    return data[0] if len(data) == 1 else data
