"""General helper functions for all products."""
import netCDF4
import cloudnetpy.utils as utils


class CategorizeBits:
    """Class holding information about category and quality bits.

    Args:
        categorize_file (str): Categorize file name.

    """
    category_keys = ('droplet', 'falling', 'cold', 'melting', 'aerosol',
                     'insect')

    quality_keys = ('radar', 'lidar', 'clutter', 'molecular', 'attenuated',
                    'corrected')

    def __init__(self, categorize_file):
        self.variables = netCDF4.Dataset(categorize_file).variables
        self.category_bits = self._read_bits('category')
        self.quality_bits = self._read_bits('quality')

    def _read_bits(self, bit_type):
        """ Converts bitfield into dictionary."""
        bitfield = self.variables[f"{bit_type}_bits"][:]
        keys = getattr(CategorizeBits, f"{bit_type}_keys")
        return {key: utils.isbit(bitfield, i) for i, key in enumerate(keys)}


class ProductClassification(CategorizeBits):
    """Base class for creating different classifications in the child classes
    of various Cloudnet products. Child of CategorizeBits class.

    Args:
        categorize_file (str): Categorize file name.

    """
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.is_rain = self.variables['is_rain'][:]
        self.is_undetected_melting = self.variables['is_undetected_melting'][:]


def get_source(data_handler):
    """Returns uuid (or filename if uuid not found) of the source file."""
    return getattr(data_handler.dataset, 'file_uuid', data_handler.filename)


def read_nc_fields(nc_file, names):
    """Reads selected variables from a netCDF file.

    Args:
        nc_file (str): netCDF file name.
        names (str/list): Variables to be read, e.g. 'temperature' or
            ['ldr', 'lwp'].

    Returns:
        ndarray/list: Array in case of one variable passed as a string.
            List of arrays otherwise.

    """
    names = [names] if isinstance(names, str) else names
    nc = netCDF4.Dataset(nc_file)
    nc_variables = nc.variables
    data = [nc_variables[name][:] for name in names]
    nc.close()
    return data[0] if len(data) == 1 else data


def interpolate_model(cat_file, names):
    """Interpolates 2D model field into dense Cloudnet grid.

    Args:
        cat_file (str): Categorize file name.
        names (str/list): Model variable to be interpolated, e.g.
            'temperature' or ['temperature', 'pressure'].

    Returns:
        ndarray/list: Array in case of one variable passed as a string.
            List of arrays otherwise.

    """
    def _interp_field(var_name):
        values = read_nc_fields(cat_file, ['model_time', 'model_height',
                                           var_name, 'time', 'height'])
        return utils.interpolate_2d(*values)

    names = [names] if isinstance(names, str) else names
    data = [_interp_field(name) for name in names]
    return data[0] if len(data) == 1 else data
