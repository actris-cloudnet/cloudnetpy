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


def get_source(data_handler):
    """Returns uuid (or filename if uuid not found) of the source file."""
    return getattr(data_handler.dataset, 'file_uuid', data_handler.filename)


def read_nc_fields(nc_file, field_names):
    """Reads selected variables from a netCDF file and returns as a list."""
    nc_variables = netCDF4.Dataset(nc_file).variables
    return [nc_variables[name][:] for name in field_names]


def interpolate_model_field(categorize_file, variable_name):
    """Interpolates 2D model field into dense Cloudnet grid.

    Args:
        categorize_file (str): Categorize file name.
        variable_name (str): Model variable to be interpolated, e.g. 'temperature'.

    Returns:
        ndarray: 2D model field interpolated to dense Cloudnet grid.

    """
    data = read_nc_fields(categorize_file, ['model_time', 'model_height',
                                            variable_name, 'time', 'height'])
    return utils.interpolate_2d(*data)
