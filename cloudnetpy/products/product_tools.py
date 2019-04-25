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


def get_correct_dimensions(nc_file, field_names):
    """ Check if "model"-dimension exist. if not
        change model-dimension to normal dimension
    """
    variables = netCDF4.Dataset(nc_file).variables
    for i, name in enumerate(field_names):
        if name not in variables:
            field_names[i] = name.split('_')[-1]
    return field_names


def read_nc_fields(nc_file, field_names):
    """Reads selected variables from a netCDF file and returns as a list."""
    nc_variables = netCDF4.Dataset(nc_file).variables
    return [nc_variables[name][:] for name in field_names]
