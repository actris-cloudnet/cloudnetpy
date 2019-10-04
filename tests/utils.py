import os
from collections import namedtuple
import netCDF4


FIELDS = ('min', 'max', 'units')
Specs = namedtuple('Specs', FIELDS)
Specs.__new__.__defaults__ = (None,) * len(Specs._fields)


def get_test_path():
    return os.path.dirname(os.path.abspath(__file__))


def get_file_type(file_name):
    """Returns type of Cloudnet file."""
    nc = netCDF4.Dataset(file_name)
    if hasattr(nc, 'cloudnet_file_type'):
        file_type = nc.cloudnet_file_type
    elif hasattr(nc, 'radiometer_system'):
        file_type = 'hatpro'
    elif hasattr(nc, 'title') and 'mira' in nc.title.lower():
        file_type = 'mira_raw'
    elif hasattr(nc, 'source'):
        if 'chm' in nc.source.lower():
            file_type = 'chm15k_raw'
        elif 'ecmwf' in nc.source.lower():
            file_type = 'ecmwf'
    else:
        raise ValueError('Unknown file.')
    nc.close()
    return file_type
