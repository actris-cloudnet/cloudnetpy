import os
from collections import namedtuple
import netCDF4
import configparser


CONFIG_FILE_META = 'meta/metadata_config.ini'
CONFIG_FILE_DATA = 'data_quality/data_quality_config.ini'
FIELDS = ('min', 'max', 'units')
Specs = namedtuple('Specs', FIELDS)
Specs.__new__.__defaults__ = (None,) * len(Specs._fields)


def read_meta_config():
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(CONFIG_FILE_META)
    return conf


def read_data_config():
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(CONFIG_FILE_DATA)
    return conf


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
