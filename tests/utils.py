import os
import netCDF4
import configparser
import logging


def init_logger(path, fname):
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=f'{path}{fname}.log',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w')


def read_config(config_file):
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(config_file)
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
