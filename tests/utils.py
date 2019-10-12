import os
import netCDF4
import configparser
import logging
import numpy as np


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


def find_missing_keys(config, config_field, file_name):
    nc = netCDF4.Dataset(file_name)
    file_type = get_file_type(file_name)
    keys = nc.ncattrs() if 'attributes' in config_field else nc.variables.keys()
    nc.close()
    try:
        config_keys = read_config_keys(config, config_field, file_type)
        missing = set(config_keys) - set(keys)
    except:
        missing = False
    return missing


def check_var_limits(config, config_field, file_name):
    bad = {}
    nc = netCDF4.Dataset(file_name)
    keys = nc.variables.keys()
    for var, limits in config.items(config_field):
        if var in keys:
            limits = tuple(map(float, limits.split(',')))
            min_value = np.min(nc.variables[var][:])
            max_value = np.max(nc.variables[var][:])
            if min_value < limits[0] or max_value > limits[1]:
                bad[var] = [min_value, max_value]
    nc.close()
    return bad


def read_config_keys(config, field, file_type):
    keys = config[field][file_type].split(',')
    return np.char.strip(keys)


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
