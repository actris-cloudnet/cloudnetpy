import netCDF4
import configparser
import logging
import functools
from pathlib import Path


def log_errors(func):
    """Decorator to generalize log-writing in tests."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except AssertionError as error:
            problem_variables = str(error).rpartition('assert not ')[-1]
            fill_log(func.__name__, problem_variables)
            raise
    return wrapper


def fill_log(test_function, problem_variables):
    logging.warning(f"{test_function} - {problem_variables}")


def init_logger(test_file_name, log_file_name):
    if not log_file_name:
        return
    logging.basicConfig(filename=f'{log_file_name}',
                        format='%(asctime)s - %(name)s - %(message)s',
                        level=logging.DEBUG,
                        filemode='a')
    file_type = get_file_type(test_file_name)
    #site, date = get_site_info(test_file_name)
    logging.root.name = f"{test_file_name} - {file_type}"


def read_config(config_file):
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(config_file)
    return conf


def get_test_path():
    return Path(__file__).parent


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


def get_site_info(file_name):
    def _generate_site_name():
        try:
            return netCDF4.Dataset(file_name).getncattr('location')
        except AttributeError:
            return None

    def _generate_date():
        try:
            year = netCDF4.Dataset(file_name).getncattr('year')
            month = netCDF4.Dataset(file_name).getncattr('month')
            day = netCDF4.Dataset(file_name).getncattr('day')
            return f"{year}-{month}-{day}"
        except AttributeError:
            return None
    return _generate_site_name(), _generate_date()
