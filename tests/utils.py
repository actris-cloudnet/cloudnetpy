import os
import netCDF4
import configparser
import logging


def init_logger(file_name, fname):
    logging.basicConfig(format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s',
                        filename=f'{fname}',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M',
                        filemode='a')

    file_type = get_file_type(file_name)
    site, date = get_site_info(file_name)

    logging.root.name = f"{site} - {date} - {file_type}"


def fill_log(message, reason):
    logging.LoggerAdapter(logging.getLogger(__file__), {"name": 'Mace Head', "date": '2019-05-17'})
    logging.warning(message)
    logging.warning(reason)


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


def get_site_info(file_name):
    def _generate_site_name():
        try:
            return netCDF4.Dataset(file_name).getncattr('location')
        except AttributeError:
            return "None"

    def _generate_date():
        try:
            year = netCDF4.Dataset(file_name).getncattr('year')
            month = netCDF4.Dataset(file_name).getncattr('month')
            day = netCDF4.Dataset(file_name).getncattr('day')
            return f"{year}-{month}-{day}"
        except AttributeError:
            return "None"
    return _generate_site_name(), _generate_date()

