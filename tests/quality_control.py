#!../venv/bin/python3.7
import os
import sys
import pytest
import configparser
import logging


CONFIG_FILE_NAME = 'data_quality_config.ini'


def read_config():
    conf = configparser.ConfigParser()
    conf.read(CONFIG_FILE_NAME)
    return conf


def get_test_file_name(config):
    return config['misc']['test_file_name']


def get_file_keys():
    config = read_config()
    file_type = get_file_type(get_test_file_name(config))
    return [[key.strip()] for key in config[file_type]['quantities'].split(',')]


def update_test_file_name(config, new_test_file_name):
    config.set('misc', 'test_file_name', new_test_file_name)
    with open(CONFIG_FILE_NAME, "w+") as file:
        config.write(file)


def init_logger(fname):
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        filename=fname,
                        level=logging.INFO,
                        datefmt='%d-%b-%y %H:%M:%S')


def get_file_type(file_name):
    """Returns type of Cloudnet file.

    This function needs work. How can we get the file type
    from the file (name or content)???
    """
    for file_type in ('categorize', 'iwc', 'lwc', 'radar', 'ceilo'):
        if file_type in os.path.basename(file_name):
            return file_type


def _get_test_path():
    return os.path.dirname(os.path.abspath(__file__))


def main(test_file, log_file):
    config = read_config()
    update_test_file_name(config, test_file)
    init_logger(log_file)
    logging.info(f"Testing: {test_file}")
    pytest.main(["--tb=line", _get_test_path()])


if __name__ == "__main__":
    main(*sys.argv[1:])
