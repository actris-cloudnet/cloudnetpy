import os
import configparser
import numpy as np
import netCDF4

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class Quality:

    def __init__(self, filename: str):
        self.n_metadata_tests = 0
        self.n_metadata_test_failures = 0
        self.n_data_tests = 0
        self.n_data_test_failures = 0
        self._nc = netCDF4.Dataset(filename)
        self._metadata_config = _read_config(f'{FILE_PATH}/metadata_config.ini')
        self._data_config = _read_config(f'{FILE_PATH}/data_quality_config.ini')

    def check_metadata(self) -> dict:
        return {
            'missing_variables': self._find_missing_keys('required_variables'),
            'missing_global_attributes': self._find_missing_keys('required_global_attributes'),
            'invalid_global_attribute_values': self._find_invalid_global_attribute_values(),
            'invalid_units': self._find_invalid_variable_units()}

    def check_data(self) -> dict:
        return {'data_out_of_bounds': self._find_invalid_data_values()}

    def close(self) -> None:
        self._nc.close()

    def _find_invalid_data_values(self) -> list:
        invalid = []
        for var, limits in self._data_config.items('limits'):
            if var in self._nc.variables:
                self.n_data_tests += 1
                limits = tuple(map(float, limits.split(',')))
                max_value = np.max(self._nc.variables[var][:])
                min_value = np.min(self._nc.variables[var][:])
                if min_value < limits[0] or max_value > limits[1]:
                    invalid.append(var)
                    self.n_data_test_failures += 1
        return invalid

    def _find_invalid_global_attribute_values(self) -> list:
        invalid = []
        for key, limits in self._metadata_config.items('attribute_limits'):
            if hasattr(self._nc, key):
                self.n_metadata_tests += 1
                limits = tuple(map(float, limits.split(',')))
                if not limits[0] <= int(self._nc.getncattr(key)) <= limits[1]:
                    invalid.append(key)
                    self.n_metadata_test_failures += 1
        return invalid

    def _find_invalid_variable_units(self) -> list:
        invalid = []
        for key, expected_unit in self._metadata_config.items('variable_units'):
            if key in self._nc.variables:
                self.n_metadata_tests += 1
                if self._nc.variables[key].units != expected_unit:
                    invalid.append(key)
                    self.n_metadata_test_failures += 1
        return invalid

    def _find_missing_keys(self, config_section: str) -> list:
        nc_keys = self._nc.ncattrs() if 'attr' in config_section else self._nc.variables.keys()
        config_keys = self._read_config_keys(config_section)
        missing_keys = list(set(config_keys) - set(nc_keys))
        self.n_metadata_tests += len(config_keys)
        self.n_metadata_test_failures += len(missing_keys)
        return missing_keys

    def _read_config_keys(self, config_section: str) -> np.ndarray:
        field = 'all' if 'attr' in config_section else self._nc.cloudnet_file_type
        keys = self._metadata_config[config_section][field].split(',')
        return np.char.strip(keys)


def _read_config(filename: str) -> configparser.ConfigParser:
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(filename)
    return conf
