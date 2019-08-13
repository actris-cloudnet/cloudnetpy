import pytest
import netCDF4
from tests.test import initialize_test_data, missing_var_msg

INSTRUMENT_LIST = ['mira_raw', 'chm15k_raw', 'hatpro', 'ecmwf']
TEST_DATA_PATH = initialize_test_data(INSTRUMENT_LIST)


@pytest.fixture
def test_data():
    key_dict = {}
    for path, instrument in zip(TEST_DATA_PATH, INSTRUMENT_LIST):
        key_dict[instrument] = set(netCDF4.Dataset(path).variables.keys())
    return key_dict


def test_mira_raw_file(test_data):
    must_keys = {'nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg', 'NyquistVelocity',
                 'time', 'range', 'Zg', 'VELg', 'RMSg', 'LDRg', 'SNRg'}
    missing_keys = must_keys - test_data['mira_raw']
    assert not missing_keys, missing_var_msg(missing_keys, 'raw mira')


def test_ceilo_raw_file(test_data):
    must_keys = {'latitude', 'longitude', 'azimuth', 'zenith', 'time', 'range',
                 'layer', 'altitude', 'wavelength', 'laser_pulses', 'beta_raw'}
    missing_keys = must_keys - test_data['chm15k_raw']
    assert not missing_keys, missing_var_msg(missing_keys, 'raw lidar')


def test_model_file(test_data):
    must_keys = {'latitude', 'longitude', 'horizontal_resolution', 'time', 'forecast_time',
                 'level', 'pressure', 'uwind', 'vwind', 'omega', 'temperature',
                 'q', 'rh'}
    missing_keys = must_keys - test_data['ecmwf']
    assert not missing_keys, missing_var_msg(missing_keys, 'ecmwf model')


def test_mwr_file(test_data):
    must_keys = {'time_reference', 'minimum', 'maximum', 'time', 'rain_flag',
                 'elevation_angle', 'azimuth_angle', 'retrieval', 'LWP_data'}
    missing_keys = must_keys - test_data['hatpro']
    assert not missing_keys, missing_var_msg(missing_keys, 'hatpro')
