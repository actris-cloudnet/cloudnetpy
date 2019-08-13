import pytest
import netCDF4
from tests.test import initialize_test_data

INSTRUMENT_LIST = ['radar', 'ceilo']
TEST_DATA_PATH = initialize_test_data(INSTRUMENT_LIST)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(INSTRUMENT_LIST):
        keys = list(netCDF4.Dataset(TEST_DATA_PATH[i]).variables.keys())
        key_dict[instrument] = keys
    return key_dict


def _error_msg(missing_keys, name):
    return f"Variable(s) {missing_keys} missing in {name} file!"


def test_radar_file(test_data):
    must_keys = {'Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft', 'prf',
                 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg'}
    missing_keys = must_keys - test_data['radar']
    assert not missing_keys, _error_msg(missing_keys, 'radar')


def test_ceilo_file(test_data):
    must_keys = {'beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height', 'wavelength'}
    missing_keys = must_keys - test_data['ceilo']
    assert not missing_keys, _error_msg(missing_keys, 'ceilo')

