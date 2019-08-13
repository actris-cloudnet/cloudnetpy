import pytest
import netCDF4
from tests.test import initialize_test_data, missing_var_msg

INSTRUMENT_LIST = ['radar', 'ceilo']
TEST_DATA_PATH = initialize_test_data(INSTRUMENT_LIST)


@pytest.fixture
def test_data():
    key_dict = {}
    for path, instrument in zip(TEST_DATA_PATH, INSTRUMENT_LIST):
        key_dict[instrument] = set(netCDF4.Dataset(path).variables.keys())
    return key_dict


def test_radar_file(test_data):
    must_keys = {'Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft',
                 'prf', 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg'}
    missing_keys = must_keys - test_data['radar']
    assert not missing_keys, missing_var_msg(missing_keys, 'calibrated mira')


def test_ceilo_file(test_data):
    must_keys = {'beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height', 'wavelength'}
    missing_keys = must_keys - test_data['ceilo']
    assert not missing_keys, missing_var_msg(missing_keys, 'calibrated ceilo')


