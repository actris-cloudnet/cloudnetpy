import pytest
import netCDF4
from tests.test import initialize_test_data, missing_var_msg

instrument_list = ['radar', 'ceilo']
test_data_path = initialize_test_data(instrument_list)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(instrument_list):
        keys = list(netCDF4.Dataset(test_data_path[i]).variables.keys())
        key_dict[instrument] = set(keys)
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


