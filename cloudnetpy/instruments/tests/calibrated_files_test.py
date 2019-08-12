import pytest
import netCDF4
from tests.test import initialize_test_data
import warnings

warnings.filterwarnings("ignore")

instrument_list = ['radar', 'ceilo']
test_data_path = initialize_test_data(instrument_list)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(instrument_list):
        keys = list(netCDF4.Dataset(test_data_path[i]).variables.keys())
        key_dict[instrument] = keys
    return key_dict

@pytest.mark.filterwarnings("ignore:test_radar_file")
def test_radar_file(test_data):
    must_keys = ['Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft', 'prf',
                 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg']
    compared_list = list(set(must_keys).intersection(test_data['radar']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise AssertionError(f"Missing '{', '.join(missing_variables)}' variables in Radar file")

@pytest.mark.filterwarnings("ignore:test_ceilo_file")
def test_ceilo_file(test_data):
    must_keys = ['beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height', 'wavelength']
    compared_list = list(set(must_keys).intersection(test_data['ceilo']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise AssertionError(f"Missing '{', '.join(missing_variables)}' variables in Lidar file")

