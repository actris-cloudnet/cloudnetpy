import pytest
import netCDF4
from tests.test import initialize_test_data
import warnings

warnings.filterwarnings("ignore")

instrument_list = ['mira_raw', 'chm15k_raw', 'hatpro', 'ecmwf']
test_data_path = initialize_test_data(instrument_list)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(instrument_list):
        keys = list(netCDF4.Dataset(test_data_path[i]).variables.keys())
        key_dict[instrument] = keys
    return key_dict


def test_mira_raw_file(test_data):
    must_keys = ['nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg', 'NyquistVelocity',
                 'time', 'range', 'Zg', 'VELg', 'RMSg', 'LDRg', 'SNRg']
    compared_list = list(set(must_keys).intersection(test_data['mira_raw']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in raw Radar file")


def test_ceilo_raw_file(test_data):
    must_keys = ['latitude', 'longitude', 'azimuth', 'zenith', 'time', 'range',
                 'layer', 'altitude', 'wavelength', 'laser_pulses', 'beta_raw']
    compared_list = list(set(must_keys).intersection(test_data['chm15k_raw']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in raw Lidar file")


def test_model_file(test_data):
    must_keys = ['latitude', 'longitude', 'horizontal_resolution', 'time', 'forecast_time',
                 'level', 'pressure', 'uwind', 'vwind', 'omega', 'temperature',
                 'q', 'rh']
    compared_list = list(set(must_keys).intersection(test_data['ecmwf']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in Model file")


def test_mwr_file(test_data):
    must_keys = ['time_reference', 'minimum', 'maximum', 'time', 'rain_flag',
                 'elevation_angle','azimuth_angle', 'retrieval', 'LWP_data']
    compared_list = list(set(must_keys).intersection(test_data['hatpro']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in Microwave radiometer file")

