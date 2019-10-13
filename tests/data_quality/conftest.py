import pytest
from tests import utils

CONFIG = utils.read_config('data_quality/data_quality_config.ini')


@pytest.fixture
def missing_variables(pytestconfig):
    file_name = pytestconfig.option.test_file
    return utils.find_missing_keys(CONFIG, 'quantities', file_name)


@pytest.fixture
def data(pytestconfig):
    file = pytestconfig.option.test_file
    return InputData(file)


class InputData:
    def __init__(self, file):
        self.bad_values = utils.check_var_limits(CONFIG, 'limits', file)
