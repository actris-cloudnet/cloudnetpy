from tests.utils import fill_log
# Test functions for data testing


def test_variable_values(data):
    assert not data.bad_values, fill_log("Data not withing limits", data.bad_values)
