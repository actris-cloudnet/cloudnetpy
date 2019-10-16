from tests.utils import fill_log
# Test functions for data quality testing


def test_min_max(out_of_limits_values):
    assert not out_of_limits_values,\
        fill_log("Data not within limits", out_of_limits_values)


def test_invalid(invalid_values):
    assert not invalid_values, \
        fill_log("Eather NaNs or Inf in data", invalid_values)

