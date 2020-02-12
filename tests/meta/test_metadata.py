from tests.utils import log_errors
# Test functions for metadata testing


@log_errors
def test_variables(missing_variables):
    assert not missing_variables


@log_errors
def test_global_attributes(missing_global_attributes):
    assert not missing_global_attributes


@log_errors
def test_variable_units(invalid_variable_units):
    assert not invalid_variable_units


@log_errors
def test_attribute_values(invalid_global_attribute_values):
    assert not invalid_global_attribute_values
