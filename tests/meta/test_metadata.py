from tests.utils import fill_log
# Test functions for metadata testing


def test_variables(missing_variables):
    assert not missing_variables, \
        fill_log(f"Variables are missing", missing_variables)


def test_global_attributes(missing_global_attributes):
    assert not missing_global_attributes, \
        fill_log(f"Attributes are missing", missing_global_attributes)


def test_variable_units(invalid_variable_units):
    assert not invalid_variable_units, \
        fill_log(f"Wrong variable units", invalid_variable_units)


def test_attribute_values(invalid_global_attribute_values):
    assert not invalid_global_attribute_values, \
        fill_log(f"Variable values not within the limits", invalid_global_attribute_values)
