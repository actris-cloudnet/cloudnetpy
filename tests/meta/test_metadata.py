# Test functions for metadata testing


def test_variable_keys(missing_variables):
    assert not missing_variables


def test_variable_units(variable):
    assert not variable.bad_units


def test_attribute_keys(missing_global_attributes):
    assert not missing_global_attributes


def test_attribute_values(global_attribute):
    assert not global_attribute.bad_values
