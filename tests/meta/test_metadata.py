# Test functions for metadata testing


def test_variables_keys(missing_variables):
    assert not missing_variables


def test_variables_units(variable):
    assert not variable.unit


def test_variables_values(variable):
    assert not variable.value


def test_attributes_keys(missing_global_attributes):
    assert not missing_global_attributes


def test_attributes_units(global_attribute):
    assert not global_attribute.unit


def test_attributes_values(global_attribute):
    assert not global_attribute.value

