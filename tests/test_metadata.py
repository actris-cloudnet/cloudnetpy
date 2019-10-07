# Test functions for metadata testing


def test_variables_keys(variable_names):
    m = variable_names.missing
    assert not variable_names


def test_variables_units(variable):
    assert not variable.unit


def test_variables_values(variable):
    assert not variable.value


def test_attributes_keys(global_attribute_names):
    m = global_attribute_names.missing
    assert not global_attribute_names.missing, print(m)


def test_attributes_units(global_attribute):
    assert not global_attribute.unit


def test_attributes_values(global_attribute):
    assert not global_attribute.value

