from tests.utils import fill_log
# Test functions for metadata testing


def test_variable_keys(missing_variables):
    assert not missing_variables, \
        fill_log(f"Variables are missing", missing_variables)


def test_variable_units(variable):
    assert not variable.bad_units, \
        fill_log(f"Wrong variable units",variable.bad_units )


def test_variable_values(variable):
    assert not variable.bad_values, \
        fill_log(f"Variable values not withing the limits", variable.bad_values)


def test_attribute_keys(missing_global_attributes):
    assert not missing_global_attributes, \
        fill_log(f"Attributes are missing", missing_global_attributes)


def test_attribute_values(global_attribute):
    assert not global_attribute.bad_values, \
        fill_log(f"Attribute values not withing the limits", global_attribute.bad_values)

