"""Tests for CloudnetPy product files."""
from collections import namedtuple
from tests.test import read_attribute_names, read_variable_names, read_attribute, read_variable
from cloudnetpy import utils
import numpy as np

CURRENT_YEAR = int(utils.get_time()[:4])

FIELDS = ('min', 'max', 'units', 'attributes',)
Specs = namedtuple('Specs', FIELDS, defaults=(None,)*len(FIELDS))

COMMON_ATTRIBUTES = {
    'year': Specs(
        min=2000,
        max=CURRENT_YEAR,
    ),
    'month': Specs(
        min=1,
        max=12,
    ),
    'day': Specs(
        min=1,
        max=31,
    ),
    'file_uuid': Specs(),
    'cloudnetpy_version': Specs(),
    'Conventions': Specs(),
    'location': Specs(),
    'history': Specs(),
    'title': Specs(),
    'source': Specs(),
}

COMMON_VARIABLES = {
    'height': Specs(
        min=0,
        max=20000,
        units='m',
    ),
    'time': Specs(
        min=0,
        max=24,
        units='decimal hours since midnight',
    ),
    'altitude': Specs(
        min=0,
        max=8000,
        units='m',
    ),
    'latitude': Specs(
        min=-90,
        max=90,
        units='degrees_north',
    ),
    'longitude': Specs(
        min=-180,
        max=180,
        units='degrees_east',
    ),
}

PRODUCT_VARIABLES = {
    'classification': {
        'target_classification': Specs(
            min=0,
            max=10,
        ),
        'detection_status': Specs(
            min=0,
            max=7,
        ),
    },
    'iwc': {
        'iwc': Specs(
            min=0,
            max=1,
            units='kg m-3',
        ),
        'iwc_error': Specs(
            min=0,
            max=10,
            units='dB',
        ),
        'iwc_bias': Specs(
            min=0,
            max=1,
            units='dB',
        ),
        'iwc_sensitivity': Specs(
            min=0,
            max=1,
            units='kg m-3'
        ),
        'iwc_inc_rain': Specs(
            min=0,
            max=1,
            units='kg m-3',
        ),
        'iwc_retrieval_status': Specs(
            min=0,
            max=7,
        ),
    }

}

PRODUCTS = ['classification', 'drizzle', 'iwc', 'lwc']


def test_common_attributes():
    """Tests that each product file has correct global attributes."""
    for identifier in PRODUCTS:
        for attr_name in COMMON_ATTRIBUTES:
            attr_value = _check_attribute_existence(identifier, attr_name)
            _check_min(COMMON_ATTRIBUTES[attr_name].min, attr_value)
            _check_max(COMMON_ATTRIBUTES[attr_name].max, attr_value)


def test_common_variables():
    """Tests that each product file has correct common variables."""
    for identifier in PRODUCTS:
        _test_variables(COMMON_VARIABLES, identifier)


def test_product_variables():
    """Tests that each product file has correct product-dependent variables."""
    for identifier in PRODUCT_VARIABLES:
        _test_variables(PRODUCT_VARIABLES[identifier], identifier)


def _test_variables(var_dict, identifier):
    for var_name in var_dict:
        var_value = _check_variable_existence(identifier, var_name)
        _check_variable_fields(var_dict[var_name], var_value)


def _check_variable_existence(identifier, var_name):
    test_file_variables = read_variable_names(identifier)
    _check_existence(test_file_variables, var_name)
    return read_variable(identifier, var_name)


def _check_attribute_existence(identifier, attr_name):
    test_file_attributes = read_attribute_names(identifier)
    _check_existence(test_file_attributes, attr_name)
    return read_attribute(identifier, attr_name)


def _check_existence(names_in_file, name):
    assert name in names_in_file


def _check_variable_fields(spec, var_value):
    _check_min(spec.min, var_value[:])
    _check_max(spec.max, var_value[:])
    _check_unit(spec.units, var_value)


def _check_min(spec_value, value):
    if spec_value:
        scalar_float = _get_scalar_float(value, np.min)
        assert scalar_float >= spec_value


def _check_max(spec_value, value):
    if spec_value:
        scalar_float = _get_scalar_float(value, np.max)
        assert scalar_float <= spec_value


def _check_unit(spec_value, variable):
    if spec_value:
        assert getattr(variable, 'units') == spec_value


def _get_scalar_float(value, fun=None):
    if isinstance(value, np.ndarray):
        return fun(value)
    else:
        return float(value)
