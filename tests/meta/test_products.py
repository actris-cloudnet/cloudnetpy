"""Tests for CloudnetPy product files."""
from collections import namedtuple
import pytest
from cloudnetpy import utils
import numpy.ma as ma
from tests.utils import Specs


COMMON_ATTRIBUTES = {
    'year': Specs(
        min=2000,
        max=int(utils.get_time()[:4]),
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
    'cloudnet_file_type': Specs(),
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
        'target_classification': Specs(),
        'detection_status': Specs(),
    },
    'iwc': {
        'iwc': Specs(units='kg m-3'),
        'iwc_error': Specs(units='dB'),
        'iwc_bias': Specs(units='dB'),
        'iwc_sensitivity': Specs(units='kg m-3'),
        'iwc_inc_rain': Specs(units='kg m-3'),
        'iwc_retrieval_status': Specs(),
    },
    'lwc': {
        'lwc': Specs(units='kg m-3'),
        'lwc_error': Specs(units='dB'),
        'lwp': Specs(units='g m-2'),
        'lwp_error': Specs(units='g m-2'),
        'lwc_retrieval_status': Specs(),
    },
    'drizzle': {
        'Do': Specs(),
        'mu': Specs(),
    }
}


@pytest.mark.product
@pytest.mark.radar
class TestCommonAttributes:
    keys = COMMON_ATTRIBUTES.keys()
    fixture = 'global_attribute'

    def test_common_attributes(self, global_attribute_names):
        assert not self.keys - global_attribute_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_common_attribute_values(self, global_attribute):
        assert global_attribute.value is not None

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_common_attribute_max_values(self, global_attribute):
        spec = COMMON_ATTRIBUTES[global_attribute.name].max
        if spec is not None:
            assert int(global_attribute.value) <= spec

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_common_attribute_min_values(self, global_attribute):
        spec = COMMON_ATTRIBUTES[global_attribute.name].min
        if spec is not None:
            assert int(global_attribute.value) >= spec


@pytest.mark.product
class TestCommonVariables:
    keys = COMMON_VARIABLES.keys()
    fixture = 'variable'

    def test_variables(self, variable_names):
        assert not self.keys - variable_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_common_variable_min_values(self, variable):
        spec = COMMON_VARIABLES[variable.name].min
        if spec is not None:
            assert ma.min(variable.value) >= spec

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_common_variable_max_values(self, variable):
        spec = COMMON_VARIABLES[variable.name].max
        if spec is not None:
            assert ma.max(variable.value) <= spec

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_common_variable_unit_values(self, variable):
        spec = COMMON_VARIABLES[variable.name].units
        if spec is not None:
            assert variable.units == spec


def _product_setup(name):
    return name, PRODUCT_VARIABLES[name].keys(), 'variable'


@pytest.mark.iwc
class TestIwc:
    name, keys, fixture = _product_setup('iwc')

    def test_variables(self, variable_names):
        assert not self.keys - variable_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_unit_values(self, variable):
        spec = PRODUCT_VARIABLES[self.name][variable.name].units
        if spec is not None:
            assert variable.units == spec


@pytest.mark.lwc
class TestLwc:
    name, keys, fixture = _product_setup('lwc')

    def test_variables(self, variable_names):
        assert not self.keys - variable_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_unit_values(self, variable):
        spec = PRODUCT_VARIABLES[self.name][variable.name].units
        if spec is not None:
            assert variable.units == spec


@pytest.mark.drizzle
class TestDrizzle:
    name, keys, fixture = _product_setup('drizzle')

    def test_variables(self, variable_names):
        assert not self.keys - variable_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_unit_values(self, variable):
        spec = PRODUCT_VARIABLES[self.name][variable.name].units
        if spec is not None:
            assert variable.units == spec


@pytest.mark.classification
class TestClassification:
    name, keys, fixture = _product_setup('classification')

    def test_variables(self, variable_names):
        assert not self.keys - variable_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_unit_values(self, variable):
        spec = PRODUCT_VARIABLES[self.name][variable.name].units
        if spec is not None:
            assert variable.units == spec
