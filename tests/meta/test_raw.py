"""Tests for raw radar/lidar files, and model / hatpro files."""
import pytest


class TestMiraRaw:

    def test_variables(self, variable_names):
        assert not variable_names

    def test_variables_units(self, variable):
        assert not variable.unit

    def test_variables_values(self, variable):
        assert not variable.value

    def test_attributes(self, global_attribute_names):
        assert not global_attribute_names

    def test_attributes_units(self, global_attribute):
        assert not global_attribute.unit

    def test_attributes_values(self, global_attribute):
        assert not global_attribute.value


class TestChm15kRaw:

    def test_variables(self, variable_names):
        assert not variable_names

    def test_variables_units(self, variable):
        assert not variable.unit

    def test_variables_values(self, variable):
        assert not variable.value

    def test_attributes(self, global_attribute_names):
        assert not global_attribute_names

    def test_attributes_units(self, global_attribute):
        assert not global_attribute.unit

    def test_attributes_values(self, global_attribute):
        assert not global_attribute.value


class TestHatpro:

    def test_variables(self, variable_names):
        assert not variable_names

    def test_attributes(self, global_attribute_names):
        assert not global_attribute_names


class TestEcmwf:

    def test_variables(self, variable_names):
        assert not variable_names

    def test_attributes(self, global_attribute_names):
        assert not global_attribute_names
