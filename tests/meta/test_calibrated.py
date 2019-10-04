"""Tests for calibrated CloudnetPy files."""
from tests.utils import Specs
import pytest

RADAR_VARIABLES = {
    'Ze': Specs(units='dBZ'),
    'v': Specs(units='m s-1'),
    'width': Specs(units='m s-1'),
    'ldr': Specs(units='dB'),
    'latitude': Specs(units='degrees_north'),
    'longitude': Specs(units='degrees_east'),
    'altitude': Specs(units='m'),
    'time': Specs(units='decimal hours since midnight'),
    'radar_frequency': Specs('GHz')
}

LIDAR_VARIABLES = {
    'beta': Specs(units='sr-1 m-1'),
    'beta_raw': Specs(units='sr-1 m-1'),
    'beta_smooth': Specs(units='sr-1 m-1'),
    'range': Specs(units='m'),
    'time': Specs(units='decimal hours since midnight'),
    'wavelength': Specs(units='nm'),
    'height': Specs(units='m'),
}


@pytest.mark.radar
class TestRadar:
    fixture = 'variable'
    keys = RADAR_VARIABLES.keys()

    def test_variables(self, variable_names):
        assert not self.keys - variable_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_unit_values(self, variable):
        spec = RADAR_VARIABLES[variable.name].units
        if spec is not None:
            assert variable.units == spec


@pytest.mark.lidar
class TestLidar:
    fixture = 'variable'
    keys = LIDAR_VARIABLES.keys()

    def test_variables(self, variable_names):
        assert not self.keys - variable_names

    @pytest.mark.parametrize(fixture, keys, indirect=True)
    def test_unit_values(self, variable):
        spec = LIDAR_VARIABLES[variable.name].units
        if spec is not None:
            assert variable.units == spec
