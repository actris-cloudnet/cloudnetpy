import netCDF4
import numpy as np
import pytest
from numpy import ma

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.cloudnetarray import CloudnetArray


@pytest.fixture()
def dummy_nc_file(tmp_path):
    """Creates a minimal netCDF file for NcRadar."""
    file_path = tmp_path / "radar.nc"
    with netCDF4.Dataset(file_path, "w") as nc:
        nc.createDimension("time", 10)
        nc.createVariable("time", "f8", ("time",))[:] = np.arange(10)
    return file_path


def _create_radar(dummy_nc_file, elevation, azimuth):
    """Creates NcRadar with given elevation and azimuth arrays."""
    radar = NcRadar(dummy_nc_file, site_meta={"name": "Test"})
    radar.data["elevation"] = CloudnetArray(ma.array(elevation), "elevation")
    radar.data["azimuth_angle"] = CloudnetArray(ma.array(azimuth), "azimuth_angle")
    return radar


DEFAULTS = {
    "elevation_threshold": 1.1,
    "elevation_diff_threshold": 1e-6,
    "azimuth_diff_threshold": 1e-3,
}


class TestZenithAzimuthAnglesNormal:
    """Test with valid elevation and azimuth data."""

    def test_all_stable(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 90.0)
        azimuth = np.full(n, 180.0)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert all(result)
        assert len(result) == n

    def test_varying_elevation_filtered(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 90.0)
        elevation[3] = 85.0
        azimuth = np.full(n, 180.0)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert not result[3]
        assert sum(result) < n

    def test_varying_azimuth_filtered(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 90.0)
        azimuth = np.full(n, 180.0)
        azimuth[5] = 190.0
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert not result[5]


class TestZenithAzimuthAnglesAllMaskedAzimuth:
    """Test with all azimuth values masked."""

    def test_stable_elevation_all_masked_azimuth(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 90.0)
        azimuth = ma.masked_all(n)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert all(result)

    def test_varying_elevation_all_masked_azimuth(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 90.0)
        elevation[2] = 85.0
        azimuth = ma.masked_all(n)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert not result[2]


class TestZenithAzimuthAnglesAllMasked:
    """Test with both elevation and azimuth fully masked."""

    def test_all_masked(self, dummy_nc_file):
        n = 10
        elevation = ma.masked_all(n)
        azimuth = ma.masked_all(n)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert all(result)


class TestZenithAzimuthAnglesPartiallyMasked:
    """Test with some masked values."""

    def test_some_masked_elevation(self, dummy_nc_file):
        n = 10
        elevation = ma.array(np.full(n, 90.0))
        elevation[4] = ma.masked
        azimuth = np.full(n, 180.0)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert result[4]

    def test_some_masked_azimuth(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 90.0)
        azimuth = ma.array(np.full(n, 180.0))
        azimuth[4] = ma.masked
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(**DEFAULTS)
        assert result[4]


class TestZenithAzimuthAnglesTooFewProfiles:
    """Test error when too few valid profiles remain."""

    def test_all_unstable_raises(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 45.0)
        azimuth = np.full(n, 180.0)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        with pytest.raises(ValidTimeStampError):
            radar.add_zenith_and_azimuth_angles(**DEFAULTS)


class TestZenithAzimuthAnglesOffsets:
    """Test zenith and azimuth offset parameters."""

    def test_zenith_offset(self, dummy_nc_file):
        n = 10
        elevation = np.full(n, 89.5)
        azimuth = np.full(n, 180.0)
        radar = _create_radar(dummy_nc_file, elevation, azimuth)
        result = radar.add_zenith_and_azimuth_angles(
            **DEFAULTS,
            zenith_offset=-0.5,
        )
        assert all(result)
        assert "zenith_offset" in radar.data
