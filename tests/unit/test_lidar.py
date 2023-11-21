import netCDF4
import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_equal

from cloudnetpy.categorize.lidar import Lidar, get_gap_ind

WAVELENGTH = 900.0


@pytest.fixture(scope="session")
def fake_lidar_file(tmpdir_factory):
    """Creates a simple lidar file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("radar_file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        n_time, n_height = 4, 4
        root_grp.createDimension("time", n_time)
        root_grp.createDimension("height", n_height)
        root_grp.createVariable("time", "f8", "time")[:] = np.arange(n_time)
        var = root_grp.createVariable("height", "f8", "height")
        var[:] = np.arange(n_height)
        var.units = "km"
        root_grp.createVariable("wavelength", "f8")[:] = WAVELENGTH
        root_grp.createVariable("latitude", "f8")[:] = 60.43
        root_grp.createVariable("longitude", "f8")[:] = 25.4
        var = root_grp.createVariable("altitude", "f8")
        var[:] = 120.3
        var.units = "m"
        var = root_grp.createVariable("beta", "f8", ("time", "height"))
        var[:] = ma.array(
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            dtype=float,
            mask=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        )
    return file_name


def test_init(fake_lidar_file):
    obj = Lidar(fake_lidar_file)
    assert obj.data["lidar_wavelength"].data == WAVELENGTH
    assert obj.data["beta_bias"].data == 3
    assert obj.data["beta_error"].data == 0.5


def test_rebin(fake_lidar_file):
    obj = Lidar(fake_lidar_file)
    time_new = np.array([1.1, 2.1])
    height_new = np.array([505, 1501])
    ind = obj.interpolate_to_grid(time_new, height_new)
    result = np.array([[2, 3], [2, 3]])
    assert_array_equal(obj.data["beta"].data, result)
    assert ind == [0, 1]


@pytest.mark.parametrize("original_grid, new_grid, threshold, expected", [
    (np.array([1, 2, 3, 4, 5]), np.array([1.9, 2.2, 3.1, 4.0, 4.9]), 1, []),
    (np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50]), 1, [0, 1, 2, 3, 4]),
    (np.array([1, 2, 3, 4, 5]), np.array([1.1, 2.1, 3.2, 4.2, 5.3]), 0.15, [2, 3, 4]),
    (np.array([]), np.array([]), 0.5, [])
])
def test_get_gap_ind(
        original_grid: np.ndarray,
        new_grid: np.ndarray,
        threshold: float,
        expected: np.ndarray):
    assert get_gap_ind(original_grid, new_grid, threshold) == expected
