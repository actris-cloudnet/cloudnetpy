import netCDF4
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cloudnetpy.categorize.mwr import Mwr


@pytest.fixture(scope="session")
def fake_mwr_file(tmpdir_factory):
    """Creates a simple mwr for testing."""
    file_name = tmpdir_factory.mktemp("data").join("mwr_file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        n_time = 5
        root_grp.createDimension("time", n_time)
        var = root_grp.createVariable("time", "f8", "time")
        var[:] = np.arange(n_time)
        var = root_grp.createVariable("lwp", "f8", "time")
        var[:] = np.array([0.1, 2.5, -0.1, 0.2, 0.0])
        var.units = "g / m^2"
    return file_name


def test_rebin_to_grid(fake_mwr_file):
    obj = Mwr(fake_mwr_file)
    time_new = np.array([1.4, 2.8])
    obj.rebin_to_grid(time_new)
    result = np.array([1.2, 0.2])
    assert_array_equal(obj.data["lwp"][:], result)


@pytest.fixture(scope="session")
def bad_mwr_file(tmpdir_factory):
    """Creates invalid mwr file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("mwr_file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        root_grp.createDimension("time", 5)
        var = root_grp.createVariable("xxx", "f8", "time")
        var[:] = np.array([0.1, 2.5, -0.1, 0.2, 0.0])
    return file_name


def test_init_lwp_data(fake_mwr_file):
    obj = Mwr(fake_mwr_file)
    result = np.array([0.1, 2.5, -0.1, 0.2, 0.0])
    assert_array_equal(obj.data["lwp"][:], result)


def test_init_lwp_error(fake_mwr_file):
    obj = Mwr(fake_mwr_file)
    lwp = obj.data["lwp"][:]
    lwp_error = obj.data["lwp_error"][:]
    assert_array_equal(lwp_error, np.sqrt((0.25 * lwp) ** 2 + 0.02**2))


def test_missing_variable(bad_mwr_file):
    with pytest.raises(KeyError):
        Mwr(bad_mwr_file)
