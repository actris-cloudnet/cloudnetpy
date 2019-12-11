import numpy as np
from numpy import testing
from collections import namedtuple
import pytest
import netCDF4
import cloudnetpy.products.drizzle as drizzle
from cloudnetpy.products.drizzle import *


DIMENSIONS = ('time', 'height', 'model_time', 'model_height')
TEST_ARRAY = np.arange(3)
CategorizeBits = namedtuple('CategorizeBits', ['category_bits', 'quality_bits'])


@pytest.fixture(scope='session')
def drizzle_source_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    _create_dimensions(root_grp)
    _create_dimension_variables(root_grp)
    var = root_grp.createVariable('altitude', 'f8')
    var[:] = 1
    var.units = 'km'
    var = root_grp.createVariable('lwp', 'f8', 'time')
    var[:] = [1, 1, 0.5]
    var = root_grp.createVariable('lwp_error', 'f8', 'time')
    var[:] = [0.2, 0.2, 0.1]
    var = root_grp.createVariable('is_rain', 'i4', 'time')
    var[:] = [0, 1, 1]
    var = root_grp.createVariable('category_bits', 'i4', 'time')
    var[:] = [0, 1, 2]
    var = root_grp.createVariable('quality_bits', 'i4', 'time')
    var[:] = [8, 16, 32]
    var = root_grp.createVariable('temperature', 'f8', ('time', 'height'))
    var[:] = np.array([[282, 280, 278],
                       [286, 284, 282],
                      [284, 282, 280]])
    var = root_grp.createVariable('pressure', 'f8', ('time', 'height'))
    var[:] = np.array([[1010, 1000, 990],
                       [1020, 1010, 1000],
                      [1030, 1020, 1010]])
    root_grp.close()
    return file_name


def _create_dimensions(root_grp):
    n_dim = len(TEST_ARRAY)
    for dim_name in DIMENSIONS:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp):
    for dim_name in DIMENSIONS:
        x = root_grp.createVariable(dim_name, 'f8', (dim_name,))
        x[:] = TEST_ARRAY
        if dim_name == 'height':
            x.units = 'm'


# Luodaan tähän ensin taas fake_source filu,

def test_convert_z_units():
    assert True


def test_read_mie_lut():
    assert True


def test_get_mie_file():
    _get_mie_file = drizzle.DrizzleSource._get_mie_file()
    print(_get_mie_file)

    assert True


def test_get_wl_band():
    assert True


@pytest.mark.parametrize("x, result", [
    (-1000, -1),
    (-100, -0.99999),
    (-10, -0.9),
    (-1, np.exp(-1 / 10 * np.log(10)) - 1),
])
def test_db2lin(x, result):
    testing.assert_array_almost_equal(drizzle.db2lin(x), result, decimal=5)


def test_db2lin_raise():
    with pytest.raises(ValueError):
        drizzle.db2lin(150)


@pytest.mark.parametrize("x, result", [
    (1e6, 60),
    (1e5, 50),
    (1e4, 40),
])
def test_lin2db(x, result):
    testing.assert_array_almost_equal(drizzle.lin2db(x), result, decimal=3)


def test_lin2db_raise():
    with pytest.raises(ValueError):
        drizzle.lin2db(-1)


def test_get_drizzle_indices():
    dia = np.array([-1, 2 * 1e-5, 1, 1e-6])
    d = drizzle.CalculateErrors._get_drizzle_indices(dia)
    correct = {'drizzle': [False, True, True, True],
               'small': [False, True, False, False],
               'tiny': [False, False, False, True]}
    for key in d.keys():
        testing.assert_array_equal(d[key], correct[key])

