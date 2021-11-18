import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import netCDF4
from cloudnetpy.instruments import general
from tempfile import NamedTemporaryFile


def test_get_files_with_common_range():
    file1 = NamedTemporaryFile()
    nc = netCDF4.Dataset(file1.name, "w", format="NETCDF4_CLASSIC")
    nc.createDimension('range', 100)
    nc.createVariable('range', dimensions='range', datatype='f8')
    nc.close()
    file2 = NamedTemporaryFile()
    nc = netCDF4.Dataset(file2.name, "w", format="NETCDF4_CLASSIC")
    nc.createDimension('range', 99)
    nc.createVariable('range', dimensions='range', datatype='f8')
    nc.close()
    file3 = NamedTemporaryFile()
    nc = netCDF4.Dataset(file3.name, "w", format="NETCDF4_CLASSIC")
    nc.createDimension('range', 100)
    nc.createVariable('range', dimensions='range', datatype='f8')
    nc.close()
    files = general.get_files_with_common_range([file1.name, file2.name, file3.name])
    assert len(files) == 2
    assert file1.name in files
    assert file3.name in files
    assert file2.name not in files


def test_add_zenith_angle_vector():
    ele = ma.array([90, 90, 90, 89, 86, 90, 90])

    class Foo:
        def __init__(self):
            self.data = {'elevation': ele}

    foo = Foo()
    valid_ind = general.add_zenith_angle(foo)
    assert valid_ind == [True, True, True, False, False, True, True]
    expected = ma.array([0, 0, 0, 1, 4, 0, 0])
    assert_array_equal(foo.data['zenith_angle'].data, expected.data)
    assert 'elevation' not in foo.data


def test_add_zenith_angle_scalar():
    ele = ma.array([90])

    class Foo:
        def __init__(self):
            self.data = {'elevation': ele}
            self.time = [1.1, 1.2, 1.3]

    foo = Foo()
    valid_ind = general.add_zenith_angle(foo)
    assert valid_ind == [0, 1, 2]
    assert_array_equal(foo.data['zenith_angle'].data, 0)
    assert 'elevation' not in foo.data
