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

