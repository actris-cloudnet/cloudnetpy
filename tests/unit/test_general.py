import netCDF4

from cloudnetpy.instruments import general


def test_get_files_with_common_range(tmp_path):
    path1 = tmp_path / "file1.nc"
    with netCDF4.Dataset(path1, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("range", 100)
        nc.createVariable("range", dimensions="range", datatype="f8")
    path2 = tmp_path / "file2.nc"
    with netCDF4.Dataset(path2, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("range", 99)
        nc.createVariable("range", dimensions="range", datatype="f8")
    path3 = tmp_path / "file3.nc"
    with netCDF4.Dataset(path3, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("range", 100)
        nc.createVariable("range", dimensions="range", datatype="f8")
    files = general.get_files_with_common_range([path1, path2, path3])
    assert len(files) == 2
    assert path1 in files
    assert path3 in files
    assert path2 not in files
