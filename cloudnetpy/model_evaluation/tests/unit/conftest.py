from datetime import date

import netCDF4
import numpy as np
import pytest


@pytest.fixture(scope="session")
def file_metadata() -> dict:
    year, month, day = "2019", "05", "23"
    return {
        "year": year,
        "month": month,
        "day": day,
        "location": "Kumpula",
        "case_date": date(int(year), int(month), int(day)),
        "altitude_km": 0.5,
    }


@pytest.fixture(scope="session")
def model_file(tmpdir_factory, file_metadata) -> str:
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    time = 3
    level = 2
    root_grp.createDimension("time", time)
    root_grp.createDimension("level", level)
    _create_global_attributes(root_grp, file_metadata)
    var = root_grp.createVariable("time", "f8", "time")
    var[:] = np.array([2, 6, 10])
    var = root_grp.createVariable("level", "f8", "level")
    var[:] = level
    var = root_grp.createVariable("latitude", "f8")
    var[:] = 1
    var = root_grp.createVariable("longitude", "f8")
    var[:] = 1
    var = root_grp.createVariable("horizontal_resolution", "f8")
    var[:] = 9
    var = root_grp.createVariable("height", "f8", ("time", "level"))
    var[:] = np.array([[10, 14], [8, 14], [9, 15]])
    var.units = "km"
    var = root_grp.createVariable("forecast_time", "f8", "time")
    var[:] = np.array([1, 5, 10])
    var = root_grp.createVariable("cloud_fraction", "f8", ("time", "level"))
    var[:] = np.array([[0, 2], [3, 6], [5, 8]])
    var = root_grp.createVariable("qi", "f8", ("time", "level"))
    var[:] = np.array([[0.01, 0.00], [0.02, 0.03], [0.06, 0.08]])
    var = root_grp.createVariable("ql", "f8", ("time", "level"))
    var[:] = np.array([[0.008, 0.09], [0.004, 0.007], [0.0001, 0.0002]])
    var = root_grp.createVariable("temperature", "f8", ("time", "level"))
    var[:] = np.array([[300, 301], [302, 299], [305, 298]])
    var = root_grp.createVariable("pressure", "f8", ("time", "level"))
    var[:] = np.array([[1000, 1001], [1010, 1003], [1020, 1005]])
    var = root_grp.createVariable("uwind", "f8", ("time", "level"))
    var[:] = np.array([[1, 2], [2, 2], [3, 1]])
    var = root_grp.createVariable("vwind", "f8", ("time", "level"))
    var[:] = np.array([[3, 1], [2, 1], [5, 2]])
    root_grp.close()
    return file_name


@pytest.fixture(scope="session")
def obs_file(tmpdir_factory, file_metadata) -> str:
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    time = 6
    height = 4
    root_grp.createDimension("time", time)
    root_grp.createDimension("height", height)
    _create_global_attributes(root_grp, file_metadata)
    var = root_grp.createVariable("time", "f8", "time")
    var[:] = np.array([0, 2, 4, 6, 8, 10])
    var = root_grp.createVariable("height", "f8", "height")
    var[:] = np.array([8, 9, 12, 15])
    var.units = "km"
    var = root_grp.createVariable("latitude", "f8")
    var[:] = 1
    var = root_grp.createVariable("longitude", "f8")
    var[:] = 1
    var = root_grp.createVariable("altitude", "f8")
    var[:] = 1
    var.units = "km"
    var = root_grp.createVariable("radar_frequency", "f8")
    var[:] = 35.5
    var = root_grp.createVariable("rainrate", "i4", "time")
    var[:] = np.array([1, 2, 0, 10, 5, 13])
    var = root_grp.createVariable("Z_sensitivity", "f8", "height")
    var[:] = np.array([0.1, 0.2, 0.0, 1.0])
    var = root_grp.createVariable("category_bits", "i4", ("time", "height"))
    var[:] = np.array(
        [
            [0, 1, 2, 0],
            [2, 8, 4, 1],
            [1, 4, 0, 8],
            [4, 4, 0, 1],
            [0, 8, 16, 16],
            [32, 2, 8, 32],
        ],
    )
    var = root_grp.createVariable("quality_bits", "i4", ("time", "height"))
    var[:] = np.array(
        [
            [0, 1, 2, 4],
            [8, 16, 32, 16],
            [8, 4, 2, 1],
            [0, 1, 2, 4],
            [8, 16, 32, 16],
            [8, 4, 2, 1],
        ],
    )
    var = root_grp.createVariable("iwc", "f8", ("time", "height"))
    var[:] = np.array(
        [
            [0.01, 0.02, 0.06, 0.01],
            [0.02, 0.06, 0.00, 0.03],
            [0.08, 0.00, 0.03, 0.08],
            [0.01, 0.02, 0.06, 0.01],
            [0.02, 0.06, 0.00, 0.03],
            [0.08, 0.00, 0.03, 0.08],
        ],
    )
    var = root_grp.createVariable("iwc_retrieval_status", "f8", ("time", "height"))
    var[:] = np.array(
        [
            [1, 2, 6, 1],
            [2, 6, 5, 3],
            [7, 1, 3, 4],
            [1, 2, 6, 7],
            [4, 6, 5, 3],
            [7, 5, 3, 4],
        ],
    )
    var = root_grp.createVariable("lwc", "f8", ("time", "height"))
    var[:] = np.array(
        [
            [0.08, 0.04, 0.01, 0.08],
            [0.04, 0.01, 0.09, 0.07],
            [0.02, 0.09, 0.07, 0.02],
            [0.08, 0.04, 0.01, 0.08],
            [0.04, 0.01, 0.09, 0.07],
            [0.02, 0.09, 0.07, 0.02],
        ],
    )
    var = root_grp.createVariable("data", "i4", ("time", "height"))
    var[:] = np.array(
        [
            [2, 4, 3, 6],
            [7, 1, 9, 7],
            [2, 8, 6, 1],
            [3, 5, 1, 0],
            [2, 5, 6, 1],
            [2, 9, 7, 2],
        ],
    )
    root_grp.close()
    return file_name


@pytest.fixture(scope="session")
def regrid_file(tmpdir_factory, file_metadata) -> str:
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    time = 3
    level = 2
    root_grp.createDimension("time", time)
    root_grp.createDimension("level", level)
    _create_global_attributes(root_grp, file_metadata)
    var = root_grp.createVariable("time", "f8", "time")
    var[:] = np.array([2, 6, 10])
    var = root_grp.createVariable("level", "f8", "level")
    var[:] = level
    var = root_grp.createVariable("latitude", "f8")
    var[:] = 1
    var = root_grp.createVariable("longitude", "f8")
    var[:] = 1
    var = root_grp.createVariable("horizontal_resolution", "f8")
    var[:] = 9
    var = root_grp.createVariable("ecmwf_height", "f8", ("time", "level"))
    var[:] = np.array([[10, 14], [8, 14], [9, 15]])
    var = root_grp.createVariable("ecmwf_forecast_time", "f8", "time")
    var[:] = np.array([1, 5, 10])
    var = root_grp.createVariable("ecmwf_cf", "f8", ("time", "level"))
    var[:] = np.array([[0, 2], [3, 6], [5, 8]])
    var = root_grp.createVariable("ecmwf_cf_cirrus", "f8", ("time", "level"))
    var[:] = np.array([[0, 2], [3, 6], [5, 7]])
    var = root_grp.createVariable("ecmwf_cf_snow", "f8", ("time", "level"))
    var[:] = np.array([[0, 2], [4, 6], [5, 8]])
    var = root_grp.createVariable("cf_ecmwf", "f8", ("time", "level"))
    var[:] = np.array([[0, 2], [3, 6], [5, 8]])
    var = root_grp.createVariable("cf_adv_ecmwf", "f8", ("time", "level"))
    var[:] = np.array([[0, 2], [3, 6], [5, 8]])
    var = root_grp.createVariable("temperature", "f8", ("time", "level"))
    var[:] = np.array([[300, 301], [302, 299], [305, 298]])
    var = root_grp.createVariable("pressure", "f8", ("time", "level"))
    var[:] = np.array([[1000, 1001], [1010, 1003], [1020, 1005]])
    root_grp.close()
    return file_name


def _create_global_attributes(root_grp, meta) -> None:
    for key in ("year", "month", "day", "location"):
        setattr(root_grp, key, meta[key])
