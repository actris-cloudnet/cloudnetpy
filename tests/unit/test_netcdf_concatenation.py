import numpy as np
from numpy import ma
import netCDF4
from pathlib import Path
import pytest
from numpy.testing import assert_array_equal, assert_allclose
import numpy.typing as npt

from cloudnetpy.concat_lib import concatenate_files


def _create_test_file(
    path: Path,
    times: list[int],
    ranges: list[float] | npt.NDArray = np.array([10.0, 20.0]),
    scalar: ma.MaskedArray | float | None = None,
    data1: ma.MaskedArray | None = None,
    data2: ma.MaskedArray | None = None,
    chirp_ind: list[float] | ma.MaskedArray | None = None,
    range_var: ma.MaskedArray | None = None,
    char_scalar: str | None = None,
    char_array: list[str] | None = None,
) -> Path:
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", len(times))
        nc.createDimension("range", len(ranges))
        if chirp_ind is not None:
            nc.createDimension("chirp", len(chirp_ind))

        var = nc.createVariable("time", "i4", ("time",))
        var[:] = times

        var = nc.createVariable("range", "f4", ("range",))
        var[:] = ranges

        if data1 is not None:
            var = nc.createVariable("data1", "f4", ("time", "range"))
            var[:] = data1

        if data2 is not None:
            var = nc.createVariable("data2", "f4", ("time", "range"))
            var[:] = data2

        if chirp_ind is not None:
            var = nc.createVariable("chirp_ind", "f4", ("chirp",))
            var[:] = chirp_ind

        if scalar is not None:
            var = nc.createVariable("scalar", "f4", ())
            var[:] = scalar

        if range_var is not None:
            var = nc.createVariable("range_var", "f4", ("range",))
            var[:] = range_var

        if char_scalar is not None:
            var = nc.createVariable("char_var", "S1")
            var[:] = char_scalar.encode("utf-8")

        if char_array is not None:
            var = nc.createVariable("char_array", "S1", ("range",))
            var[:] = np.array([s.encode("utf-8") for s in char_array], dtype="S1")

        nc.title = "TEST_FILE"

    return path


def test_basic_concatenation_without_masked_values(tmp_path):
    arr1 = ma.array([[0, 1], [2, 3]], mask=False)
    arr2 = ma.array([[4, 5], [6, 7]], mask=False)
    f1 = _create_test_file(tmp_path / "f1.nc", times=[0, 1], data1=arr1)
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3], data1=arr2)
    out = tmp_path / "out.nc"

    result = concatenate_files([f1, f2], str(out))

    assert result == [f1, f2]

    with netCDF4.Dataset(out) as nc:
        arr = nc.variables["time"][:]
        assert_array_equal(arr, [0, 1, 2, 3])

        expected: ma.MaskedArray = ma.array(
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            mask=False,
            dtype="f4",
        )
        arr = nc.variables["data1"][:]
        assert_array_equal(arr, expected)

        assert getattr(nc, "title") == "TEST_FILE"


def test_basic_concatenation_with_masked_values(tmp_path):
    arr1 = ma.array([[0, 1], [2, 3]], mask=[[False, False], [False, True]])
    arr2 = ma.array([[4, 5], [6, 7]], mask=[[False, False], [True, False]])
    f1 = _create_test_file(tmp_path / "f1.nc", times=[0, 1], data1=arr1)
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3], data1=arr2)
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        expected: ma.MaskedArray = ma.array(
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            mask=[[False, False], [False, True], [False, False], [True, False]],
            dtype="f4",
        )
        arr = nc.variables["data1"][:]
        assert_array_equal(arr, expected)


def test_scalar_broadcasting_without_masked_values(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        scalar=ma.array(5.0, mask=False),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        scalar=ma.array(15.0, mask=False),
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["scalar"][:]
        assert_allclose(data, [5.0, 5.0, 15.0, 15.0])


def test_scalar_broadcasting_with_masked_values(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        scalar=ma.array(5.0, mask=False),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        scalar=ma.array(15.0, mask=True),
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["scalar"][:]
        expected: ma.MaskedArray = ma.array(
            [5.0, 5.0, 15.0, 15.0],
            mask=[False, False, True, True],
            dtype="f4",
        )
        assert_allclose(data, expected)


def test_scalar_broadcasting_with_missing_variable(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        scalar=ma.array(5.0, mask=False),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["scalar"][:]
        expected: ma.MaskedArray = ma.array(
            [5.0, 5.0, np.nan, np.nan],
            mask=[False, False, True, True],
            dtype="f4",
        )
        assert_allclose(data, expected)


def test_variables_filter(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        ranges=[1.0, 2.0],
        data1=ma.array([[0, 1], [2, 3]], mask=False),
        data2=ma.array([[1, 2], [3, 4]], mask=False),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        ranges=[1.0, 2.0],
        data1=ma.array([[4, 5], [6, 7]], mask=False),
        data2=ma.array([[5, 6], [7, 8]], mask=False),
    )
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3], ranges=[1.0, 2.0])
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out), variables=["data1"])

    with netCDF4.Dataset(out) as nc:
        vars_out = set(nc.variables.keys())
        assert "data1" in vars_out
        assert "data2" not in vars_out
        assert "time" in vars_out
        assert "range" in vars_out


def test_ignore_list(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0],
        ranges=[5.0],
        data1=ma.array([[0]], mask=False),
        data2=ma.array([[1]], mask=False),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[1],
        ranges=[5.0],
        data1=ma.array([[2]], mask=False),
        data2=ma.array([[3]], mask=False),
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out), ignore=["data1"])

    with netCDF4.Dataset(out) as nc:
        vars_out = set(nc.variables.keys())
        assert "data" not in vars_out
        assert "data2" in vars_out
        assert "time" in vars_out
        assert "range" in vars_out


def test_interpolation(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        data1=ma.array([[0, 1], [2, 3]], mask=False),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        ranges=[10.0, 20.0, 30.0],
        data1=ma.array([[4, 5, 6], [7, 8, 9]], mask=False),
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["data1"][:]
        expected: ma.MaskedArray = ma.array(
            [[0, 1], [2, 3], [4, 5], [7, 8]], mask=False, dtype="f4"
        )
        assert_allclose(data, expected)


def test_interpolation_II(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        ranges=[10.0, 20.0, 30.0],
        data1=ma.array(
            [[0, 1, 2], [3, 4, 5]], mask=[[False, False, False], [False, False, False]]
        ),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        data1=ma.array([[6, 7], [8, 9]], mask=[[False, False], [False, False]]),
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["data1"][:]
        expected: ma.MaskedArray = ma.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 7], [8, 9, 9]],
            mask=[
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, False, True],
            ],
            dtype="f4",
        )

        assert_allclose(data, expected)


def test_interpolation_III(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        ranges=[10.0, 20.0, 30.0],
        data1=ma.array(
            [[0, 1, 2], [3, 4, 5]], mask=[[False, True, False], [False, False, False]]
        ),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        ranges=[16.0, 24.0],
        data1=ma.array([[6, 7], [8, 9]], mask=[[False, False], [False, True]]),
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["data1"][:]
        expected: ma.MaskedArray = ma.array(
            [[0, 1, 2], [3, 4, 5], [6, 6, 7], [8, 8, 9]],
            mask=[
                [False, True, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
            ],
            dtype="f4",
        )

        assert_allclose(data, expected)


def test_interpolation_IV(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        ranges=[10.0, 20.0, 30.0],
        range_var=ma.array([1, 2, 3], mask=False),
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        ranges=[16.0, 24.0],
        range_var=ma.array([5, 6], mask=False),
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["range_var"][:]
        expected = ma.array([[1, 2, 3], [1, 2, 3], [5, 5, 6], [5, 5, 6]], mask=False)
        assert_allclose(data, expected)

        data = nc.variables["range"][:]
        expected = ma.array([10.0, 20.0, 30.0], mask=False)
        assert_allclose(data, expected)


def test_extra_dimension_broadcasting(tmp_path):
    f1 = _create_test_file(tmp_path / "f1.nc", times=[0, 1], chirp_ind=[1, 2, 3])
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3], chirp_ind=[4, 5, 6])
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["chirp_ind"][:]
        expected = np.array(
            [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]],
            dtype="f4",
        )
        assert_allclose(data, expected)


def test_extra_dimension_broadcasting_with_masked_values(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        chirp_ind=ma.array([1, 2, 3], mask=[False, True, False]),
    )
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3], chirp_ind=[4, 5, 6])
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["chirp_ind"][:]
        expected: ma.MaskedArray = ma.array(
            [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]],
            mask=[
                [False, True, False],
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ],
            dtype="f4",
        )
        assert_allclose(data, expected)
        assert_array_equal(data.mask, expected.mask)


def test_extra_dimension_discrepancy(tmp_path):
    f1 = _create_test_file(tmp_path / "f1.nc", times=[0, 1], chirp_ind=[1, 2, 3])
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3], chirp_ind=[4, 5, 6, 7])
    out = tmp_path / "out.nc"

    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        concatenate_files([f1, f2], str(out))


def test_missing_variable_in_second_file(tmp_path):
    f1 = _create_test_file(tmp_path / "f1.nc", times=[0, 1], chirp_ind=[1, 2, 3])
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3])
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))
    with netCDF4.Dataset(out) as nc:
        assert "chirp_ind" in nc.variables
        data = nc.variables["chirp_ind"][:]
        expected = np.array(
            [[1, 2, 3], [1, 2, 3], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            dtype="f4",
        )
        assert_allclose(data, expected)


def test_missing_variable_in_first_file(tmp_path):
    f1 = _create_test_file(tmp_path / "f1.nc", times=[0, 1])
    f2 = _create_test_file(tmp_path / "f2.nc", times=[2, 3], chirp_ind=[1, 2, 3])
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))
    with netCDF4.Dataset(out) as nc:
        assert "chirp_ind" not in nc.variables


def test_character_variable(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        char_scalar="a",
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        char_scalar="b",
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["char_var"][:]
        expected = np.array([b"a", b"a", b"b", b"b"], dtype="S1")
        assert_array_equal(data, expected)


def test_character_array_interpolation(tmp_path):
    f1 = _create_test_file(
        tmp_path / "f1.nc",
        times=[0, 1],
        ranges=[10.0, 20.0, 30],
        char_array=["a", "b", "c"],
    )
    f2 = _create_test_file(
        tmp_path / "f2.nc",
        times=[2, 3],
        char_array=["d", "e"],
    )
    out = tmp_path / "out.nc"

    concatenate_files([f1, f2], str(out))

    with netCDF4.Dataset(out) as nc:
        data = nc.variables["char_array"][:]
        expected = np.array(
            [
                [b"a", b"b", b"c"],
                [b"a", b"b", b"c"],
                [b"d", b"e", np.nan],
                [b"d", b"e", np.nan],
            ],
            dtype="S1",
        )
        assert_array_equal(data, expected)
