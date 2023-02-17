import glob
import shutil
from os import path

import netCDF4
import numpy as np
import pytest
from numpy import ma

from cloudnetpy import concat_lib
from cloudnetpy.exceptions import InconsistentDataError

SCRIPT_PATH = path.dirname(path.realpath(__file__))


class TestUpdateNc:

    files = glob.glob(f"{SCRIPT_PATH}/data/cl61d/*.nc")
    files.sort()

    def test_does_append_to_end(self, tmp_path):
        temp_path = tmp_path / "end.nc"
        concat_lib.concatenate_files(
            self.files[:2], str(temp_path), concat_dimension="profile"
        )
        succ = concat_lib.update_nc(str(temp_path), self.files[2])
        assert succ == 1
        with netCDF4.Dataset(temp_path) as nc:
            time = nc.variables["time"][:]
            assert len(time) == 3 * 12
            for ind, timestamp in enumerate(time[:-1]):
                assert timestamp < time[ind + 1]

    def test_does_not_append_to_beginning(self, tmp_path):
        temp_path = tmp_path / "beginning.nc"
        concat_lib.concatenate_files(
            self.files[1:3], str(temp_path), concat_dimension="profile"
        )
        succ = concat_lib.update_nc(str(temp_path), self.files[0])
        assert succ == 0
        with netCDF4.Dataset(temp_path) as nc:
            time = nc.variables["time"][:]
            assert len(time) == 2 * 12
            for ind, timestamp in enumerate(time[:-1]):
                assert timestamp < time[ind + 1]

    def test_does_not_append_to_middle(self, tmp_path):
        temp_path = tmp_path / "middle.nc"
        files = [self.files[0], self.files[2]]
        concat_lib.concatenate_files(files, str(temp_path), concat_dimension="profile")
        succ = concat_lib.update_nc(str(temp_path), self.files[1])
        assert succ == 0
        with netCDF4.Dataset(temp_path) as nc:
            time = nc.variables["time"][:]
            assert len(time) == 2 * 12
            for ind, timestamp in enumerate(time[:-1]):
                assert timestamp < time[ind + 1]

    def test_does_not_append_with_invalid_file(self, tmp_path):
        temp_path = tmp_path / "invalid.nc"
        non_nc_file = f"{SCRIPT_PATH}/data/vaisala/cl51.DAT"
        files = [self.files[0], self.files[2]]
        concat_lib.concatenate_files(files, str(temp_path), concat_dimension="profile")
        succ = concat_lib.update_nc(str(temp_path), non_nc_file)
        assert succ == 0


class TestConcat:

    files = [
        f"{SCRIPT_PATH}/data/chm15k/00100_A202010222015_CHM170137.nc",
        f"{SCRIPT_PATH}/data/chm15k/00100_A202010220005_CHM170137.nc",
    ]

    n_time = 10
    n_range = 1024
    n_range_hr = 32
    n_layer = 3

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self, tmp_path):
        output_path = tmp_path / "concat.nc"
        self.concat = concat_lib._Concat(self.files, str(output_path))
        self.file = self.concat.concatenated_file
        yield
        self.concat._close()

    def test_sorting_input_files(self):
        assert self.concat.filenames[0] == self.files[1]
        assert self.concat.filenames[1] == self.files[0]

    def test_create_dimension(self):
        for dim in ("time", "range", "range_hr", "layer"):
            assert dim in self.file.dimensions

    def test_create_constants(self):
        self.concat.get_common_variables()
        for var in ("range", "range_hr", "layer", "latitude", "longitude"):
            assert var in self.concat.common_variables
        for var in ("time", "life_time", "beta_raw"):
            assert var not in self.concat.common_variables

    def test_create_global_attributes(self):
        self.concat.create_global_attributes(
            new_attributes={"kissa": 50, "koira": "23"}
        )
        for attr in ("day", "title", "month", "comment", "kissa", "koira"):
            assert hasattr(self.file, attr)
        assert self.file.kissa == 50
        assert self.file.koira == "23"

    def test_concat_data(self):
        self.concat.get_common_variables()
        self.concat.concat_data(variables=None, ignore=None, allow_vary=None)
        assert len(self.file.variables["time"]) == 2 * self.n_time
        assert len(self.file.variables["range"]) == self.n_range
        assert len(self.file.variables["layer"]) == self.n_layer

    def test_concat_only_some_variables_data(self):
        self.concat.get_common_variables()
        variables = ["cbh", "sci"]
        self.concat.concat_data(variables, ignore=None, allow_vary=None)
        assert len(self.file.variables["time"]) == 2 * self.n_time
        assert len(self.file.variables["range"]) == self.n_range
        for var in variables:
            assert var in self.file.variables
        for var in ("cde", "nn3"):
            assert var not in self.file.variables


class TestCommonVariables:

    n_time = 10
    n_range = 1024
    n_range_hr = 32
    n_layer = 3

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self, tmp_path):
        self.file1 = str(tmp_path / "file1.nc")
        shutil.copy(
            f"{SCRIPT_PATH}/data/chm15k/00100_A202010222015_CHM170137.nc",
            self.file1,
        )
        self.file2 = str(tmp_path / "file2.nc")
        shutil.copy(
            f"{SCRIPT_PATH}/data/chm15k/00100_A202010220005_CHM170137.nc",
            self.file2,
        )
        self.files = [self.file1, self.file2]
        self.output = tmp_path / "concat.nc"
        yield

    def _write_scalar(self, file, key, value):
        with netCDF4.Dataset(file, "r+") as nc:
            var = nc.createVariable(key, "i4")
            var[:] = value

    def _write_array(self, file, key, value):
        with netCDF4.Dataset(file, "r+") as nc:
            nc.createDimension(key, len(value))
            var = nc.createVariable(key, "i4", key)
            var[:] = value

    def test_consistent_scalars(self):
        self._write_scalar(self.file1, "kissa", 1)
        self._write_scalar(self.file2, "kissa", 1)

        with concat_lib._Concat(self.files, str(self.output)) as concat:
            concat.get_common_variables()
            assert "kissa" in concat.common_variables
            concat.concat_data(variables=None, ignore=None, allow_vary=None)
            assert len(concat.concatenated_file["time"]) == 2 * self.n_time
            assert len(concat.concatenated_file["range"]) == self.n_range
            assert len(concat.concatenated_file["layer"]) == self.n_layer
            assert concat.concatenated_file["kissa"][:] == 1

    def test_inconsistent_scalars(self):
        self._write_scalar(self.file1, "kissa", 1)
        self._write_scalar(self.file2, "kissa", 2)

        with concat_lib._Concat(self.files, str(self.output)) as concat:
            concat.get_common_variables()
            assert "kissa" in concat.common_variables
            with pytest.raises(InconsistentDataError) as excinfo:
                concat.concat_data(variables=None, ignore=None, allow_vary=None)
            assert (
                f"Inconsistent values in variable 'kissa' between files '{self.file1}' and '{self.file2}'"
                in str(excinfo.value)
            )

    def test_consistent_arrays(self):
        self._write_array(self.file1, "kissa", [1, 2])
        self._write_array(self.file2, "kissa", [1, 2])

        with concat_lib._Concat(self.files, str(self.output)) as concat:
            concat.get_common_variables()
            assert "kissa" in concat.common_variables
            concat.concat_data(variables=None, ignore=None, allow_vary=None)
            assert len(concat.concatenated_file["time"]) == 2 * self.n_time
            assert len(concat.concatenated_file["range"]) == self.n_range
            assert len(concat.concatenated_file["layer"]) == self.n_layer
            assert np.array_equal(concat.concatenated_file["kissa"][:], [1, 2])

    def test_inconsistent_arrays(self):
        self._write_array(self.file1, "kissa", [1, 2])
        self._write_array(self.file2, "kissa", [2, 1])

        with concat_lib._Concat(self.files, str(self.output)) as concat:
            concat.get_common_variables()
            assert "kissa" in concat.common_variables
            with pytest.raises(InconsistentDataError) as excinfo:
                concat.concat_data(variables=None, ignore=None, allow_vary=None)
            assert (
                f"Inconsistent values in variable 'kissa' between files '{self.file1}' and '{self.file2}'"
                in str(excinfo.value)
            )

    def test_consistent_masked_arrays(self):
        self._write_array(
            self.file1, "kissa", ma.masked_array([1, 2, 3], mask=[1, 0, 1])
        )
        self._write_array(
            self.file2, "kissa", ma.masked_array([3, 2, 1], mask=[1, 0, 1])
        )

        with concat_lib._Concat(self.files, str(self.output)) as concat:
            concat.get_common_variables()
            assert "kissa" in concat.common_variables
            concat.concat_data(variables=None, ignore=None, allow_vary=None)
            assert len(concat.concatenated_file["time"]) == 2 * self.n_time
            assert len(concat.concatenated_file["range"]) == self.n_range
            assert len(concat.concatenated_file["layer"]) == self.n_layer
            assert ma.allequal(
                concat.concatenated_file["kissa"][:],
                ma.masked_array([1, 2, 3], mask=[1, 0, 1]),
            )

    def test_inconsistent_masked_arrays(self):
        self._write_array(
            self.file1, "kissa", ma.masked_array([1, 2, 3], mask=[1, 0, 1])
        )
        self._write_array(
            self.file2, "kissa", ma.masked_array([2, 3, 4], mask=[1, 0, 1])
        )

        with concat_lib._Concat(self.files, str(self.output)) as concat:
            concat.get_common_variables()
            assert "kissa" in concat.common_variables
            with pytest.raises(InconsistentDataError) as excinfo:
                concat.concat_data(variables=None, ignore=None, allow_vary=None)
            assert (
                f"Inconsistent values in variable 'kissa' between files '{self.file1}' and '{self.file2}'"
                in str(excinfo.value)
            )


def test_concatenate_files_with_mira(tmp_path):
    files = [
        f"{SCRIPT_PATH}/data/mira/20210102_1400.mmclx",
        f"{SCRIPT_PATH}/data/mira/20210102_0000.mmclx",
    ]
    output_file = tmp_path / "mira.nc"
    variables = ["microsec", "SNRCorFaCo"]
    concat_lib.concatenate_files(
        files,
        str(output_file),
        variables=variables,
        new_attributes={"kissa": 50},
    )
    with netCDF4.Dataset(output_file) as nc:
        assert len(nc.variables["time"]) == 145 + 146
        assert len(nc.variables["range"]) == 413
        assert nc.data_model == "NETCDF4_CLASSIC"
        for var in ("prf", "microsec"):
            assert var in nc.variables
        for var in ("VELg", "elv"):
            assert var not in nc.variables
        assert nc.kissa == 50
