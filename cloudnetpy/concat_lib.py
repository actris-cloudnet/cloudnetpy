"""Module for concatenating netCDF files."""
import logging
from typing import Optional, Union

import netCDF4
import numpy as np

from cloudnetpy import utils


def update_nc(old_file: str, new_file: str) -> int:
    """Appends data to existing netCDF file.

    Args:
        old_file: Filename of a existing netCDF file.
        new_file: Filename of a new file whose data will be appended to the end.

    Returns:
        1 = success, 0 = failed to add new data.

    Notes:
        Requires 'time' variable with unlimited dimension.

    """
    try:
        with netCDF4.Dataset(old_file, "a") as nc_old, netCDF4.Dataset(new_file) as nc_new:
            valid_ind = _find_valid_time_indices(nc_old, nc_new)
            if len(valid_ind) > 0:
                _update_fields(nc_old, nc_new, valid_ind)
                return 1
            return 0
    except OSError:
        return 0


def concatenate_files(
    filenames: list,
    output_file: str,
    concat_dimension: str = "time",
    variables: Optional[list] = None,
    new_attributes: Optional[dict] = None,
) -> None:
    """Concatenate netCDF files in one dimension.

    Args:
        filenames: List of files to be concatenated.
        output_file: Output file name.
        concat_dimension: Dimension name for concatenation. Default is 'time'.
        variables: List of variables with the 'concat_dimension' to be concatenated.
            Default is None when all variables with 'concat_dimension' will be saved.
        new_attributes: Optional new global attributes as {'attribute_name': value}.

    Notes:
        Arrays without 'concat_dimension', scalars, and global attributes will be taken from
        the first file. Groups, possibly present in a NETCDF4 formatted file, are ignored.

    """
    with Concat(filenames, output_file, concat_dimension) as concat:
        concat.get_constants()
        concat.create_global_attributes(new_attributes)
        concat.concat_data(variables)


class Concat:
    def __init__(self, filenames: list, output_file: str, concat_dimension: str = "time"):
        self.filenames = sorted(filenames)
        self.concat_dimension = concat_dimension
        self.first_file = netCDF4.Dataset(self.filenames[0])
        self.concatenated_file = self._init_output_file(output_file)
        self.constants = ()

    def create_global_attributes(self, new_attributes: Union[dict, None]) -> None:
        """Copies global attributes from one of the source files."""
        _copy_attributes(self.first_file, self.concatenated_file)
        if new_attributes is not None:
            for key, value in new_attributes.items():
                setattr(self.concatenated_file, key, value)

    def get_constants(self):
        """Finds constants, i.e. arrays that have no concat_dimension and are not concatenated."""
        for key, value in self.first_file.variables.items():
            try:
                dims = self._get_dim(value[:])
            except np.core._exceptions.UFuncTypeError:  # pylint: disable=W0212
                logging.warning(f"Problem with reading {key} - skipping it")
                continue
            if self.concat_dimension not in dims:
                self.constants += (key,)

    def close(self):
        """Closes open files."""
        self.first_file.close()
        self.concatenated_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def concat_data(self, variables: Optional[list] = None):
        """Concatenates data arrays."""
        self._write_initial_data(variables)
        if len(self.filenames) > 1:
            for filename in self.filenames[1:]:
                self._append_data(filename)

    def _write_initial_data(self, variables: Union[list, None]) -> None:
        for key in self.first_file.variables.keys():
            if (
                variables is not None
                and key not in variables
                and key not in self.constants
                and key != self.concat_dimension
            ):
                continue
            self.first_file[key].set_auto_scale(False)
            array = self.first_file[key][:]
            dimensions = self._get_dim(array)
            fill_value = getattr(self.first_file[key], "_FillValue", None)
            var = self.concatenated_file.createVariable(
                key,
                array.dtype,
                dimensions,
                zlib=True,
                complevel=3,
                shuffle=False,
                fill_value=fill_value,
            )
            var.set_auto_scale(False)
            var[:] = array
            _copy_attributes(self.first_file[key], var)

    def _append_data(self, filename: str) -> None:
        file = netCDF4.Dataset(filename)
        file.set_auto_scale(False)
        ind0 = len(self.concatenated_file.variables[self.concat_dimension])
        ind1 = ind0 + len(file.variables[self.concat_dimension])
        for key in self.concatenated_file.variables.keys():
            array = file[key][:]
            if array.ndim == 0 or key in self.constants:
                continue
            if array.ndim == 1:
                self.concatenated_file.variables[key][ind0:ind1] = array
            else:
                self.concatenated_file.variables[key][ind0:ind1, :] = array
        file.close()

    def _get_dim(self, array: np.ndarray) -> tuple:
        """Returns tuple of dimension names, e.g., ('time', 'range') that match the array size."""
        if utils.isscalar(array):
            return ()
        variable_size = []
        file_dims = self.concatenated_file.dimensions
        for length in array.shape:
            try:
                dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
            except IndexError:
                dim = self.concat_dimension
            variable_size.append(dim)
        return tuple(variable_size)

    def _init_output_file(self, output_file: str) -> netCDF4.Dataset:
        data_model = "NETCDF4" if self.first_file.data_model == "NETCDF4" else "NETCDF4_CLASSIC"
        nc = netCDF4.Dataset(output_file, "w", format=data_model)
        for dim in self.first_file.dimensions.keys():
            dim_len = None if dim == self.concat_dimension else self.first_file.dimensions[dim].size
            nc.createDimension(dim, dim_len)
        return nc


def _copy_attributes(source: netCDF4.Dataset, target: netCDF4.Dataset) -> None:
    for attr in source.ncattrs():
        if attr != "_FillValue":
            value = getattr(source, attr)
            setattr(target, attr, value)


def _find_valid_time_indices(nc_old: netCDF4.Dataset, nc_new: netCDF4.Dataset):
    return np.where(nc_new.variables["time"][:] > nc_old.variables["time"][-1])[0]


def _update_fields(nc_old: netCDF4.Dataset, nc_new: netCDF4.Dataset, valid_ind: list):
    ind0 = len(nc_old.variables["time"])
    idx = [ind0 + x for x in valid_ind]
    concat_dimension = nc_old.variables["time"].dimensions[0]
    for field in nc_new.variables:
        if field not in nc_old.variables:
            continue
        dimensions = nc_new.variables[field].dimensions
        if concat_dimension in dimensions:
            concat_ind = dimensions.index(concat_dimension)
            if len(dimensions) == 1:
                nc_old.variables[field][idx] = nc_new.variables[field][valid_ind]
            elif len(dimensions) == 2 and concat_ind == 0:
                nc_old.variables[field][idx, :] = nc_new.variables[field][valid_ind, :]
            elif len(dimensions) == 2 and concat_ind == 1:
                nc_old.variables[field][:, idx] = nc_new.variables[field][:, valid_ind]


def truncate_netcdf_file(filename: str, output_file: str, n_profiles: int):
    """Truncates netcdf file in 'time' dimension taking only n_profiles.
    Useful for creating small files for tests.
    """
    with netCDF4.Dataset(filename, "r") as nc, netCDF4.Dataset(
        output_file, "w", format=nc.data_model
    ) as nc_new:
        for dim in nc.dimensions.keys():
            dim_len = None if dim == "time" else nc.dimensions[dim].size
            nc_new.createDimension(dim, dim_len)
        for attr in nc.ncattrs():
            value = getattr(nc, attr)
            setattr(nc_new, attr, value)
        for key in nc.variables:
            array = nc.variables[key][:]
            dimensions = nc.variables[key].dimensions
            fill_value = getattr(nc.variables[key], "_FillValue", None)
            var = nc_new.createVariable(
                key, array.dtype, dimensions, zlib=True, fill_value=fill_value
            )
            if dimensions and "time" in dimensions[0]:
                if array.ndim == 1:
                    var[:] = array[:n_profiles]
                if array.ndim == 2:
                    var[:] = array[:n_profiles, :]
            else:
                var[:] = array
            for attr in nc.variables[key].ncattrs():
                if attr != "_FillValue":
                    value = getattr(nc.variables[key], attr)
                    setattr(var, attr, value)
