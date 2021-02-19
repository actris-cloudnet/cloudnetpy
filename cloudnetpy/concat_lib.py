"""Module for concatenating netCDF files."""
from typing import Union, Optional
import numpy as np
import netCDF4
from cloudnetpy import utils


def concatenate_files(filenames: list, output_file: str,
                      variables: Optional[list] = None,
                      new_attributes: Optional[dict] = None) -> None:
    """Concatenate list of netCDF files in time dimension.

    Args:
        filenames: List of files to be concatenated.
        output_file: Output file name.
        variables: List of variables with the 'time' dimension to be concatenated.
            Default is None when all variables with 'time' will be saved.
        new_attributes: Optional new global attributes as {'attribute_name': value}.

    Notes:
        Arrays without 'time' dimension, scalars, and global attributes will be taken from
        the first file.

    """
    concat = Concat(filenames, output_file)
    concat.get_constants()
    concat.create_global_attributes(new_attributes)
    concat.concat_data(variables)
    concat.close()


class Concat:
    def __init__(self, filenames: list, output_file: str):
        self.filenames = sorted(filenames)
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
        """Finds constants, i.e. arrays that have no time dimension and are not concatenated."""
        for key, value in self.first_file.variables.items():
            dims = self._get_dim(value[:])
            if 'time' not in dims:
                self.constants += (key,)

    def close(self):
        """Closes open files."""
        self.first_file.close()
        self.concatenated_file.close()

    def concat_data(self, variables: Optional[list] = None):
        """Concatenates data arrays."""
        self._write_initial_data(variables)
        if len(self.filenames) > 1:
            for filename in self.filenames[1:]:
                self._append_data(filename)

    def _write_initial_data(self, variables: Union[list, None]) -> None:
        for key in self.first_file.variables.keys():
            if (variables is not None and key not in variables
                    and key not in self.constants and key != 'time'):
                continue
            self.first_file[key].set_auto_scale(False)
            array = self.first_file[key][:]
            dimensions = self._get_dim(array)
            var = self.concatenated_file.createVariable(key, array.dtype, dimensions, zlib=True,
                                                        complevel=3, shuffle=False)
            var.set_auto_scale(False)
            var[:] = array
            _copy_attributes(self.first_file[key], var)

    def _append_data(self, filename: str) -> None:
        file = netCDF4.Dataset(filename)
        file.set_auto_scale(False)
        ind0 = len(self.concatenated_file.variables['time'])
        ind1 = ind0 + len(file.variables['time'])
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
        variable_size = ()
        file_dims = self.concatenated_file.dimensions
        for length in array.shape:
            try:
                dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
            except IndexError:
                dim = 'time'
            variable_size += (dim,)
        return variable_size

    def _init_output_file(self, output_file: str) -> netCDF4.Dataset:
        nc = netCDF4.Dataset(output_file, 'w', format='NETCDF4_CLASSIC')
        for dim in self.first_file.dimensions.keys():
            dim_len = None if dim == 'time' else len(self.first_file[dim])
            nc.createDimension(dim, dim_len)
        return nc


def _copy_attributes(source: netCDF4.Dataset, target: netCDF4.Dataset) -> None:
    for attr in source.ncattrs():
        value = getattr(source, attr)
        setattr(target, attr, value)
