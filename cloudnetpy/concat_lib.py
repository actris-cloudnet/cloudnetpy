"""Module for concatenating netCDF files."""
from typing import Union
import numpy as np
import netCDF4
from cloudnetpy import utils


def concatenate_files(filenames: list, output_file: str,
                      dimensions: tuple = ('range',),
                      variables: Union[list, None] = None,
                      new_attributes: Union[dict, None] = None) -> None:
    """Concatenate list of netCDF files in time dimension.

    Args:
        filenames (list): List of files to be concatenated.
        output_file (str): Output file name.
        dimensions (tuple, optional): Optional list of additional dimensions besides 'time' to be
            included in the output file. Default is ('range',).
        variables (list, optional): List of variables with the 'time' dimension to be concatenated.
            Default is None when all variables with 'time' will be saved.
        new_attributes (dict, optional): Optional new global attributes as {'attribute_name': value}.
            The attribute value will be saved as string.

    Notes:
        Arrays without 'time' dimension, scalars, and global attributes will be taken from
        the first file.

    """
    concat = Concat(filenames, output_file)
    for dim in ('time',) + dimensions:
        concat.create_dimension(dim)
    concat.get_constants()
    concat.create_global_attributes(new_attributes)
    concat.concat_data(variables)
    concat.close()


class Concat:
    def __init__(self, filenames: list, output_file: str):
        self.filenames = sorted(filenames)
        self.first_file = netCDF4.Dataset(self.filenames[0])
        self.concatenated_file = netCDF4.Dataset(output_file, 'w', format='NETCDF4_CLASSIC')
        self.constants = ()

    def create_dimension(self, dim_name: str) -> None:
        """Creates required dimensions, 'time' will be unlimited."""
        if dim_name == 'time':
            dim_len = None
        else:
            dim_len = len(self.first_file[dim_name])
        self.concatenated_file.createDimension(dim_name, dim_len)

    def create_global_attributes(self, new_attributes: Union[dict, None] = None) -> None:
        """Copies global attributes from one of the source files."""
        _copy_attributes(self.first_file, self.concatenated_file)
        if new_attributes is not None:
            for key, value in new_attributes.items():
                setattr(self.concatenated_file, key, str(value))

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

    def concat_data(self, variables: Union[list, None] = None):
        """Concatenates data arrays."""
        self._write_initial_data(variables)
        if len(self.filenames) > 1:
            for filename in self.filenames[1:]:
                self._append_data(filename)

    def _write_initial_data(self, variables: Union[list, None] = None) -> None:
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


def _copy_attributes(source, target) -> None:
    for attr in source.ncattrs():
        value = getattr(source, attr)
        setattr(target, attr, str(value))
