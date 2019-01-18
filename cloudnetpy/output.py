""" Functions for Categorize output file writing."""

from cloudnetpy import utils
from cloudnetpy.metadata import ATTRIBUTES


def write_vars2nc(rootgrp, cloudnet_variables, zlib):
    """Iterate over Cloudnet instances and write to given rootgrp."""

    def _get_dimensions(array):
        """Finds correct dimensions for a variable."""
        if utils.isscalar(array):
            return ()
        variable_size = ()
        file_dims = rootgrp.dimensions
        array_dims = array.shape
        for length in array_dims:
            dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
            variable_size = variable_size + (dim,)
        return variable_size

    for key in cloudnet_variables:
        obj = cloudnet_variables[key]
        size = _get_dimensions(obj.data)
        nc_variable = rootgrp.createVariable(obj.name, obj.data_type, size, zlib=zlib)
        nc_variable[:] = obj.data
        for attr in obj.fetch_attributes():
            setattr(nc_variable, attr, getattr(obj, attr))


def copy_dimensions(source, target, dim_list):
    """Copies dimensions from one file to another. """
    for dim_name, dimension in source.dimensions.items():
        if dim_name in dim_list:
            target.createDimension(dim_name, len(dimension))


def copy_variables(source, target, var_list):
    """Copies variables (and their attributes) from one file to another."""
    for var_name, variable in source.variables.items():
        if var_name in var_list:
            var_out = target.createVariable(var_name, variable.datatype, variable.dimensions)
            var_out.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            var_out[:] = variable[:]


def copy_global(source, target, attr_list):
    """Copies global attributes from one file to another."""
    for attr_name in source.ncattrs():
        if attr_name in attr_list:
            setattr(target, attr_name, source.getncattr(attr_name))


def update_attributes(cloudnet_variables):
    """Overrides existing attributes such as 'units' etc. 
    using hard-coded values. New attributes are added.

    Args:
        cloudnet_variables (dict): CloudnetArray instances.

    """
    for key in cloudnet_variables:
        if key in ATTRIBUTES:
            cloudnet_variables[key].set_attributes(ATTRIBUTES[key])
