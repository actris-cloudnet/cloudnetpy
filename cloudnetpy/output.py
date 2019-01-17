""" Functions for Categorize output file writing."""
from datetime import datetime, timezone
import uuid
import netCDF4
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy import config
from cloudnetpy.metadata import ATTRIBUTES


def write_vars2nc(rootgrp, cnet_variables, zlib):
    """Iterate over Cloudnet instances and write to given rootgrp."""

    def _get_dimensions(array):
        """Finds correct dimensions for a variable."""
        if utils.isscalar(array):
            return ()
        size = ()
        file_dims = rootgrp.dimensions
        array_dims = array.shape
        for length in array_dims:
            dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
            size = size + (dim,)
        return size

    for name in cnet_variables:
        obj = cnet_variables[name]
        size = _get_dimensions(obj.data)
        ncvar = rootgrp.createVariable(obj.name, obj.data_type, size, zlib=zlib)
        ncvar[:] = obj.data
        for attr in obj.fetch_attributes():
            setattr(ncvar, attr, getattr(obj, attr))


def status_name(long_name):
    """ Default retrieval status variable name """
    return long_name + ' retrieval status'


def bias_name(long_name):
    """ Default bias variable name """
    return 'Possible bias in ' + long_name.lower() + ', one standard deviation'


def err_name(long_name):
    """ Default error variable name """
    return 'Random error in ' + long_name.lower() + ', one standard deviation'


def err_comm(long_name):
    """ Default error comment """
    return ('This variable is an estimate of the one-standard-deviation random error\n'
            'in ' + long_name.lower() + 'due to the uncertainty of the retrieval, including\n'
            'the random error in the radar and lidar parameters.')


def bias_comm(long_name):
    """ Default bias comment """
    return ('This variable is an estimate of the possible systematic error in '
            + long_name.lower() + 'due to the\n'
            'uncertainty in the calibration of the radar and lidar.')


def anc_names(var, bias=False, err=False, sens=False):
    """Returns list of ancillary variable names."""
    out = ''
    if bias:
        out += f"{var}_bias "
    if err:
        out += f"{var}_error "
    if sens:
        out += f"{var}_sensitivity "
    return out[:-1]


def copy_dimensions(file_from, file_to, dims_to_be_copied):
    """Copies dimensions from one file to another. """
    for dname, dim in file_from.dimensions.items():
        if dname in dims_to_be_copied:
            file_to.createDimension(dname, len(dim))


def copy_variables(file_from, file_to, vars_to_be_copied):
    """Copies variables (and their attributes) from one file to another."""
    for vname, varin in file_from.variables.items():
        if vname in vars_to_be_copied:
            varout = file_to.createVariable(vname, varin.datatype, varin.dimensions)
            varout.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            varout[:] = varin[:]


def copy_global(file_from, file_to, attrs_to_be_copied):
    """Copies global attributes from one file to another."""
    for aname in file_from.ncattrs():
        if aname in attrs_to_be_copied:
            setattr(file_to, aname, file_from.getncattr(aname))


def update_attributes(cloudnet_variables):
    """Overrides existing attributes such as 'units' etc. 
    using hard-coded values. New attributes are added.

    Args:
        cloudnet_variables (dict): CloudnetArray instances.

    """
    for field in cloudnet_variables:
        if field in ATTRIBUTES:
            cloudnet_variables[field].set_attributes(ATTRIBUTES[field])
