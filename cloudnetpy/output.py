""" Functions for file writing."""
import os
import netCDF4
from cloudnetpy import utils, version
from cloudnetpy.metadata import COMMON_ATTRIBUTES
from cloudnetpy.products import product_tools


def update_attributes(cloudnet_variables, attributes):
    """Overrides existing CloudnetArray-attributes.

    Overrides existing attributes using hard-coded values.
    New attributes are added.

    Args:
        cloudnet_variables (dict): CloudnetArray instances.
        attributes (dict): Product-specific attributes.

    """
    for key in cloudnet_variables:
        if key in attributes:
            cloudnet_variables[key].set_attributes(attributes[key])
        if key in COMMON_ATTRIBUTES:
            cloudnet_variables[key].set_attributes(COMMON_ATTRIBUTES[key])


def save_product_file(short_id, obj, file_name, copy_from_cat=()):
    """Saves a standard Cloudnet product file.

    Args:
        short_id (str): Short file identifier, e.g. 'lwc', 'iwc', 'drizzle',
            'classification'.
        obj (object): Instance containing product specific attributes: `time`,
            `dataset`, `data`.
        file_name (str): Name of the output file to be generated.
        copy_from_cat (tuple, optional): Variables to be copied from the
            categorize file.

    """
    identifier = _get_identifier(short_id)
    dimensions = {'time': len(obj.time),
                  'height': len(obj.dataset.variables['height'])}
    root_group = init_file(file_name, dimensions, obj.data)
    add_file_type(root_group, short_id)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height') + copy_from_cat
    copy_variables(obj.dataset, root_group, vars_from_source)
    root_group.title = f"{identifier.capitalize()} file from {obj.dataset.location}"
    root_group.source = f"Categorize file: {product_tools.get_source(obj)}"
    copy_global(obj.dataset, root_group, ('location', 'day', 'month', 'year'))
    merge_history(root_group, identifier, obj)
    root_group.close()


def _get_identifier(short_id):
    valid_ids = ('lwc', 'iwc', 'drizzle', 'classification')
    if short_id not in valid_ids:
        raise ValueError('Invalid product id.')
    if short_id == 'iwc':
        return 'ice water content'
    elif short_id == 'lwc':
        return 'liquid water content'
    return short_id


def merge_history(root_group, file_type, *sources):
    """Merges history fields from one or several files and creates a new record.

    Args:
        root_group (netCDF Dataset): The netCDF Dataset instance.
        file_type (str): Long description of the file.
        *sources (obj): Objects that were used to generate this product. Their
            `history` attribute will be copied to the new product.

    """
    new_record = f"{utils.get_time()} - {file_type} file created"
    old_history = ''
    for source in sources:
        old_history += f"\n{source.dataset.history}"
    root_group.history = f"{new_record}{old_history}"


def init_file(file_name, dimensions, obs, keep_uuid=None):
    """Initializes a Cloudnet file for writing.

    Args:
        file_name (str): File name to be generated.
        dimensions (dict): Dictionary containing dimension for this file.
        obs (dict): Dictionary containing :class:`CloudnetArray` instances.
        keep_uuid (bool, optional): If True and old file with the same name
            exists, uses UUID from that existing file.

    """
    old_id = _get_old_uuid(keep_uuid, file_name)
    root_group = netCDF4.Dataset(file_name, 'w', format='NETCDF4_CLASSIC')
    for key, dimension in dimensions.items():
        root_group.createDimension(key, dimension)
    _write_vars2nc(root_group, obs)
    _add_standard_global_attributes(root_group, old_id)
    return root_group


def _get_old_uuid(keep_uuid, file_name):
    if keep_uuid and os.path.isfile(file_name):
        nc = netCDF4.Dataset(file_name)
        uuid = nc.file_uuid
        nc.close()
        return uuid
    return None


def _write_vars2nc(rootgrp, cloudnet_variables):
    """Iterates over Cloudnet instances and write to given rootgrp."""

    def _get_dimensions(array):
        """Finds correct dimensions for a variable."""
        if utils.isscalar(array):
            return ()
        variable_size = ()
        file_dims = rootgrp.dimensions
        array_dims = array.shape
        for length in array_dims:
            dim = [key for key in file_dims.keys()
                   if file_dims[key].size == length][0]
            variable_size = variable_size + (dim,)
        return variable_size

    for key in cloudnet_variables:
        obj = cloudnet_variables[key]
        size = _get_dimensions(obj.data)
        nc_variable = rootgrp.createVariable(obj.name, obj.data_type, size,
                                             zlib=True)
        nc_variable[:] = obj.data
        for attr in obj.fetch_attributes():
            setattr(nc_variable, attr, getattr(obj, attr))


def _add_standard_global_attributes(root_group, uuid=None):
    root_group.Conventions = 'CF-1.7'
    root_group.cloudnetpy_version = version.__version__
    root_group.file_uuid = uuid or utils.get_uuid()


def copy_variables(source, target, var_list):
    """Copies variables (and their attributes) from one file to another.

    Args:
        source (object): Source object.
        target (object): Target object.
        var_list (list): List of variables to be copied.

    """
    for var_name, variable in source.variables.items():
        if var_name in var_list:
            var_out = target.createVariable(var_name, variable.datatype,
                                            variable.dimensions)
            var_out.setncatts({k: variable.getncattr(k)
                               for k in variable.ncattrs()})
            var_out[:] = variable[:]


def copy_global(source, target, attr_list):
    """Copies global attributes from one file to another.

    Args:
        source (object): Source object.
        target (object): Target object.
        attr_list (list): List of attributes to be copied.

    """
    for attr_name in source.ncattrs():
        if attr_name in attr_list:
            setattr(target, attr_name, source.getncattr(attr_name))


def add_file_type(root_group, file_type):
    """Adds cloudnet_file_type global attribute.

    Args:
        root_group (object): netCDF Dataset instance.
        file_type (str): Name of the Cloudnet file type.

    """
    root_group.cloudnet_file_type = file_type
