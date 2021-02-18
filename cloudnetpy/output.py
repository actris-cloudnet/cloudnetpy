""" Functions for file writing."""
import os
from typing import Union, Optional
import numpy as np
import netCDF4
from cloudnetpy import utils, version
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData
from cloudnetpy.instruments.mira import Mira
from cloudnetpy.instruments.basta import Basta


def save_radar_level1b(source_full_path: str,
                       radar: Union[Basta, Mira],
                       output_file: str,
                       keep_uuid: Union[bool, None],
                       uuid: Union[str, None],
                       vars_from_source: Optional[tuple] = ()) -> str:
    """Saves pre-processed cloud radar data to a Cloudnet Level 1b "radar" file."""
    file_type = 'radar'
    dimensions = {'time': len(radar.time),
                  'range': len(radar.range)}

    nc = init_file(output_file, dimensions, radar.data, keep_uuid, uuid)
    uuid = nc.file_uuid
    add_file_type(nc, file_type)
    nc_source = netCDF4.Dataset(source_full_path)
    copy_variables(nc_source, nc, vars_from_source)
    nc.title = f"{file_type.capitalize()} file from {radar.location}"
    nc.year, nc.month, nc.day = radar.date
    nc.location = radar.location
    nc.history = f"{utils.get_time()} - {file_type} file created"
    nc.source = radar.source
    add_references(nc)
    nc.close()
    nc_source.close()
    return uuid


def save_product_file(short_id: str,
                      obj,
                      file_name: str,
                      keep_uuid: bool,
                      uuid: Union[str, None],
                      copy_from_cat: Optional[tuple] = ()) -> str:
    """Saves a standard Cloudnet product file.

    Args:
        short_id: Short file identifier, e.g. 'lwc', 'iwc', 'drizzle', 'classification'.
        obj: Instance containing product specific attributes: `time`, `dataset`, `data`.
        file_name: Name of the output file to be generated.
        keep_uuid: If True and old file with the same name exists, uses UUID from that
            existing file.
        uuid: Set specific UUID for the file.
        copy_from_cat: Variables to be copied from the categorize file.

    """
    human_readable_file_type = _get_identifier(short_id)
    dimensions = {'time': len(obj.time),
                  'height': len(obj.dataset.variables['height'])}
    nc = init_file(file_name, dimensions, obj.data, keep_uuid, uuid)
    uuid = nc.file_uuid
    add_file_type(nc, short_id)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height') + copy_from_cat
    copy_variables(obj.dataset, nc, vars_from_source)
    nc.title = f"{human_readable_file_type.capitalize()} file from {obj.dataset.location}"
    nc.source_file_uuids = get_source_uuids(nc, obj)
    copy_global(obj.dataset, nc, ('location', 'day', 'month', 'year'))
    merge_history(nc, human_readable_file_type, obj)
    add_references(nc, short_id)
    nc.close()
    return uuid


def get_source_uuids(*sources) -> str:
    """Returns file_uuid attributes of objects.

    Args:
        *sources: Objects whose file_uuid attributes are read (if exist).

    Returns:
        str: UUIDs separated by comma.

    """
    uuids = [source.dataset.file_uuid for source in sources if hasattr(source, 'dataset')
             and hasattr(source.dataset, 'file_uuid')]
    unique_uuids = list(set(uuids))
    return ', '.join(unique_uuids)


def merge_history(nc: netCDF4.Dataset, file_type: str, *sources) -> None:
    """Merges history fields from one or several files and creates a new record.

    Args:
        nc: The netCDF Dataset instance.
        file_type: Long description of the file.
        *sources: Objects that were used to generate this product. Their `history` attribute will
            be copied to the new product.

    """
    new_record = f"{utils.get_time()} - {file_type} file created"
    old_history = ''
    for source in sources:
        old_history += f"\n{source.dataset.history}"
    nc.history = f"{new_record}{old_history}"


def init_file(file_name: str,
              dimensions: dict,
              cloudnet_arrays: dict,
              keep_uuid: Optional[bool] = None,
              uuid: Union[str, None] = None) -> netCDF4.Dataset:
    """Initializes a Cloudnet file for writing.

    Args:
        file_name: File name to be generated.
        dimensions: Dictionary containing dimension for this file.
        cloudnet_arrays: Dictionary containing :class:`CloudnetArray` instances.
        keep_uuid: If True and old file with the same name exists, uses UUID from that file.
        uuid: Set specific UUID for the file.

    """
    specific_uuid = uuid or _get_old_uuid(keep_uuid, file_name)
    nc = netCDF4.Dataset(file_name, 'w', format='NETCDF4_CLASSIC')
    for key, dimension in dimensions.items():
        nc.createDimension(key, dimension)
    _write_vars2nc(nc, cloudnet_arrays)
    _add_standard_global_attributes(nc, specific_uuid)
    return nc


def copy_variables(source: netCDF4.Dataset,
                   target: netCDF4.Dataset,
                   keys: tuple) -> None:
    """Copies variables (and their attributes) from one file to another.

    Args:
        source: Source object.
        target: Target object.
        keys: Variable names to be copied.

    """
    for key in keys:
        if key in source.variables:
            variable = source.variables[key]
            var_out = target.createVariable(key, variable.datatype, variable.dimensions)
            var_out.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            var_out[:] = variable[:]


def copy_global(source: netCDF4.Dataset,
                target: netCDF4.Dataset,
                attributes: tuple) -> None:
    """Copies global attributes from one file to another.

    Args:
        source: Source object.
        target: Target object.
        attributes: List of attributes to be copied.

    """
    source_attributes = source.ncattrs()
    for attr in attributes:
        if attr in source_attributes:
            setattr(target, attr, source.getncattr(attr))


def add_file_type(nc: netCDF4.Dataset, file_type: str) -> None:
    """Adds cloudnet_file_type global attribute.

    Args:
        nc: netCDF Dataset instance.
        file_type: Name of the Cloudnet file type.

    """
    nc.cloudnet_file_type = file_type


def add_references(nc: netCDF4.Dataset, identifier: Optional[str] = None) -> None:
    """Adds references attribute to netCDF file.

    Args:
        nc: netCDF Dataset instance.
        identifier: Cloudnet file type, e.g., 'iwc'.

    """
    references = 'https://doi.org/10.21105/joss.02123'
    if identifier:
        if identifier in ('lwc', 'categorize'):
            references += ', https://doi.org/10.1175/BAMS-88-6-883'
        if identifier == 'iwc':
            references += ', https://doi.org/10.1175/JAM2340.1'
        if identifier == 'drizzle':
            references += ', https://doi.org/10.1175/JAM-2181.1'
    nc.references = references


def add_time_attribute(attributes: dict, date: list) -> dict:
    """"Adds time attribute with correct units.

    Args:
        attributes: Attributes of variables.
        date: Date as ['YYYY', 'MM', 'DD'].

    Returns:
        dict: Same attributes with 'time' attribute added.

    """
    date = '-'.join(date)
    attributes['time'] = MetaData(units=f'hours since {date} 00:00:00')
    return attributes


def update_attributes(cloudnet_variables: dict, attributes: dict) -> None:
    """Overrides existing CloudnetArray-attributes.

    Overrides existing attributes using hard-coded values.
    New attributes are added.

    Args:
        cloudnet_variables: CloudnetArray instances.
        attributes: Product-specific attributes.

    """
    for key in cloudnet_variables:
        if key in attributes:
            cloudnet_variables[key].set_attributes(attributes[key])
        if key in COMMON_ATTRIBUTES:
            cloudnet_variables[key].set_attributes(COMMON_ATTRIBUTES[key])


def _write_vars2nc(nc: netCDF4.Dataset, cloudnet_variables: dict) -> None:
    """Iterates over Cloudnet instances and write to netCDF file."""
    for obj in cloudnet_variables.values():
        size = _get_dimensions(nc, obj.data)
        nc_variable = nc.createVariable(obj.name, obj.data_type, size, zlib=True)
        nc_variable[:] = obj.data
        for attr in obj.fetch_attributes():
            setattr(nc_variable, attr, getattr(obj, attr))


def _get_dimensions(nc: netCDF4.Dataset, data: np.ndarray) -> tuple:
    """Finds correct dimensions for a variable."""
    if utils.isscalar(data):
        return ()
    variable_size = ()
    file_dims = nc.dimensions
    array_dims = data.shape
    for length in array_dims:
        dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
        variable_size = variable_size + (dim,)
    return variable_size


def _get_identifier(short_id: str) -> str:
    valid_ids = ('lwc', 'iwc', 'drizzle', 'classification')
    if short_id not in valid_ids:
        raise ValueError('Invalid product id.')
    if short_id == 'iwc':
        return 'ice water content'
    elif short_id == 'lwc':
        return 'liquid water content'
    return short_id


def _add_standard_global_attributes(nc: netCDF4.Dataset,
                                    uuid: Union[str, None] = None) -> None:
    nc.Conventions = 'CF-1.7'
    nc.cloudnetpy_version = version.__version__
    nc.file_uuid = uuid or utils.get_uuid()


def _get_old_uuid(keep_uuid: bool, full_path: str) -> Union[str, None]:
    if keep_uuid and os.path.isfile(full_path):
        nc = netCDF4.Dataset(full_path)
        uuid = nc.file_uuid
        nc.close()
        return uuid
    return None
