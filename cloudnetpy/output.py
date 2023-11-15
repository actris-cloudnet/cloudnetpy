"""Functions for file writing."""
import datetime
import logging
from os import PathLike
from uuid import UUID

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import utils, version
from cloudnetpy.instruments.instruments import Instrument
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData


def save_level1b(
    obj,
    output_file: PathLike | str,
    uuid: UUID | str | None = None,
) -> str:
    """Saves Cloudnet Level 1b file."""
    dimensions = _get_netcdf_dimensions(obj)
    with init_file(output_file, dimensions, obj.data, uuid) as nc:
        file_uuid = nc.file_uuid
        fix_attribute_name(nc)
        location = obj.site_meta["name"]
        nc.cloudnet_file_type = obj.instrument.domain
        nc.title = get_l1b_title(obj.instrument, location)
        if isinstance(obj.date, list):
            nc.year, nc.month, nc.day = obj.date
        elif isinstance(obj.date, datetime.date):
            nc.year = str(obj.date.year)
            nc.month = str(obj.date.month).zfill(2)
            nc.day = str(obj.date.day).zfill(2)
        else:
            raise TypeError
        nc.location = location
        nc.history = get_l1b_history(obj.instrument)
        nc.source = get_l1b_source(obj.instrument)
        if hasattr(obj, "serial_number") and obj.serial_number is not None:
            nc.serial_number = obj.serial_number
        nc.references = get_references()
    return file_uuid


def _get_netcdf_dimensions(obj) -> dict:
    dimensions = {
        key: len(obj.data[key][:]) for key in ("time", "range") if key in obj.data
    }
    # RPG cloud radar
    if "chirp_start_indices" in obj.data:
        dimensions["chirp_sequence"] = len(obj.data["chirp_start_indices"][:])
    # disdrometer
    if hasattr(obj, "n_diameter") and hasattr(obj, "n_velocity"):
        dimensions["diameter"] = obj.n_diameter
        dimensions["velocity"] = obj.n_velocity
        dimensions["nv"] = 2
    # HATPRO l1c
    if "tb" in obj.data:
        dimensions["frequency"] = obj.data["tb"][:].shape[1]
        dimensions["receiver_nb"] = len(obj.data["receiver_nb"][:])
        dimensions["band"] = 2
        dimensions["t_amb_nb"] = 2
    if "irt" in obj.data:
        dimensions["ir_channel"] = obj.data["irt"][:].shape[1]
    return dimensions


def save_product_file(
    short_id: str,
    obj,
    file_name: str,
    uuid: str | None = None,
    copy_from_cat: tuple = (),
) -> str:
    """Saves a standard Cloudnet product file.

    Args:
    ----
        short_id: Short file identifier, e.g. 'lwc', 'iwc', 'drizzle', 'classification'.
        obj: Instance containing product specific attributes: `time`, `dataset`, `data`.
        file_name: Name of the output file to be generated.
        uuid: Set specific UUID for the file.
        copy_from_cat: Variables to be copied from the categorize file.

    """
    human_readable_file_type = _get_identifier(short_id)
    dimensions = {
        "time": len(obj.time),
        "height": len(obj.dataset.variables["height"]),
    }
    with init_file(file_name, dimensions, obj.data, uuid) as nc:
        file_uuid = nc.file_uuid
        nc.cloudnet_file_type = short_id
        vars_from_source = (
            "altitude",
            "latitude",
            "longitude",
            "time",
            "height",
            *copy_from_cat,
        )
        copy_variables(obj.dataset, nc, vars_from_source)
        nc.title = (
            f"{human_readable_file_type.capitalize()} products from"
            f" {obj.dataset.location}"
        )
        nc.source_file_uuids = get_source_uuids(nc, obj)
        copy_global(
            obj.dataset,
            nc,
            ("location", "day", "month", "year", "source", "voodoonet_version"),
        )
        merge_history(nc, human_readable_file_type, {"categorize": obj})
        nc.references = get_references(short_id)
    return file_uuid


def get_l1b_source(instrument: Instrument) -> str:
    """Returns level 1b file source."""
    return " ".join(
        item for item in [instrument.manufacturer, instrument.model] if item is not None
    )


def get_l1b_history(instrument: Instrument) -> str:
    """Returns level 1b file history."""
    return f"{utils.get_time()} - {instrument.domain} file created"


def get_l1b_title(instrument: Instrument, location: str) -> str:
    """Returns level 1b file title."""
    if instrument.model:
        prefix = " ".join(
            item for item in [instrument.model, instrument.category] if item is not None
        )
    else:
        prefix = instrument.category.capitalize()
    return f"{prefix} from {location}"


def get_references(identifier: str | None = None, extra: list | None = None) -> str:
    """ "Returns references.

    Args:
    ----
        identifier: Cloudnet file type, e.g., 'iwc'.

    """
    references = "https://doi.org/10.21105/joss.02123"
    match identifier:
        case "der":
            references += (
                ", https://doi.org/10.1175/1520-0426(2002)019<0835:TROSCD>2.0.CO;2"
            )
        case "ier":
            references += (
                ", https://doi.org/10.1175/JAM2340.1"
                ", https://doi.org/10.1175/JAM2543.1"
                ", https://doi.org/10.5194/amt-13-5335-2020"
            )
        case "lwc" | "categorize":
            references += ", https://doi.org/10.1175/BAMS-88-6-883"
        case "iwc":
            references += ", https://doi.org/10.1175/JAM2340.1"
        case "drizzle":
            references += ", https://doi.org/10.1175/JAM-2181.1"
    if extra is not None:
        for reference in extra:
            references += f", {reference}"
    return references


def get_source_uuids(*sources) -> str:
    """Returns file_uuid attributes of objects.

    Args:
    ----
        *sources: Objects whose file_uuid attributes are read (if exist).

    Returns:
    -------
        str: UUIDs separated by comma.

    """
    uuids = [
        source.dataset.file_uuid
        for source in sources
        if hasattr(source, "dataset") and hasattr(source.dataset, "file_uuid")
    ]
    unique_uuids = list(set(uuids))
    return ", ".join(unique_uuids)


def merge_history(nc: netCDF4.Dataset, file_type: str, data: dict) -> None:
    """Merges history fields from one or several files and creates a new record.

    Args:
    ----
        nc: The netCDF Dataset instance.
        file_type: Long description of the file.
        data: Dictionary of objects with history attribute.

    """
    new_record = f"{utils.get_time()} - {file_type} file created"
    histories = []
    for key, obj in data.items():
        if (
            not isinstance(obj, str | list)
            and obj is not None
            and hasattr(obj.dataset, "history")
        ):
            history = obj.dataset.history
            history = history.split("\n")[-1] if key == "model" else history
            histories.append(history)
    histories.sort(reverse=True)
    old_history = [f"\n{history}" for history in histories]
    old_history_str = "".join(old_history)
    nc.history = f"{new_record}{old_history_str}"


def add_source_instruments(nc: netCDF4.Dataset, data: dict) -> None:
    """Adds source attribute to categorize file."""
    sources = [obj.source for obj in data.values() if hasattr(obj, "source")]
    sources = [sources[0]] + [f"\n{source}" for source in sources[1:]]
    nc.source = "".join(sources)


def init_file(
    file_name: PathLike | str,
    dimensions: dict,
    cloudnet_arrays: dict,
    uuid: UUID | str | None = None,
) -> netCDF4.Dataset:
    """Initializes a Cloudnet file for writing.

    Args:
    ----
        file_name: File name to be generated.
        dimensions: Dictionary containing dimension for this file.
        cloudnet_arrays: Dictionary containing :class:`CloudnetArray` instances.
        uuid: Set specific UUID for the file.

    """
    nc = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    for key, dimension in dimensions.items():
        nc.createDimension(key, dimension)
    _write_vars2nc(nc, cloudnet_arrays)
    add_standard_global_attributes(nc, uuid)
    return nc


def copy_variables(
    source: netCDF4.Dataset,
    target: netCDF4.Dataset,
    keys: tuple,
) -> None:
    """Copies variables (and their attributes) from one file to another.

    Args:
    ----
        source: Source object.
        target: Target object.
        keys: Variable names to be copied.

    """
    for key in keys:
        if key in source.variables:
            fill_value = getattr(source.variables[key], "_FillValue", False)
            variable = source.variables[key]
            var_out = target.createVariable(
                key,
                variable.datatype,
                variable.dimensions,
                fill_value=fill_value,
            )
            var_out.setncatts(
                {
                    k: variable.getncattr(k)
                    for k in variable.ncattrs()
                    if k != "_FillValue"
                },
            )
            var_out[:] = variable[:]


def copy_global(
    source: netCDF4.Dataset,
    target: netCDF4.Dataset,
    attributes: tuple,
) -> None:
    """Copies global attributes from one file to another.

    Args:
    ----
        source: Source object.
        target: Target object.
        attributes: List of attributes to be copied.

    """
    source_attributes = source.ncattrs()
    for attr in attributes:
        if attr in source_attributes:
            setattr(target, attr, source.getncattr(attr))


def add_time_attribute(
    attributes: dict,
    date: list[str] | datetime.date,
    key: str = "time",
) -> dict:
    """Adds time attribute with correct units."""
    if isinstance(date, list):
        date_str = "-".join(date)
    elif isinstance(date, datetime.date):
        date_str = date.isoformat()
    else:
        raise TypeError
    units = f"hours since {date_str} 00:00:00 +00:00"
    if key not in attributes:
        attributes[key] = MetaData(units=units)
    else:
        attributes[key] = attributes[key]._replace(units=units)
    return attributes


def add_source_attribute(attributes: dict, data: dict) -> dict:
    """Adds source attribute."""
    variables = {
        "radar": (
            "v",
            "width",
            "v_sigma",
            "ldr",
            "Z",
            "zdr",
            "sldr",
            "radar_frequency",
            "nyquist_velocity",
            "rainfall_rate",
        ),
        "lidar": ("beta", "lidar_wavelength"),
        "mwr": ("lwp",),
        "model": ("uwind", "vwind", "Tw", "q", "pressure", "temperature"),
    }
    for instrument, keys in variables.items():
        source = data[instrument].dataset.source
        for key in keys:
            if key in attributes:
                attributes[key] = attributes[key]._replace(source=source)
            else:
                attributes[key] = MetaData(source=source)
    return attributes


def update_attributes(cloudnet_variables: dict, attributes: dict) -> None:
    """Overrides existing CloudnetArray-attributes.

    Overrides existing attributes using hard-coded values.
    New attributes are added.

    Args:
    ----
        cloudnet_variables: CloudnetArray instances.
        attributes: Product-specific attributes.

    """
    for key in cloudnet_variables:
        if key in COMMON_ATTRIBUTES:
            cloudnet_variables[key].set_attributes(COMMON_ATTRIBUTES[key])
        if key in attributes:
            cloudnet_variables[key].set_attributes(attributes[key])


def _write_vars2nc(nc: netCDF4.Dataset, cloudnet_variables: dict) -> None:
    """Iterates over Cloudnet instances and write to netCDF file."""
    for obj in cloudnet_variables.values():
        if ma.isMaskedArray(obj.data):
            fill_value = netCDF4.default_fillvals[obj.data_type]
        else:
            fill_value = False

        size = obj.dimensions or _get_dimensions(nc, obj.data)

        nc_variable = nc.createVariable(
            obj.name,
            obj.data_type,
            size,
            zlib=True,
            fill_value=fill_value,
        )
        nc_variable[:] = obj.data
        for attr in obj.fetch_attributes():
            setattr(nc_variable, attr, getattr(obj, attr))


def _get_dimensions(nc: netCDF4.Dataset, data: np.ndarray) -> tuple:
    """Finds correct dimensions for a variable."""
    if utils.isscalar(data):
        return ()
    variable_size: list = []
    file_dims = nc.dimensions
    array_dims = data.shape
    for length in array_dims:
        dim = [key for key in file_dims if file_dims[key].size == length][0]  # noqa: RUF015
        variable_size = [*variable_size, dim]
    return tuple(variable_size)


def _get_identifier(short_id: str) -> str:
    valid_ids = ("lwc", "iwc", "drizzle", "classification", "der", "ier")
    if short_id not in valid_ids:
        msg = f"Invalid file identifier: {short_id}"
        raise ValueError(msg)
    if short_id == "iwc":
        return "ice water content"
    if short_id == "lwc":
        return "liquid water content"
    if short_id == "ier":
        return "ice effective radius"
    if short_id == "der":
        return "droplet effective radius"
    return short_id


def add_standard_global_attributes(
    nc: netCDF4.Dataset,
    uuid: UUID | str | None = None,
) -> None:
    nc.Conventions = "CF-1.8"
    nc.cloudnetpy_version = version.__version__
    nc.file_uuid = str(uuid) if uuid is not None else utils.get_uuid()


def fix_attribute_name(nc: netCDF4.Dataset) -> None:
    """Changes incorrect 'unit' variable attribute to correct 'units'.

    This is true at least for 'drg' variable in raw MIRA files.

    """
    for var in nc.variables:
        if "unit" in nc[var].ncattrs():
            logging.info('Renaming "unit" attribute into "units"')
            nc[var].setncattr("units", nc[var].unit)
            nc[var].delncattr("unit")


def fix_time_attributes(nc: netCDF4.Dataset) -> None:
    nc.variables["time"].standard_name = "time"
    nc.variables["time"].long_name = "Time UTC"
    nc.variables["time"].calendar = "standard"
    nc.variables[
        "time"
    ].units = f"hours since {nc.year}-{nc.month}-{nc.day} 00:00:00 +00:00"


def replace_attribute_with_standard_value(
    nc: netCDF4.Dataset,
    variables: tuple,
    attributes: tuple,
) -> None:
    for key in variables:
        if key in COMMON_ATTRIBUTES and key in nc.variables:
            for attr in attributes:
                if (value := getattr(COMMON_ATTRIBUTES[key], attr)) is not None:
                    setattr(nc.variables[key], attr, value)
