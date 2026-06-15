import os
from datetime import datetime
from os import PathLike
from uuid import UUID

import netCDF4

from cloudnetpy import output
from cloudnetpy.model_evaluation.model_metadata import MODEL_PREFIX, PRODUCT_NAMES

from .metadata import (
    CYCLE_ATTRIBUTES,
    MODEL_ATTRIBUTES,
    MODEL_L3_ATTRIBUTES,
    REGRID_PRODUCT_ATTRIBUTES,
)


def update_attributes(model_downsample_variables: dict, attributes: dict) -> None:
    """Sets variable attributes for the L3 downsampled file.

    Model (simulated) fields are prefixed with ``model_``; observation fields
    downsampled to the model grid use the bare product key.

    Args:
        model_downsample_variables (dict): Array instances.
        attributes (dict): Product-specific attributes (e.g. time units).
    """
    for key, variable in model_downsample_variables.items():
        if key in attributes:
            variable.set_attributes(attributes[key])
        if key in MODEL_ATTRIBUTES:
            variable.set_attributes(MODEL_ATTRIBUTES[key])
        elif key.startswith(MODEL_PREFIX):
            base = key.removeprefix(MODEL_PREFIX)
            if base in MODEL_L3_ATTRIBUTES:
                variable.set_attributes(MODEL_L3_ATTRIBUTES[base])
            elif base in CYCLE_ATTRIBUTES:
                variable.set_attributes(CYCLE_ATTRIBUTES[base])
        elif key in REGRID_PRODUCT_ATTRIBUTES:
            variable.set_attributes(REGRID_PRODUCT_ATTRIBUTES[key])


def save_downsampled_file(
    id_mark: str,
    file_name: str | PathLike,
    objects: tuple,
    files: tuple[list[str | PathLike], str | PathLike],
    uuid: UUID,
    model_name: str | None = None,
    site_name: str | None = None,
) -> None:
    """Saves a standard downsampled day product file.

    Args:
        id_mark (str): File identifier, format "(product name)_(model name)"
        file_name (str): Name of the output file to be generated
        objects (tuple): Include two objects: The :class:'ModelManager' and
                      The :class:'ObservationManager.
        files (tuple): Includes two sourcefile group: List of model file(s) used
                       for processing output file and Cloudnet L2 product file
        model_name (str): Human-readable model name for plot titles. Falls back
                       to the model id when not given.
        site_name (str): Human-readable site name for the location attribute and
                       plot subtitle. Falls back to the source file's location
                       when not given.
        keep_uuid (bool): If True, keeps the UUID of the old file, if that exists.
                          Default is False when new UUID is generated.
        uuid (str): Set specific UUID for the file.
    """
    obj = objects[0]
    n_levels = obj.data[obj.keys["height"]][:].shape[-1]
    dimensions = {"time": len(obj.time), "level": n_levels}
    location = site_name or obj.dataset.location
    with output.init_file(file_name, dimensions, obj.data, uuid) as root_group:
        _augment_global_attributes(root_group)
        product = id_mark.split("_", maxsplit=1)[0]
        root_group.cloudnet_file_type = "l3-" + product
        product_name = PRODUCT_NAMES.get(product, product)
        root_group.title = f"Observed and modeled {product_name} over {location}"
        root_group.model_id = obj.model
        root_group.model_name = model_name or obj.model
        _add_source(root_group, objects, files)
        output.copy_global(obj.dataset, root_group, ("day", "month", "year"))
        root_group.location = location
        if not hasattr(obj.dataset, "day"):
            root_group.year, root_group.month, root_group.day = obj.date
        output.merge_history(root_group, f"L3 {product_name}", obj)


def _augment_global_attributes(root_group: netCDF4.Dataset) -> None:
    root_group.Conventions = "CF-1.8"


def _add_source(root_ground: netCDF4.Dataset, objects: tuple, files: tuple) -> None:
    """Generates source info for multiple files."""
    model, obs = objects
    model_files, obs_file = files
    filenames = [os.path.basename(obs_file)] + [
        os.path.basename(f) for f in model_files
    ]
    root_ground.source = "\n".join(filenames)
    root_ground.source_file_uuids = output.get_source_uuids([model, obs])


def add_time_attribute(date: datetime) -> dict:
    """Adds time attribute with correct units.

    Args:
        attributes: Attributes of variables.
        date: Date as Y M D 0 0 0.

    Returns:
        dict: Same attributes with 'time' attribute added.
    """
    return {
        "time": MODEL_ATTRIBUTES["time"]._replace(
            units=f"hours since {date:%Y-%m-%d} 00:00:00 +00:00",
        ),
    }
