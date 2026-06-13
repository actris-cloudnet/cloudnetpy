import os
from datetime import datetime
from os import PathLike
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from cloudnetpy.model_evaluation.products.model_products import ModelManager
    from cloudnetpy.model_evaluation.products.observation_products import (
        ObservationManager,
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
    product: str,
    file_name: str | PathLike,
    model_obj: "ModelManager",
    obs_obj: "ObservationManager",
    model_files: list[str | PathLike],
    product_file: str | PathLike,
    uuid: UUID,
    model_name: str | None = None,
    site_name: str | None = None,
) -> None:
    """Saves a standard downsampled day product file.

    Args:
        product (str): Product name, e.g. "cf".
        file_name (str): Name of the output file to be generated.
        model_obj (ModelManager): Model fields downsampled to the model grid.
        obs_obj (ObservationManager): Cloudnet L2 observation product.
        model_files (list): Model file(s) used for processing the output file.
        product_file (str): Cloudnet L2 product file.
        uuid (str): Set specific UUID for the file.
        model_name (str): Human-readable model name for plot titles. Falls back
                       to the model id when not given.
        site_name (str): Human-readable site name for the location attribute and
                       plot subtitle. Falls back to the source file's location
                       when not given.
    """
    n_levels = model_obj.data[model_obj.keys["height"]][:].shape[-1]
    dimensions = {"time": len(model_obj.time), "level": n_levels}
    location = site_name or model_obj.dataset.location
    with output.init_file(file_name, dimensions, model_obj.data, uuid) as root_group:
        _augment_global_attributes(root_group)
        root_group.cloudnet_file_type = "l3-" + product
        product_name = PRODUCT_NAMES.get(product, product)
        root_group.title = f"Observed and modeled {product_name} over {location}"
        root_group.model_id = model_obj.model
        root_group.model_name = model_name or model_obj.model
        _add_source(root_group, model_obj, obs_obj, model_files, product_file)
        output.copy_global(model_obj.dataset, root_group, ("day", "month", "year"))
        root_group.location = location
        if not hasattr(model_obj.dataset, "day"):
            root_group.year, root_group.month, root_group.day = model_obj.date
        output.merge_history(root_group, f"L3 {product_name}", model_obj)


def _augment_global_attributes(root_group: netCDF4.Dataset) -> None:
    root_group.Conventions = "CF-1.8"


def _add_source(
    root_group: netCDF4.Dataset,
    model_obj: "ModelManager",
    obs_obj: "ObservationManager",
    model_files: list[str | PathLike],
    product_file: str | PathLike,
) -> None:
    """Generates source info for multiple files."""
    filenames = [os.path.basename(product_file)] + [
        os.path.basename(f) for f in model_files
    ]
    root_group.source = "\n".join(filenames)
    root_group.source_file_uuids = output.get_source_uuids([model_obj, obs_obj])


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
