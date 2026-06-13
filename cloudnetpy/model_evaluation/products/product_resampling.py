import logging
from os import PathLike
from uuid import UUID

import cloudnetpy.model_evaluation.products.tools as tl
from cloudnetpy.model_evaluation.file_handler import (
    add_time_attribute,
    save_downsampled_file,
    update_attributes,
)
from cloudnetpy.model_evaluation.products.advance_methods import AdvanceProductMethods
from cloudnetpy.model_evaluation.products.grid_methods import ProductGrid
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager
from cloudnetpy.model_evaluation.utils import file_exists
from cloudnetpy.utils import get_uuid


def process_L3_day_product(
    model: str,
    obs: str,
    model_file: str | PathLike,
    product_file: str | PathLike,
    output_file: str | PathLike,
    uuid: str | UUID | None = None,
    *,
    overwrite: bool = False,
) -> UUID:
    """Generate a downsampling of observations to match the model grid.

    Generates an L3 product nc-file for a single model run. It contains the
    model's own fields (prefixed with ``model_``) and the observation products
    downsampled onto the model grid. The model identity is stored in the file's
    global attributes, not in the variable names.

    Args:
        model (str): Name of model
        obs (str): Name of product to generate
        model_file (str): Model file path
        product_file (str): Source file path of L2 observation product
        output_file (str): Path and name of L3 day scale product output file
        uuid (str): Set specific UUID for the file.
        overwrite (bool): Recreate the output file if it already exists.

    Raises:
        RuntimeError: Failed to create the L3 product file.
        ValueError (Warning): No ice clouds in model data
        FileExistsError: Output file exists and overwrite is False.

    Examples:
        >>> from cloudnetpy.model_evaluation.products.product_resampling import \
        process_L3_day_product
        >>> process_L3_day_product('ecmwf', 'cf', 'ecmwf.nc',
        ... 'categorize.nc', 'l3-cf.nc')
    """
    uuid = get_uuid(uuid)
    if file_exists(output_file) and not overwrite:
        msg = f"Output file {output_file} exists, use overwrite=True to recreate"
        raise FileExistsError(msg)
    product_obj = ObservationManager(obs, product_file)
    model_obj = ModelManager(model_file, model, obs)
    try:
        AdvanceProductMethods(model_obj, model_file, product_obj)
    except ValueError as e:
        logging.info(e)
    ProductGrid(model_obj, product_obj)
    attributes = add_time_attribute(product_obj.date)
    update_attributes(model_obj.data, attributes)
    tl.add_date(model_obj, product_obj)
    save_downsampled_file(
        f"{obs}_{model}",
        output_file,
        (model_obj, product_obj),
        ([model_file], product_file),
        uuid,
    )
    return uuid
