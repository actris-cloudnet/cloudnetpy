import logging

import cloudnetpy.model_evaluation.products.tools as tl
from cloudnetpy.model_evaluation.file_handler import (
    add_time_attribute,
    add_var2ncfile,
    save_downsampled_file,
    update_attributes,
)
from cloudnetpy.model_evaluation.products.advance_methods import AdvanceProductMethods
from cloudnetpy.model_evaluation.products.grid_methods import ProductGrid
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager
from cloudnetpy.model_evaluation.utils import file_exists


def process_L3_day_product(
    model: str,
    obs: str,
    model_files: list,
    product_file: str,
    output_file: str,
    uuid: str | None = None,
    *,
    overwrite: bool = False,
) -> str:
    """Main function to generate downsample of observations to match model grid.

    This function will generate a L3 product nc-file. It includes the information of
    downsampled observation products for each model cycles and model products
    and other variables of each cycles.

    Args:
        model (str): Name of model
        obs (str): Name of product to generate
        model_files (list): List of model + cycles file path(s) to be generated
        product_file (str): Source file path of L2 observation product
        output_file (str): Path and name of L3 day scale product output file
        keep_uuid (bool): If True, keeps the UUID of the old file, if that exists.
                          Default is False when new UUID is generated.
        uuid (str): Set specific UUID for the file.
        overwrite (bool): If file exists, but still want to recreate it then True,
                          default False

    Raises:
        RuntimeError: Failed to create the L3 product file.
        ValueError (Warning): No ice clouds in model data

    Notes:
        Model file(s) are given as a list to make all different cycles to be at same
        nc-file. If list includes more than one model file, nc-file is created within
        the first round. With rest of rounds, downsample observation and model data
        is added to same L3 day nc-file.

    Examples:
        >>> from cloudnetpy.model_evaluation.products.product_resampling import \
        process_L3_day_product
        >>> product = 'cf'
        >>> model = 'ecmwf'
        >>> model_file = 'ecmwf.nc'
        >>> input_file = 220190517_mace-head_categorize.nchead_categorize.nc
        >>> output_file = 'cf_ecmwf.nc'
        >>> process_L3_day_product(model, product, [model_file], input_file,
        output_file)
    """
    product_obj = ObservationManager(obs, product_file)
    tl.check_model_file_list(model, model_files)
    for m_file in model_files:
        model_obj = ModelManager(
            m_file,
            model,
            output_file,
            obs,
            check_file=not overwrite,
        )
        try:
            AdvanceProductMethods(model_obj, m_file, product_obj)
        except ValueError as e:
            logging.info(e)
        ProductGrid(model_obj, product_obj)
        attributes = add_time_attribute(product_obj.date)
        update_attributes(model_obj.data, attributes)
        if not file_exists(output_file) or overwrite:
            tl.add_date(model_obj, product_obj)
            uuid_out = save_downsampled_file(
                f"{obs}_{model}",
                output_file,
                (model_obj, product_obj),
                (model_files, product_file),
                uuid,
            )
        else:
            add_var2ncfile(model_obj, output_file)
    return uuid_out
