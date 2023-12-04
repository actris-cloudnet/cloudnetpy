import tempfile
from typing import Literal

import netCDF4
from mwrpy.level2.lev2_collocated import generate_lev2_multi as gen_multi
from mwrpy.level2.lev2_collocated import generate_lev2_single as gen_single
from mwrpy.version import __version__ as mwrpy_version

from cloudnetpy import output, utils
from cloudnetpy.products import product_tools


def generate_mwr_single(
    mwr_l1c_file: str, output_file: str, uuid: str | None = None
) -> str:
    """
    Generates MWR single-pointing product including liquid water path, integrated
    water vapor, etc. from zenith measurements.

    Args:
        mwr_l1c_file: The Level 1C MWR file to be processed.
        output_file: The file path where the output file should be saved.
        uuid: The UUID, if any, associated with the output file. Defaults to None.

    Returns:
        UUID of generated file.

    Example:
        >>> generate_mwr_single('input_mwr_l1c_file', 'output_file', 'abcdefg1234567')
    """
    return _generate_product(mwr_l1c_file, output_file, uuid, "single")


def generate_mwr_multi(
    mwr_l1c_file: str, output_file: str, uuid: str | None = None
) -> str:
    """
    Generates MWR multiple-pointing product, including relative humidity profiles,
    etc. from scanning measurements.

    Args:
        mwr_l1c_file: The input file in MWR L1C format.
        output_file: The location where the output file should be generated.
        uuid: The UUID for the MWR multi product, defaults to None if
            not provided.

    Returns:
        UUID of generated file.
    """
    return _generate_product(mwr_l1c_file, output_file, uuid, "multi")


def _generate_product(
    mwr_l1c_file: str,
    output_file: str,
    uuid: str | None,
    product: Literal["multi", "single"],
) -> str:
    fun = gen_multi if product == "multi" else gen_single
    with tempfile.TemporaryDirectory() as temp_dir:
        coeffs = product_tools.get_read_mwrpy_coeffs(mwr_l1c_file, temp_dir)
        fun(None, mwr_l1c_file, output_file, coeff_files=coeffs)
    with (
        netCDF4.Dataset(mwr_l1c_file, "r") as nc_input,
        netCDF4.Dataset(output_file, "r+") as nc_output,
    ):
        mwr = Mwr(nc_input, nc_output, uuid)
        return mwr.harmonize(product)


class Mwr:
    def __init__(
        self, nc_l1c: netCDF4.Dataset, nc_l2: netCDF4.Dataset, uuid: str | None
    ):
        self.nc_l1c = nc_l1c
        self.nc_l2 = nc_l2
        self.uuid = uuid if uuid is not None else utils.get_uuid()

    def harmonize(self, product: Literal["multi", "single"]) -> str:
        self._truncate_global_attributes()
        self._copy_variable_values()
        self._copy_global_attributes()
        self._fix_variable_attributes()
        self._write_missing_global_attributes(product)
        return self.uuid

    def _truncate_global_attributes(self):
        for attr in self.nc_l2.ncattrs():
            delattr(self.nc_l2, attr)

    def _copy_variable_values(self):
        keys = ("latitude", "longitude", "altitude")
        for var in keys:
            if var in self.nc_l2.variables:
                self.nc_l2.variables[var][:] = self.nc_l1c.variables[var][:]

    def _copy_global_attributes(self):
        keys = ("year", "month", "day", "location", "source")
        output.copy_global(self.nc_l1c, self.nc_l2, keys)

    def _fix_variable_attributes(self):
        output.replace_attribute_with_standard_value(
            self.nc_l2,
            (
                "lwp",
                "iwv",
                "temperature",
                "azimuth_angle",
                "latitude",
                "longitude",
                "altitude",
            ),
            ("units", "long_name", "standard_name"),
        )

    def _write_missing_global_attributes(self, product: Literal["multi", "single"]):
        output.add_standard_global_attributes(self.nc_l2, self.uuid)
        product_type = "multiple-pointing" if product == "multi" else "single-pointing"
        self.nc_l2.title = f"MWR {product_type} from {self.nc_l1c.location}"
        self.nc_l2.cloudnet_file_type = f"mwr-{product}"
        output.fix_time_attributes(self.nc_l2)
        self.nc_l2.history = (
            f"{utils.get_time()} - MWR {product_type} file created \n"
            f"{self.nc_l1c.history}"
        )
        self.nc_l2.source_file_uuids = self.nc_l1c.file_uuid
        self.nc_l2.mwrpy_version = mwrpy_version
