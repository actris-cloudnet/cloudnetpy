import tempfile
from tempfile import NamedTemporaryFile

import netCDF4
from mwrpy.level2.write_lev2_nc import MissingInputData, lev2_to_nc
from mwrpy.version import __version__ as mwrpy_version

from cloudnetpy import output, utils
from cloudnetpy.exceptions import MissingInputFileError
from cloudnetpy.products import product_tools


def generate_mwr_multi(
    mwr_l1c_file: str,
    output_file: str,
    uuid: str | None = None,
) -> str:
    file_uuid = uuid if uuid is not None else utils.get_uuid()

    with (
        NamedTemporaryFile() as temp_file,
        NamedTemporaryFile() as abs_hum_file,
        NamedTemporaryFile() as rel_hum_file,
        NamedTemporaryFile() as t_pot_file,
        NamedTemporaryFile() as eq_temp_file,
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        coeffs = product_tools.get_read_mwrpy_coeffs(mwr_l1c_file, temp_dir)

        for prod, file in zip(
            ("2P02", "2P03", "2P04", "2P07", "2P08"),
            (temp_file, abs_hum_file, rel_hum_file, t_pot_file, eq_temp_file),
            strict=True,
        ):
            try:
                lev2_to_nc(
                    prod,
                    mwr_l1c_file,
                    file.name,
                    coeff_files=coeffs,
                    temp_file=temp_file.name if prod not in ("2P02", "2P03") else None,
                    hum_file=abs_hum_file.name
                    if prod not in ("2P02", "2P03")
                    else None,
                )
            except MissingInputData as err:
                raise MissingInputFileError from err

        with (
            netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
            netCDF4.Dataset(mwr_l1c_file, "r") as nc_l1c,
            netCDF4.Dataset(temp_file.name, "r") as nc_temp,
            netCDF4.Dataset(rel_hum_file.name, "r") as nc_rel_hum,
            netCDF4.Dataset(t_pot_file.name, "r") as nc_t_pot,
            netCDF4.Dataset(eq_temp_file.name, "r") as nc_eq_temp,
        ):
            nc_output.createDimension("time", len(nc_temp.variables["time"][:]))
            nc_output.createDimension("height", len(nc_temp.variables["height"][:]))

            for source, variables in (
                (nc_l1c, ("latitude", "longitude", "altitude")),
                (nc_temp, ("time", "height", "temperature")),
                (nc_rel_hum, ("relative_humidity",)),
                (nc_t_pot, ("potential_temperature",)),
                (nc_eq_temp, ("equivalent_potential_temperature",)),
            ):
                output.copy_variables(source, nc_output, variables)

            output.add_standard_global_attributes(nc_output, file_uuid)
            output.copy_global(nc_l1c, nc_output, ("year", "month", "day", "location"))
            nc_output.title = f"MWR multiple-pointing from {nc_l1c.location}"
            nc_output.cloudnet_file_type = "mwr-multi"
            nc_output.mwrpy_version = mwrpy_version
            output.fix_time_attributes(nc_output)
            nc_output.history = (
                f"{utils.get_time()} - MWR multiple-pointing product created \n"
                f"{nc_l1c.history}"
            )
            nc_output.source_file_uuids = nc_l1c.file_uuid
            nc_output.source = nc_l1c.source

    return file_uuid
