from tempfile import NamedTemporaryFile

import netCDF4
from mwrpy.process_mwrpy import process_product

from cloudnetpy import utils
from cloudnetpy.output import copy_global, copy_variables


def generate_mwr_multi(
    mwr_l1c_file: str, output_file: str, uuid: str | None = None
) -> str:
    file_uuid = uuid if uuid is not None else utils.get_uuid()

    coeffs = "hyytiala"

    with (
        NamedTemporaryFile() as temp_file,
        NamedTemporaryFile() as abs_hum_file,
        NamedTemporaryFile() as rel_hum_file,
        NamedTemporaryFile() as t_pot_file,
        NamedTemporaryFile() as eq_temp_file,
    ):
        process_product("2P02", coeffs, mwr_l1c_file, temp_file.name)
        process_product("2P03", coeffs, mwr_l1c_file, abs_hum_file.name)

        for prod, file in zip(
            ("2P04", "2P07", "2P08"), (rel_hum_file, t_pot_file, eq_temp_file)
        ):
            process_product(
                prod,
                coeffs,
                mwr_l1c_file,
                file.name,
                temp_file=temp_file.name,
                hum_file=abs_hum_file.name,
            )

        with (
            netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
            netCDF4.Dataset(mwr_l1c_file, "r") as nc_l1c,
            netCDF4.Dataset(temp_file.name, "r") as nc_temp,
            netCDF4.Dataset(rel_hum_file.name, "r") as nc_rel_hum,
            netCDF4.Dataset(t_pot_file.name, "r") as nc_t_pot,
            netCDF4.Dataset(eq_temp_file.name, "r") as nc_eq_temp,
        ):
            # Dimensions
            nc_output.createDimension("time", len(nc_temp.variables["time"][:]))
            nc_output.createDimension("height", len(nc_temp.variables["height"][:]))
            nc_output.createDimension("bnds", 2)

            # History
            new_history = (
                f"{utils.get_time()} - MWR multiple-pointing product created \n"
                f"{nc_l1c.history}"
            )
            nc_output.history = new_history

            # From Level 1c file
            copy_global(
                nc_l1c, nc_output, ("year", "month", "day", "Conventions", "location")
            )
            copy_variables(nc_l1c, nc_output, ("latitude", "longitude", "altitude"))

            # From Level 2 products
            copy_variables(
                nc_temp,
                nc_output,
                (
                    "time",
                    "height",
                    "temperature",
                ),
            )
            copy_variables(nc_rel_hum, nc_output, ("relative_humidity",))
            copy_variables(nc_t_pot, nc_output, ("potential_temperature",))
            copy_variables(nc_eq_temp, nc_output, ("equivalent_potential_temperature",))

            # Global attributes
            nc_output.file_uuid = file_uuid
            nc_output.cloudnet_file_type = "mwr-multi"
            nc_output.title = f"MWR multiple-pointing from {nc_l1c.location}"

            # Fix some attributes
            nc_output.variables["time"].standard_name = "time"
            nc_output.variables["time"].long_name = "Time UTC"
            nc_output.variables["time"].calendar = "standard"
            nc_output.variables["time"].units = (
                f"hours since "
                f"{nc_output.year}-{nc_output.month}-{nc_output.day} "
                f"00:00:00 +00:00"
            )

    return file_uuid
