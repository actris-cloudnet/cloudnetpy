from tempfile import NamedTemporaryFile

import netCDF4
from mwrpy.process_mwrpy import process_product

from cloudnetpy import utils
from cloudnetpy.output import copy_global, copy_variables


def generate_mwr_single(
    mwr_l1c_file: str, output_file: str, uuid: str | None = None
) -> str:
    file_uuid = uuid if uuid is not None else utils.get_uuid()

    coeffs = "hyytiala"

    with (
        NamedTemporaryFile() as lwp_file,
        NamedTemporaryFile() as iwv_file,
        NamedTemporaryFile() as t_prof_file,
        NamedTemporaryFile() as abs_hum_file,
    ):
        process_product("2I01", coeffs, mwr_l1c_file, lwp_file.name)
        process_product("2I02", coeffs, mwr_l1c_file, iwv_file.name)
        process_product("2P01", coeffs, mwr_l1c_file, t_prof_file.name)
        process_product("2P03", coeffs, mwr_l1c_file, abs_hum_file.name)

        # Combine data
        with (
            netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
            netCDF4.Dataset(lwp_file.name, "r") as nc_lwp,
            netCDF4.Dataset(iwv_file.name, "r") as nc_iwv,
            netCDF4.Dataset(abs_hum_file.name, "r") as nc_hum,
            netCDF4.Dataset(t_prof_file.name, "r") as nc_t_prof,
            netCDF4.Dataset(mwr_l1c_file, "r") as nc_l1c,
        ):
            # Dimensions
            nc_output.createDimension("height", len(nc_t_prof.variables["height"][:]))
            nc_output.createDimension("time", len(nc_lwp.variables["time"][:]))

            # Global attributes
            copy_global(
                nc_l1c, nc_output, ("year", "month", "day", "Conventions", "location")
            )
            nc_output.title = f"MWR single-pointing from {nc_l1c.location}"
            nc_output.file_uuid = file_uuid
            nc_output.cloudnet_file_type = "mwr-single"

            # History
            new_history = (
                f"{utils.get_time()} - MWR single-pointing product created \n"
                f"{nc_l1c.history}"
            )
            nc_output.history = new_history

            # From Level 1c file
            copy_variables(nc_l1c, nc_output, ("latitude", "longitude", "altitude"))

            # From Level 2 files
            copy_variables(
                nc_lwp,
                nc_output,
                (
                    "time",
                    "lwp",
                    "lwp_random_error",
                    "lwp_offset",
                    "lwp_systematic_error",
                    "azimuth_angle",
                ),
            )
            copy_variables(nc_iwv, nc_output, ("iwv",))
            copy_variables(nc_t_prof, nc_output, ("temperature", "height"))
            copy_variables(nc_hum, nc_output, ("absolute_humidity",))

            # Fix some attributes
            nc_output.variables["time"].standard_name = "time"
            nc_output.variables["time"].long_name = "Time UTC"
            nc_output.variables["time"].calendar = "standard"
            nc_output.variables["time"].units = (
                f"hours since "
                f"{nc_output.year}-{nc_output.month}-{nc_output.day} "
                f"00:00:00 +00:00"
            )

            nc_output.variables["lwp"].long_name = "Liquid water path"
            nc_output.variables[
                "lwp"
            ].standard_name = "atmosphere_cloud_liquid_water_content"
            nc_output.variables[
                "iwv"
            ].standard_name = "atmosphere_mass_content_of_water_vapor"
            nc_output.variables["iwv"].long_name = "Integrated water vapour"
            nc_output.variables["azimuth_angle"].long_name = "Azimuth angle"
            nc_output.variables["temperature"].long_name = "Temperature"

    return file_uuid
