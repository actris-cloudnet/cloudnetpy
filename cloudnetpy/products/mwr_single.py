import tempfile
from tempfile import NamedTemporaryFile

import netCDF4
from mwrpy.level2.write_lev2_nc import lev2_to_nc
from mwrpy.version import __version__ as mwrpy_version

from cloudnetpy import output, utils
from cloudnetpy.products import product_tools


def generate_mwr_single(
    mwr_l1c_file: str,
    output_file: str,
    uuid: str | None = None,
) -> str:
    file_uuid = uuid if uuid is not None else utils.get_uuid()

    with (
        NamedTemporaryFile() as lwp_file,
        NamedTemporaryFile() as iwv_file,
        NamedTemporaryFile() as t_prof_file,
        NamedTemporaryFile() as abs_hum_file,
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        coeffs = product_tools.get_read_mwrpy_coeffs(mwr_l1c_file, temp_dir)

        for prod, file in zip(
            ("2I01", "2I02", "2P01", "2P03"),
            (lwp_file, iwv_file, t_prof_file, abs_hum_file),
            strict=True,
        ):
            lev2_to_nc(prod, mwr_l1c_file, file.name, coeff_files=coeffs)

        with (
            netCDF4.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_output,
            netCDF4.Dataset(lwp_file.name, "r") as nc_lwp,
            netCDF4.Dataset(iwv_file.name, "r") as nc_iwv,
            netCDF4.Dataset(abs_hum_file.name, "r") as nc_hum,
            netCDF4.Dataset(t_prof_file.name, "r") as nc_t_prof,
            netCDF4.Dataset(mwr_l1c_file, "r") as nc_l1c,
        ):
            nc_output.createDimension("height", len(nc_t_prof.variables["height"][:]))
            nc_output.createDimension("time", len(nc_lwp.variables["time"][:]))

            for source, variables in (
                (nc_iwv, ("iwv",)),
                (nc_hum, ("absolute_humidity",)),
                (nc_t_prof, ("temperature", "height")),
                (nc_l1c, ("latitude", "longitude", "altitude")),
                (
                    nc_lwp,
                    (
                        "time",
                        "lwp",
                        "lwp_random_error",
                        "lwp_offset",
                        "lwp_systematic_error",
                        "azimuth_angle",
                    ),
                ),
            ):
                output.copy_variables(source, nc_output, variables)

            output.add_standard_global_attributes(nc_output, file_uuid)
            output.copy_global(nc_l1c, nc_output, ("year", "month", "day", "location"))
            nc_output.title = f"MWR single-pointing from {nc_l1c.location}"
            nc_output.cloudnet_file_type = "mwr-single"
            nc_output.mwrpy_version = mwrpy_version
            output.fix_time_attributes(nc_output)
            output.replace_attribute_with_standard_value(
                nc_output,
                ("lwp", "iwv", "temperature", "azimuth_angle"),
                ("units", "long_name", "standard_name"),
            )
            nc_output.history = (
                f"{utils.get_time()} - MWR single-pointing product created \n"
                f"{nc_l1c.history}"
            )
            nc_output.source_file_uuids = nc_l1c.file_uuid
            nc_output.source = nc_l1c.source

    return file_uuid
