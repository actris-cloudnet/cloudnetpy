from tempfile import NamedTemporaryFile

import netCDF4
from mwrpy.process_mwrpy import process_product
from mwrpy.version import __version__ as mwrpy_version

from cloudnetpy import output, utils


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
        for prod, file in zip(
            ("2P02", "2P03", "2P04", "2P07", "2P08"),
            (temp_file, abs_hum_file, rel_hum_file, t_pot_file, eq_temp_file),
        ):
            process_product(
                prod,
                coeffs,
                mwr_l1c_file,
                file.name,
                temp_file=temp_file.name if prod not in ("2P02", "2P03") else None,
                hum_file=abs_hum_file.name if prod not in ("2P02", "2P03") else None,
            )

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

    return file_uuid
