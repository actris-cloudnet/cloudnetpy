"""Helpers shared by the ARM instrument readers."""

import os
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import netCDF4

from cloudnetpy.exceptions import ValidTimeStampError


def read_geolocation(nc: netCDF4.Dataset, site_meta: dict) -> dict:
    """Returns site_meta with lat/lon/alt filled from the file if missing."""
    site_meta = {**site_meta}
    for key, names in (
        ("latitude", ("latitude", "lat")),
        ("longitude", ("longitude", "lon")),
        ("altitude", ("altitude", "alt")),
    ):
        if key in site_meta:
            continue
        for name in names:
            if name in nc.variables:
                site_meta[key] = float(nc[name][:])
                break
    return site_meta


def concatenate_files(
    raw_files: str | PathLike | Sequence[str | PathLike],
    temp_dir: str,
    variables: Sequence[str],
) -> str | PathLike:
    """Concatenates time-dimensioned `variables` of several ARM files into one.

    Returns the input file as is if only one file is given.
    """
    files: list[Path]
    if isinstance(raw_files, (str, PathLike)):
        if not os.path.isdir(raw_files):
            return raw_files
        files = [
            Path(raw_files) / f
            for f in os.listdir(raw_files)
            if f.lower().endswith((".cdf", ".nc"))
        ]
    else:
        files = [Path(f) for f in raw_files]
    files = sorted(files, key=lambda f: f.name)
    if len(files) == 1:
        return files[0]
    output_file = Path(temp_dir) / "concatenated.nc"
    with netCDF4.Dataset(output_file, "w") as nc_out:
        nc_out.createDimension("time", None)
        for ind, file in enumerate(files):
            with netCDF4.Dataset(file) as nc_in:
                if ind == 0:
                    nc_out.setncatts({k: nc_in.getncattr(k) for k in nc_in.ncattrs()})
                    time_units = nc_in["time"].units
                elif nc_in["time"].units != time_units:
                    msg = "Inconsistent time units in ARM files"
                    raise ValidTimeStampError(msg)
                n_time = len(nc_out.dimensions["time"])
                for key in nc_in.variables:
                    if key not in variables and nc_in[key].ndim != 0:
                        continue
                    if key not in nc_out.variables:
                        var = nc_out.createVariable(
                            key, nc_in[key].dtype, nc_in[key].dimensions
                        )
                        var.setncatts(
                            {k: nc_in[key].getncattr(k) for k in nc_in[key].ncattrs()}
                        )
                        if nc_in[key].ndim == 0:
                            var[:] = nc_in[key][:]
                    if nc_in[key].ndim != 0:
                        nc_out[key][n_time:] = nc_in[key][:]
    return output_file
