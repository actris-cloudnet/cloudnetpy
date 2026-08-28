"""Helpers shared by the ARM instrument readers."""

import netCDF4


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
