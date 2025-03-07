from os import PathLike
from pathlib import Path

import doppy
import doppy.netcdf
import netCDF4
import numpy as np
import numpy.typing as npt
import scipy.constants
from doppy.product.turbulence import HorizontalWind, Options, Turbulence, VerticalWind
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import cloudnetpy
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.output import copy_variables
from cloudnetpy.utils import get_time, get_uuid


def generate_epsilon_from_lidar(
    doppler_lidar_file: str | PathLike,
    doppler_lidar_wind_file: str | PathLike,
    output_file: str | PathLike,
    uuid: str | None,
):
    sliding_window_in_seconds = 3 * 60
    uuid = uuid if uuid is not None else get_uuid()
    opts = _get_options(doppler_lidar_file)
    opts.period = sliding_window_in_seconds
    vert = _vertical_wind_from_doppler_lidar_file(doppler_lidar_file)
    hori = _horizontal_wind_from_doppler_lidar_file(doppler_lidar_wind_file)
    turb = Turbulence.from_winds(vert, hori, opts)

    with (
        netCDF4.Dataset(Path(doppler_lidar_file), "r") as nc_src,
        doppy.netcdf.Dataset(Path(output_file), format="NETCDF4_CLASSIC") as nc,
    ):
        nc.add_dimension("time")
        nc.add_dimension("height", size=len(turb.height))
        nc.add_time(
            name="time",
            dimensions=("time",),
            standard_name="time",
            long_name="Time UTC",
            data=turb.time,
            dtype="f8",
        )
        nc.add_variable(
            name="height",
            dimensions=("height",),
            units="m",
            data=turb.height,
            standard_name=nc_src["height"].standard_name,
            long_name=nc_src["height"].long_name,
            dtype="f4",
        )
        nc.add_variable(
            name="epsilon",
            dimensions=("time", "height"),
            units="m2 s-3",
            data=turb.turbulent_kinetic_energy_dissipation_rate,
            mask=turb.mask,
            dtype="f4",
            long_name="Dissipation rate of turbulent kinetic energy",
        )
        nc.add_scalar_variable(
            name="ray_accumulation_time",
            units="s",
            long_name="Ray accumulation time",
            data=opts.ray_accumulation_time,
            dtype="f4",
        )
        nc.add_scalar_variable(
            name="rolling_window_period",
            units="s",
            long_name="Rolling window period",
            data=opts.period,
            dtype="f4",
        )

        nc.add_attribute("file_uuid", uuid)
        nc.add_attribute("cloudnet_file_type", "epsilon-lidar")
        nc.add_attribute("doppy_version", doppy.__version__)
        nc.add_attribute("cloudnetpy_version", cloudnetpy.__version__)
        nc.add_attribute(
            "title",
            "Dissipation rate of turbulent kinetic energy (lidar) "
            f"from {nc_src.location}",
        )

    copy_attributes_from_src(doppler_lidar_file, output_file)

    with (
        netCDF4.Dataset(output_file, "r+") as nc_out,
        netCDF4.Dataset(doppler_lidar_file, "r") as nc_src_stare,
        netCDF4.Dataset(doppler_lidar_wind_file, "r") as nc_src_wind,
    ):
        copy_variables(
            nc_src_stare, nc_out, ("latitude", "longitude", "altitude", "source")
        )
        nc_out.source_file_uuids = f"{nc_src_stare.file_uuid}, {nc_src_wind.file_uuid}"
        sources = {nc_src_stare.source, nc_src_wind.source}
        nc_out.source = ", ".join(sources)
        history = (
            f"{get_time()} - epsilon-lidar file created using doppy "
            f"v{doppy.__version__} and cloudnetpy v{cloudnetpy.__version__}\n"
            f"{nc_src_stare.history}\n"
            f"{nc_src_wind.history}"
        )
        history = "\n".join(
            line.strip() for line in history.splitlines() if line.strip()
        )
        nc_out.history = history
        nc_out.references = "https://doi.org/10.1175/2010JTECHA1455.1"
    return uuid


def copy_attributes_from_src(src: str | PathLike, trg: str | PathLike) -> None:
    with netCDF4.Dataset(src, "r") as nc_src, netCDF4.Dataset(trg, "a") as nc_trg:
        for attr in ("year", "month", "day", "location", "Conventions"):
            nc_trg.setncattr(attr, nc_src.getncattr(attr))


def _horizontal_wind_from_doppler_lidar_file(
    doppler_lidar_wind_file: str | PathLike,
) -> HorizontalWind:
    with netCDF4.Dataset(doppler_lidar_wind_file, "r") as nc:
        time = _datetime64_from_nc_var(nc["time"])
        height = np.array(nc["height"][:].data, dtype=np.float64)
        uwind = np.array(nc["uwind"][:].data, dtype=np.float64)
        vwind = np.array(nc["vwind"][:].data, dtype=np.float64)
        umask = np.array(nc["uwind"][:].mask, dtype=np.bool_)
        vmask = np.array(nc["vwind"][:].mask, dtype=np.bool_)
        V = np.sqrt(uwind**2 + vwind**2)
        mask = umask | vmask
        if np.all(mask):
            raise ValidTimeStampError
        t = np.broadcast_to(time[:, None], mask.shape)[~mask]
        h = np.broadcast_to(height[None, :], mask.shape)[~mask]
        interp_linear = LinearNDInterpolator(list(zip(t, h, strict=False)), V[~mask])
        interp_nearest = NearestNDInterpolator(list(zip(t, h, strict=False)), V[~mask])
        T, H = np.meshgrid(time, height, indexing="ij")
        V_linear = interp_linear(T, H)
        V_nearest = interp_nearest(T, H)
        isnan = np.isnan(V_linear)
        V_interp = V_linear
        V_interp[isnan] = V_nearest[isnan]
        if np.isnan(V_interp).any():
            msg = "Unexpected nans"
            raise ValueError(msg)
        return HorizontalWind(time=time, height=height, V=V_interp)


def _get_options(doppler_lidar_file: str | PathLike) -> Options:
    with netCDF4.Dataset(doppler_lidar_file, "r") as nc:
        if "ray_accumulation_time" in nc.variables:
            return Options(ray_accumulation_time=nc["ray_accumulation_time"][:])
        if "pulses_per_ray" in nc.variables:
            prf = _infer_pulse_repetition_frequency(
                np.array(nc["range"][:].data, dtype=np.float64)
            )
            return Options(ray_accumulation_time=float(nc["pulses_per_ray"][:] / prf))
        msg = "Missing ray info"
        raise ValueError(msg)


def _infer_pulse_repetition_frequency(range_: npt.NDArray[np.float64]):
    c = scipy.constants.c
    dist = range_.max() - range_.min()
    round_trip_time = 2 * dist / c

    T_LOW = 1 / 10_000  # Halo XR instruments operate on lower frequency
    T_HIGH = 1 / 15_000  # Rest should operate on higher frequency
    if round_trip_time / T_HIGH < 1:
        return 15e3
    if round_trip_time / T_LOW < 1:
        return 10e3
    msg = f"Suspiciously large range ({dist}m). " "Cannot infer pulse repetition rate"
    raise ValueError(msg)


def _vertical_wind_from_doppler_lidar_file(
    doppler_lidar_file: str | PathLike,
) -> VerticalWind:
    with netCDF4.Dataset(doppler_lidar_file, "r") as nc:
        time = _datetime64_from_nc_var(nc["time"])
        height = np.array(nc["height"][:].data, dtype=np.float64)
        w = np.array(nc["v"][:].data, dtype=np.float64)
        mask = np.array(nc["v"][:].mask, dtype=np.bool_)
    if isinstance(mask, np.ndarray) and mask.any():
        w[mask] = np.nan

    return VerticalWind(time=time, height=height, w=w, mask=mask)


def _datetime64_from_nc_var(var: netCDF4.Variable) -> npt.NDArray[np.datetime64]:
    return np.array(
        netCDF4.num2date(
            var[:].data,
            units=var.units,
            calendar=var.calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        ),
        dtype="datetime64[us]",
    )
