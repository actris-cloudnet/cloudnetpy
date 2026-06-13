import netCDF4
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from numpy import ma

from cloudnetpy.model_evaluation.model_metadata import MODEL_PREFIX


def read_model_name(nc_file: str, model: str) -> str:
    """Returns a human-readable model name for plot titles.

    The model description is stored as the `model_name` global attribute when
    the file is created (from the model file's `source`). Falls back to the
    model identifier if the attribute is missing.
    """
    with netCDF4.Dataset(nc_file) as nc:
        return getattr(nc, "model_name", model)


def parse_wanted_names(
    nc_file: str,
    name: str,
    variables: list | None = None,
    *,
    advance: bool = False,
) -> tuple[list, list]:
    """Returns standard and advection lists of product types to plot."""
    names = variables or parse_dataset_keys(nc_file, name, advance=advance)
    standard_n = [n for n in names if name in n and "adv" not in n]
    standard_n = sort_model2first_element(standard_n)
    advection_n = [n for n in names if name in n and "adv" in n]
    model_names = [n for n in names if n.startswith(MODEL_PREFIX)]
    for i, model_n in enumerate(model_names):
        advection_n.insert(0 + i, model_n)
    if len(advection_n) < len(standard_n):
        return standard_n, []
    if len(advection_n) > len(standard_n):
        return advection_n, []
    return standard_n, advection_n


def parse_dataset_keys(nc_file: str, product: str, *, advance: bool) -> list:
    names = [n for n in netCDF4.Dataset(nc_file).variables if product in n]
    if not advance:
        names = [n for n in names if "cirrus" not in n and "snow" not in n]
    return names


def sort_model2first_element(a: list) -> list:
    mm = [n for n in a if n.startswith(MODEL_PREFIX)]
    for i, m in enumerate(mm):
        a.remove(m)
        a.insert(0 + i, m)
    return a


def read_data_characters(nc_file: str, name: str) -> tuple:
    """Gets dimensions and data for plotting."""
    nc = netCDF4.Dataset(nc_file)
    data = nc.variables[name][:]
    data = mask_small_values(data, name)
    x = nc.variables["time"][:]
    x = reshape_1d2nd(x, data)
    try:
        y = nc.variables[f"{MODEL_PREFIX}height"][:]
    except KeyError as err:
        msg = f"Missing variable {MODEL_PREFIX}height"
        raise RuntimeError(msg) from err
    y = y / 1000
    try:
        mask = y.mask
        if mask.any():
            x, y, data = change2one_dim_axes(ma.array(x), y, data)
    except AttributeError:
        return data, x, y
    return data, x, y


def mask_small_values(data: ma.MaskedArray, name: str) -> ma.MaskedArray:
    data[data <= 0] = ma.masked
    if "lwc" in name:
        data[data < 1e-5] = ma.masked
    if "iwc" in name:
        data[data < 1e-7] = ma.masked
    return data


def reshape_1d2nd(one_d: npt.NDArray, two_d: npt.NDArray) -> npt.NDArray:
    new_arr = np.zeros(two_d.shape)
    for i in range(len(two_d[0])):
        new_arr[:, i] = one_d
    return new_arr


def create_segment_values(model: ma.MaskedArray, obs: ma.MaskedArray) -> npt.NDArray:
    new_array = np.zeros(model.shape, dtype=int)
    new_array[model.mask & obs.mask] = 0  # No data
    new_array[~model.mask & obs.mask] = 1  # Only model
    new_array[~obs.mask & model.mask] = 3  # Only observation
    new_array[(~model.mask == 1) & (~obs.mask == 1)] = 2  # Both
    return new_array


def set_yaxis(ax: Axes, max_y: float, min_y: float = 0.0) -> None:
    ax.set_ylim(min_y, max_y)
    ax.set_ylabel("Height (km)", fontsize=13)


def rolling_mean(data: ma.MaskedArray, n: int = 4) -> npt.NDArray:
    mmr = []
    for i in range(len(data)):
        if not data[i : i + n].mask.all():
            mmr.append(np.nanmean(data[i : i + n]))
        else:
            mmr.append(np.nan)
    return np.asarray(mmr)


def change2one_dim_axes(
    x: ma.MaskedArray,
    y: ma.MaskedArray,
    data: npt.NDArray,
) -> tuple:
    # If any mask in x or y, change 2d to 1d axes values
    # Common shape need to match 2d data.
    for ax in [x, y]:
        try:
            mask = ax.mask
            if mask.any():
                y_out = [y[i] for i in range(len(y[:])) if not y[i].mask.all()]
                return x[:, 0], y_out[0], data.T
        except AttributeError:
            continue
    return x, y, data
