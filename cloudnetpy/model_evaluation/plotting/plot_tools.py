from typing import Tuple, Union

import netCDF4
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from numpy import ma

from cloudnetpy.model_evaluation.model_metadata import MODELS


def parse_wanted_names(
    nc_file: str, name: str, model: str, variables: Union[list, None] = None, advance: bool = False
) -> Tuple[list, list]:
    """Returns standard and advection lists of product types to plot"""
    if variables:
        names = variables
    else:
        names = parse_dataset_keys(nc_file, name, advance, model)
    standard_n = [n for n in names if name in n and "adv" not in n]
    standard_n = sort_model2first_element(standard_n, model)
    advection_n = [n for n in names if name in n and "adv" in n]
    model_names = [n for n in names if f"{model}_" in n and not f"_{model}_" in n]
    for i, model_n in enumerate(model_names):
        advection_n.insert(0 + i, model_n)
    if len(advection_n) < len(standard_n):
        return standard_n, []
    if len(advection_n) > len(standard_n):
        return advection_n, []
    return standard_n, advection_n


def parse_dataset_keys(nc_file: str, product: str, advance: bool, model: str = "") -> list:
    names = list(netCDF4.Dataset(nc_file).variables.keys())
    a_names = ["cirrus", "snow"]
    model_vars = []
    for n in names:
        if model not in n:
            model_vars.append(n)
        elif model in n and product not in n:
            model_vars.append(n)
        else:
            continue
    if not advance:
        for a in a_names:
            for n in names:
                if a in n:
                    model_vars.append(n)
    for m in model_vars:
        names.remove(m)
    return names


def sort_model2first_element(a: list, model: str) -> list:
    mm = [n for n in a if f"{model}_" in n and f"_{model}_" not in n]
    for i, m in enumerate(mm):
        a.remove(m)
        a.insert(0 + i, m)
    return a


def sort_cycles(names: list, model: str) -> Tuple[list, list]:
    model_info = MODELS[model]
    cycles = model_info.cycle
    if cycles is None:
        raise AttributeError
    cycles_split = [x.strip() for x in cycles.split(",")]
    cycles_names = [[name for name in names if cycle in name] for cycle in cycles_split]
    cycles_names.sort()
    cycles_names = [c for c in cycles_names if c]
    cycles_new = [c for c in cycles_split for name in cycles_names if c in name[0]]
    return cycles_names, cycles_new


def read_data_characters(nc_file: str, name: str, model: str) -> Tuple:
    """Gets dimensions and data for plotting"""
    nc = netCDF4.Dataset(nc_file)
    data = nc.variables[name][:]
    data = mask_small_values(data, name)
    x = nc.variables["time"][:]
    x = reshape_1d2nd(x, data)
    try:
        y = nc.variables[f"{model}_height"][:]
    except KeyError:
        model_info = MODELS[model]
        cycles = model_info.cycle
        assert cycles is not None
        cycles_split = [x.strip() for x in cycles.split(",")]
        cycle = [cycle for cycle in cycles_split if cycle in name]
        y = nc.variables[f"{model}_{cycle[0]}_height"][:]
    y = y / 1000
    try:
        mask = y.mask
        if mask.any():
            x, y, data = change2one_dim_axes(x, y, data)
    except AttributeError:
        return data, x, y
    return data, x, y


def mask_small_values(data: ma.MaskedArray, name: str):
    data[data <= 0] = ma.masked
    if "lwc" in name:
        data[data < 1e-5] = ma.masked
    if "iwc" in name:
        data[data < 1e-7] = ma.masked
    return data


def reshape_1d2nd(one_d: np.ndarray, two_d: np.ndarray) -> np.ndarray:
    new_arr = np.zeros(two_d.shape)
    for i in range(len(two_d[0])):
        new_arr[:, i] = one_d
    return new_arr


def create_segment_values(arrays: list) -> Tuple:
    # 0=no data, 1=model, 2=intersection, 3=observation
    new_array = np.zeros(arrays[0].shape, dtype=int)
    for i, array in enumerate(arrays):
        new_array[~array] = new_array[~array] + i + 1
    new_array[new_array == 2] = 4
    new_array[new_array == 3] = 2
    new_array[new_array == 4] = 3

    colors = cm.get_cmap("YlGnBu", 256)
    newcolors = colors(np.linspace(0, 1, 256))
    # No data, model, both, observation
    cmap = ListedColormap(["white", "khaki", newcolors[90], newcolors[140]])
    return new_array, cmap


def set_yaxis(ax, max_y: float, min_y: float = 0.0):
    ax.set_ylim(min_y, max_y)
    ax.set_ylabel("Height (km)", fontsize=13)


def rolling_mean(data: ma.MaskedArray, n: int = 4) -> np.ndarray:
    mmr = []
    for i in range(len(data)):
        if not data[i : i + n].mask.all():
            mmr.append(np.nanmean(data[i : i + n]))
        else:
            mmr.append(np.nan)
    return np.asarray(mmr)


def change2one_dim_axes(x: ma.MaskedArray, y: ma.MaskedArray, data: np.ndarray) -> Tuple:
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
