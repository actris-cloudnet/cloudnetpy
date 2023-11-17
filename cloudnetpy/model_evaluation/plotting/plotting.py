import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma

import cloudnetpy.model_evaluation.plotting.plot_tools as p_tools
import cloudnetpy.plotting.plotting as cloud_plt
from cloudnetpy.model_evaluation.model_metadata import MODELS
from cloudnetpy.model_evaluation.plotting.plot_meta import ATTRIBUTES
from cloudnetpy.model_evaluation.statistics.statistical_methods import DayStatistics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def generate_L3_day_plots(
    nc_file: str,
    product: str,
    model: str,
    *,
    title: bool = True,
    var_list: list | None = None,
    fig_type: str | None = "group",
    stats: tuple | None = ("error", "area", "hist", "vertical"),
    save_path: str | None = None,
    image_name: str | None = None,
    show: bool | None = False,
) -> None:
    """Generate visualizations for level 3 dayscale products.
    With figure type visualizations can be subplot in group, pair, single or
    statistic of given product. In group fig_type all different methods are plot
    in same figure but standard timegrid is separated from advection timegrid.
    In pair fig_type upper subplot is always model product and below one is
    product method. All product method in given file will be plotted in loop.
    Single fig_type will plot each product variable in a own figure.
    Statistic fig_type will plot select statistical method of all product method
    in same fig.

    Args:
    ----
        nc_file (str): Path to source file
        product (str): Name of product wanted to plot
        model (str): Name of model which downsampling was done with

        title (bool): Show title and additional labels. Operational processing
             sets this to False. Default is True.
        fig_type (str, optional): Type of figure wanted to produce. Options
            are 'group', 'pair', 'single' and 'statistic'. Default value is 'group'
        stats (list, optional): List of statistical methods to visualize in
            'statistic' fig_type generation. Default is all types, but methods can
            be called individually.
        var_list (list, optional): List of variables to be plotted. If None, all product
            variables in file will be plotted
        save_path (str, optional): If not None, visualization is saved
            to run path location

        image_name (str, optional): Saving name of generated fig
        show (bool, optional): If True, shows visualization
    Notes:
        In case of 'group' and 'statistic' fig_type advection timegrid is
        separated from standard timegrid to their own figures.
        In case of model cycles, cycles are visualized in their on figures same
        way as an individual model run would be visualized in its own in a group
        figure.

    Examples:
    --------
        >>> from cloudnetpy.model_evaluation.plotting.plotting
        import generate_L3_day_plots
        >>> l3_day_file = 'cf_ecmwf.nc'
        >>> product = 'cf'
        >>> model = 'ecmwf'
        >>> generate_L3_day_plots(l3_day_file, product, model)
        >>> l3_day_file = 'cf_ecmwf.nc'
        >>> product = 'cf'
        >>> model = 'ecmwf'
        >>> generate_L3_day_plots(l3_day_file, product, model,
        >>>                       fig_type='statistic', stats=['error'])
    """

    def _check_cycle_names():
        if not c_names:
            raise AttributeError

    cls = __import__("plotting")
    model_info = MODELS[model]
    model_name = model_info.model_name
    name_set = p_tools.parse_wanted_names(nc_file, product, model, var_list)
    for names in name_set:
        if len(names) > 0:
            try:
                cycle_names, cycles = p_tools.sort_cycles(names, model)
                for i, c_names in enumerate(cycle_names):
                    _check_cycle_names()
                    params = [
                        product,
                        c_names,
                        nc_file,
                        model,
                        model_name,
                        save_path,
                        image_name,
                    ]
                    kwargs = {"show": show, "title": title, "cycle": cycles[i]}
                    if fig_type == "statistic":
                        params = [
                            product,
                            c_names,
                            nc_file,
                            model,
                            model_name,
                            stats,
                            save_path,
                            image_name,
                        ]
                    getattr(cls, f"get_{fig_type}_plots")(*params, **kwargs)
            except AttributeError:
                params = [
                    product,
                    names,
                    nc_file,
                    model,
                    model_name,
                    save_path,
                    image_name,
                ]
                kwargs = {"show": show, "title": title}
                if fig_type == "statistic":
                    params = [
                        product,
                        names,
                        nc_file,
                        model,
                        model_name,
                        stats,
                        save_path,
                        image_name,
                    ]
                getattr(cls, f"get_{fig_type}_plots")(*params, **kwargs)


def get_group_plots(
    product: str,
    names: list,
    nc_file: str,
    model: str,
    model_name: str,
    save_path: str,
    image_name: str,
    *,
    show: bool,
    cycle: str = "",
    title: bool = True,
    include_xlimits: bool = False,
) -> None:
    """Group subplot visualization for both standard and advection downsampling.
    Generates group subplot figure for product with model and all different
    downsampling methods. Generates separated figures for standard and advection
    timegrids. All model cycles if any will be generated to their own figures.

    Args:
    ----
        product (str): Name of the product
        names (list): List of variables to be visualized to same fig
        nc_file (str): Path to a source file
        model (str): Name of used model in a downsampling process
        model_name (str): Correct name of a model
        save_path (str): Path for saving figures
        image_name (str, optional): Saving name of generated fig
        show (bool): Show figure before saving if True
        cycle (str): Name of cycle if exists
        title (bool): True or False if wanted to add title to subfig
    """
    fig, ax = initialize_figure(len(names))
    model_run = model
    name = ""
    j = 0
    for j, name in enumerate(names):
        variable_info = ATTRIBUTES[product]
        cloud_plt.set_yax(ax[j], 12, ylabel=None)
        cloud_plt.set_xax(ax[j], include_xlimits=include_xlimits)
        if title:
            _set_title(ax[j], name, product, variable_info)
        if j == 0 and title:
            _set_title(ax[j], model, product, variable_info, model_name)
        data, x, y = p_tools.read_data_characters(nc_file, name, model)
        plot_colormesh(ax[j], data, (x, y), variable_info)
    casedate = cloud_plt.set_labels(fig, ax[j], nc_file)
    if "adv" in name:
        product = product + "_adv"
    if len(cycle) > 1:
        fig.text(0.64, 0.885, f"Cycle: {cycle}", fontsize=13)
        model_run = f"{model}_{cycle}"
    cloud_plt.handle_saving(
        image_name,
        save_path,
        casedate,
        [product, model_run, "group"],
        show=show,
    )


def get_pair_plots(
    product: str,
    names: list,
    nc_file: str,
    model: str,
    model_name: str,
    save_path: str,
    image_name: str,
    cycle: str = "",
    *,
    show: bool = False,
    title: bool = True,
    include_xlimits: bool = False,
) -> None:
    """Pair subplots of model and product method.
    In upper subplot is model product and lower subplot one of the
    downsampled method of select product. Function generates all product methods
    in a given nc-file in loop.

    Args:
    ----
        product (str): Name of the product
        names (list): List of variables to be visualized to same fig
        nc_file (str): Path to a source file
        model (str): Name of used model in a downsampling process
        model_name (str): Correct name of a model
        save_path (str): Path for saving figures
        image_name (str, optional): Saving name of generated fig
        show (bool): Show figure before saving if True
        cycle (str): Name of cycle if exists
        title (bool): True or False if wanted add title to subfig
    """
    variable_info = ATTRIBUTES[product]
    model_ax = names[0]
    for i, name in enumerate(names):
        if i == 0:
            continue
        fig, ax = initialize_figure(2)
        cloud_plt.set_yax(ax[0], 12, ylabel=None)
        cloud_plt.set_yax(ax[-1], 12, ylabel=None)
        cloud_plt.set_xax(ax[0], include_xlimits=include_xlimits)
        cloud_plt.set_xax(ax[-1], include_xlimits=include_xlimits)

        if title:
            _set_title(ax[0], model, product, variable_info, model_name)
            _set_title(ax[-1], name, product, variable_info)
        model_data, mx, my = p_tools.read_data_characters(nc_file, model_ax, model)
        data, x, y = p_tools.read_data_characters(nc_file, name, model)
        plot_colormesh(ax[0], model_data, (mx, my), variable_info)
        plot_colormesh(ax[-1], data, (x, y), variable_info)
        casedate = cloud_plt.set_labels(fig, ax[-1], nc_file)
        if len(cycle) > 1:
            fig.text(0.64, 0.889, f"Cycle: {cycle}", fontsize=13)
        cloud_plt.handle_saving(
            image_name,
            save_path,
            casedate,
            [name, "pair"],
            show=show,
        )


def get_single_plots(
    product: str,
    names: list,
    nc_file: str,
    model: str,
    model_name: str,
    save_path: str,
    image_name: str,
    *,
    show: bool,
    cycle: str = "",
    title: bool = True,
    include_xlimits: bool = False,
) -> None:
    """Generates figures of each product variable from given file in loop.

    Args:
    ----
        product (str): Name of the product
        names (list): List of variables to be visualized to same fig
        nc_file (str): Path to a source file
        model (str): Name of used model in a downsampling process
        model_name (str): Correct name of a model
        save_path (str): Path for saving figures
        image_name (str, optional): Saving name of generated fig
        show (bool): Show figure before saving if True
        cycle (str): Name of cycle if exists
        title (bool): True or False if wanted to add title to subfig
    """
    variable_info = ATTRIBUTES[product]
    for _, name in enumerate(names):
        fig, ax = initialize_figure(1)
        cloud_plt.set_yax(ax[0], 12, ylabel=None)
        cloud_plt.set_xax(ax[0], include_xlimits=include_xlimits)

        if title:
            _set_title(ax[0], name, product, variable_info)
        data, x, y = p_tools.read_data_characters(nc_file, name, model)
        plot_colormesh(ax[0], data, (x, y), variable_info)
        casedate = cloud_plt.set_labels(fig, ax[0], nc_file, sub_title=title)
        if title:
            if len(cycle) > 1:
                fig.text(0.64, 0.9, f"{model_name} cycle: {cycle}", fontsize=13)
            else:
                fig.text(0.64, 0.9, f"{model_name}", fontsize=13)
        cloud_plt.handle_saving(
            image_name,
            save_path,
            casedate,
            [name, "single"],
            show=show,
        )


def plot_colormesh(ax, data: np.ndarray, axes: tuple, variable_info) -> None:
    vmin, vmax = variable_info.plot_range
    if variable_info.plot_scale == "logarithmic":
        data, vmin, vmax = cloud_plt.lin2log(data, vmin, vmax)
    cmap = plt.get_cmap(variable_info.cbar, 22)
    data[data < vmin] = ma.masked
    pl = ax.pcolormesh(*axes, data, vmin=vmin, vmax=vmax, cmap=cmap)
    colorbar = init_colorbar(pl, ax)
    if variable_info.plot_scale == "logarithmic":
        tick_labels = cloud_plt.generate_log_cbar_ticklabel_list(vmin, vmax)
        colorbar.set_ticks(np.arange(vmin, vmax + 1).tolist())
        colorbar.ax.set_yticklabels(tick_labels)
    ax.set_facecolor("white")
    colorbar.set_label(variable_info.clabel, fontsize=13)


def get_statistic_plots(
    product: str,
    names: list,
    nc_file: str,
    model: str,
    model_name: str,
    stats: list,
    save_path: str,
    image_name: str,
    *,
    show: bool,
    cycle: str = "",
    title: bool = True,
) -> None:
    """Statistical subplots for day scale products.
    Statistical analysis can be done by day scale with relative error ('error'),
    total data area analysis ('area'), histogram ('hist') or vertical profiles
    ('vertical'). Each given stats are looped through and generated as one figure
    per statistical method for a select product. All different downsampled method
    are in a same fig. Standard and advection timegrids are separated to own figs
    as well as different cycle runs.

    Args:
    ----
        product (str): Name of the product
        names (list): List of variables to be visualized to same fig
        nc_file (str): Path to a source file
        model (str): Name of used model in a downsampling process
        model_name (str): Correct name of a model
        stats (list): List of statistical method to process analysis with.
                      Options are ['error', 'area', 'hist', 'vertical']
        save_path (str): Path for saving figures
        image_name (str, optional): Saving name of generated fig
        show (bool): Show figure before saving if True
        cycle (str): Name of cycle if exists
    """
    model_run = model
    name = ""
    j = 0

    def _check_data():
        if model_missing and obs_missing:
            _raise()

    def _check_data2():
        if "error" in stat and np.all(day_stat.model_stat.mask is True):
            _raise()

    def _raise():
        err_msg = f"No data in {model_name} or observation"
        raise ValueError(err_msg)

    for stat in stats:
        try:
            obs_missing = False
            model_missing = False
            variable_info = ATTRIBUTES[product]
            fig, ax = initialize_figure(len(names) - 1, stat)
            model_data, _, _ = p_tools.read_data_characters(nc_file, names[0], model)
            if np.all(model_data.mask is True):
                model_missing = True
            for j, name in enumerate(names):
                data, x, y = p_tools.read_data_characters(nc_file, name, model)
                if np.all(data.mask is True):
                    obs_missing = True
                _check_data()
                statistics = "aerror" if product == "cf" and stat == "error" else stat
                if j > 0:
                    name_new = ""
                    name_new = _get_stat_titles(name_new, product, variable_info)
                    day_stat = DayStatistics(
                        statistics,
                        [product, model_name, name_new],
                        model_data,
                        data,
                    )
                    _check_data2()
                    initialize_statistic_plots(
                        j,
                        len(names) - 1,
                        ax[j - 1],
                        statistics,
                        day_stat,
                        model_data,
                        data,
                        (x, y),
                        variable_info,
                        title=title,
                    )
        except ValueError:
            logging.exception("Exception occurred")
        if stat not in ("hist", "vertical"):
            casedate = cloud_plt.set_labels(fig, ax[j - 1], nc_file, sub_title=title)
        else:
            casedate = cloud_plt.read_date(nc_file)
            _name = cloud_plt.read_location(nc_file)
            if title:
                cloud_plt.add_subtitle(fig, casedate, _name.capitalize())
        if len(cycle) > 1:
            fig.text(0.64, 0.885, f"Cycle: {cycle}", fontsize=13)
            model_run = f"{model}_{cycle}"
        cloud_plt.handle_saving(
            image_name,
            save_path,
            casedate,
            [name, stat, model_run],
            show=show,
        )


def initialize_statistic_plots(
    j: int,
    max_len: int,
    ax,
    method: str,
    day_stat: DayStatistics,
    model: ma.MaskedArray,
    obs: ma.MaskedArray,
    args: tuple,
    variable_info,
    *,
    title: bool = True,
    include_xlimits: bool = False,
) -> None:
    if method in ("error", "aerror"):
        plot_relative_error(ax, day_stat.model_stat.T, args, method)
        if title:
            ax.set_title(day_stat.title, fontsize=14)
        cloud_plt.set_yax(ax, 12, ylabel=None)
        cloud_plt.set_xax(ax, include_xlimits=include_xlimits)

    if method == "area":
        plot_data_area(ax, day_stat, model, obs, args, title=title)
        ax.text(
            0.9,
            -0.17,
            f"Common area: {day_stat.model_stat} %",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )
        cloud_plt.set_yax(ax, 12, ylabel=None)
        cloud_plt.set_xax(ax, include_xlimits=include_xlimits)

    if method == "hist":
        plot_histogram(ax, day_stat, variable_info)
        if j == max_len - 1 and (max_len % 2) == 0:
            ax.legend(
                loc="lower left",
                ncol=2,
                fontsize=12,
                bbox_to_anchor=(-0.03, -0.13),
            )
        if j == max_len - 1:
            ax.legend(
                loc="lower left",
                ncol=4,
                fontsize=12,
                bbox_to_anchor=(-0.03, -0.24),
            )
    if method == "vertical":
        plot_vertical_profile(ax, day_stat, args, variable_info)
        p_tools.set_yaxis(ax, 12)
        if j == max_len - 1 and (max_len % 2) == 0:
            ax.legend(
                loc="lower left",
                ncol=2,
                fontsize=12,
                bbox_to_anchor=(-0.03, -0.13),
            )
        if j == max_len - 1:
            ax.legend(
                loc="lower left",
                ncol=4,
                fontsize=12,
                bbox_to_anchor=(-0.03, -0.2),
            )


def plot_relative_error(ax, error: ma.MaskedArray, axes: tuple, method: str) -> None:
    pl = ax.pcolormesh(*axes, error[:-1, :-1].T, cmap="RdBu_r", vmin=-50, vmax=50)
    colorbar = init_colorbar(pl, ax)
    colorbar.set_label("%", fontsize=13)
    error[np.isnan(error)] = ma.masked
    error = ma.round(error, decimals=4)
    median_error = ma.median(error.compressed())
    median_error = np.round(median_error, 3)
    if method == "aerror":
        ax.text(
            0.9,
            -0.17,
            f"Median absolute error: {median_error} %",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )
    else:
        ax.text(
            0.9,
            -0.17,
            f"Median relative error: {median_error} %",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )


def plot_data_area(
    ax,
    day_stat: DayStatistics,
    model: ma.MaskedArray,
    obs: ma.MaskedArray,
    axes: tuple,
    *,
    title: bool = True,
) -> None:
    data, cmap = p_tools.create_segment_values([model.mask, obs.mask])
    pl = ax.pcolormesh(*axes, data, cmap=cmap)
    if title:
        colorbar = init_colorbar(pl, ax)
        colorbar.set_ticks(np.arange(1, 1, 3).tolist())
        ax.set_title(f"{day_stat.title}", fontsize=14)
    ax.set_facecolor("black")
    legend_elements = [
        Patch(facecolor="khaki", edgecolor="k", label="Model"),
        Patch(facecolor=cmap(0.5), edgecolor="k", label="Common"),
        Patch(facecolor=cmap(1.0), edgecolor="k", label="Observation"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        ncol=3,
        fontsize=12,
        bbox_to_anchor=(-0.005, -0.25),
    )


def plot_histogram(ax, day_stat: DayStatistics, variable_info) -> None:
    weights = np.ones_like(day_stat.model_stat) / float(len(day_stat.model_stat))
    hist_bins = np.histogram(day_stat.observation_stat, density=True)[-1]
    ax.hist(
        day_stat.model_stat,
        weights=weights,
        bins=hist_bins,
        alpha=0.7,
        facecolor="khaki",
        edgecolor="k",
        label=f"Model: {day_stat.title[0]}",
    )

    weights = np.ones_like(day_stat.observation_stat) / float(
        len(day_stat.observation_stat),
    )
    ax.hist(
        day_stat.observation_stat,
        weights=weights,
        bins=hist_bins,
        alpha=0.7,
        facecolor="steelblue",
        edgecolor="k",
        label="Observation",
    )
    ax.set_xlabel(variable_info.x_title, fontsize=13)
    if variable_info.plot_scale == "logarithmic":
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Relative frequency %", fontsize=13)
    ax.yaxis.grid(which="major")
    ax.set_title(f"{day_stat.title[-1]}", fontsize=14)


def plot_vertical_profile(
    ax,
    day_stat: DayStatistics,
    axes: tuple,
    variable_info,
) -> None:
    mrm = p_tools.rolling_mean(day_stat.model_stat)
    orm = p_tools.rolling_mean(day_stat.observation_stat)
    axes = axes[-1][0] if len(axes[-1].shape) > 1 else axes[-1]
    ax.plot(day_stat.model_stat, axes, "o", markersize=5.5, color="k")
    ax.plot(day_stat.observation_stat, axes, "o", markersize=5.5, color="k")
    ax.plot(
        day_stat.model_stat,
        axes,
        "o",
        markersize=4.5,
        color="orange",
        label=f"{day_stat.title[0]}",
    )
    ax.plot(
        day_stat.observation_stat,
        axes,
        "o",
        markersize=4.5,
        color="green",
        label="Observation",
    )

    ax.plot(mrm, axes, "-", color="k", lw=2.5)
    ax.plot(orm, axes, "-", color="k", lw=2.5)
    ax.plot(
        mrm,
        axes,
        "-",
        color="orange",
        lw=2,
        label="Mean of {day_stat.title[0]}",
    )
    ax.plot(orm, axes, "-", color="green", lw=2, label="Mean of observation")

    ax.set_title(f"{day_stat.title[-1]}", fontsize=14)
    ax.set_xlabel(variable_info.x_title, fontsize=13)
    if variable_info.plot_scale == "logarithmic":
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.yaxis.grid(which="major")
    ax.xaxis.grid(which="major")


def initialize_figure(n_subplots: int, stat: str = "") -> tuple:
    """Set up fig and ax object, if subplot"""
    if n_subplots <= 0:
        n_subplots = 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 4 + (n_subplots - 1) * 4.8))
    fig.subplots_adjust(left=0.06, right=0.73, hspace=0.31)
    if stat == "area":
        fig.subplots_adjust(left=0.06, right=0.73, hspace=0.33)
    if stat in ("hist", "vertical"):
        fig, axes = plt.subplots(1, 1, figsize=(6, 10))
        fig.subplots_adjust(top=0.85, left=0.08, right=0.75)
        if n_subplots > 1:
            fig, axes = plt.subplots(
                int(n_subplots / 2),
                2,
                figsize=(
                    12 + (n_subplots - (n_subplots / 2 + 1)) * 1,
                    6 + (n_subplots - (n_subplots / 2 + 1)) * 6,
                ),
            )
            fig.subplots_adjust(
                top=0.82 + (n_subplots - (n_subplots / 2 + 1)) * 0.025,
                bottom=0.1,
                left=0.08,
                right=0.75,
                hspace=0.2,
            )
            axes = axes.flatten()
    if stat == "vertical" and n_subplots > 1:
        fig, axes = plt.subplots(
            int(n_subplots / 2),
            2,
            figsize=(
                12 + (n_subplots - (n_subplots / 2 + 1)) * 1.2,
                7 + (n_subplots - (n_subplots / 2 + 1)) * 7.3,
            ),
        )
        fig.subplots_adjust(
            top=0.842 + (n_subplots - (n_subplots / 2 + 1)) * 0.012,
            bottom=0.1,
            left=0.08,
            right=0.75,
            hspace=0.16,
        )
        axes = axes.flatten()
    if n_subplots == 1:
        axes = [axes]
    return fig, axes


def init_colorbar(plot, axis) -> Colorbar:
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="1%", pad=0.25)
    return plt.colorbar(plot, fraction=1.0, ax=axis, cax=cax)


def _set_title(
    ax,
    field_name: str,
    product: str,
    variable_info,
    model_name: str = "",
) -> None:
    """Generates subtitles for different product types"""
    parts = field_name.split("_")
    if parts[0] == product:
        title = _get_product_title(variable_info)
        if product == "cf":
            title = _get_cf_title(field_name, variable_info)
        if product == "iwc":
            title = _get_iwc_title(field_name, variable_info)
        if "adv" in field_name:
            adv = " Downsampled using advection time"
            ax.text(0.9, -0.13, adv, size=12, ha="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=14)
    else:
        name = variable_info.name
        if len(model_name) > 1:
            ax.set_title(f"{name} of {model_name}", fontsize=14)
        else:
            ax.set_title(f"Simulated {name}")


def _get_cf_title(field_name: str, variable_info) -> str:
    title = f"{variable_info.name}, Area"
    if "V" in field_name:
        title = f"{variable_info.name}, Volume"
    return title


def _get_iwc_title(field_name: str, variable_info) -> str:
    name = variable_info.name
    if "att" in field_name:
        title = f"{name} with good attenuation"
    elif "rain" in field_name:
        title = f"{name} with rain"
    else:
        title = f"{name}"
    return title


def _get_product_title(variable_info) -> str:
    return f"{variable_info.name}"


def _get_stat_titles(field_name: str, product: str, variable_info) -> str:
    title = _get_product_title_stat(variable_info)
    if product == "cf":
        title = _get_cf_title_stat(field_name, variable_info)
    if product == "iwc":
        title = _get_iwc_title_stat(field_name, variable_info)
    if "adv" in field_name:
        adv = " (Advection time)"
        return f"{title}{adv}"
    return title


def _get_cf_title_stat(field_name: str, variable_info) -> str:
    name = variable_info.name
    title = f"{name} area"
    if "V" in field_name:
        title = f"{name} volume"
    return title


def _get_iwc_title_stat(field_name: str, variable_info) -> str:
    name = variable_info.name
    if "att" in field_name:
        title = f"{name} with good attenuation"
    elif "rain" in field_name:
        title = f"{name} with rain"
    else:
        title = f"{name}"
    return title


def _get_product_title_stat(variable_info) -> str:
    name = variable_info.name
    return f"{name}"
