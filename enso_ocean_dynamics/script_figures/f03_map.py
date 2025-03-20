# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Plot maps
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy
from os.path import dirname as os__path__dirname
from os.path import join as os__path__join
from typing import Literal

# local functions
from enso_ocean_dynamics.plot.templates import fig_basic_map
from enso_ocean_dynamics.tools.default_tools import none_to_default
from enso_ocean_dynamics.tools.dictionary_tools import put_in_dict
from enso_ocean_dynamics.wrapper.processors import reader
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Default arguments
# ---------------------------------------------------------------------------------------------------------------------#
default = {
    #
    # -- Data
    #
    # list of datasets: list[str]
    "data_datasets": ["ORAS5"],
    # epoch name: str
    "data_epoch": "1993_2019",
    # list of experiments: list[str]
    "data_experiments": ["historical"],
    # list of projects: list[str]
    "data_projects": ["observations"],
    # data type: str
    "data_type": "map",
    # list of variables: list[str]
    "data_variables": ["rssh_ptro"],
    #
    # -- Processing
    #
    # date to select
    "proc_time": {"year": 1998, "month": 7},
    #
    # -- Figure
    #
    # figure format: eps, pdf, png, svg
    "fig_format": "pdf",
    # size of each panel
    "fig_panel_size": {
        "frac": {"x": 0.5, "y": 0.5},
        "panel_1": {"x_delt": 2, "x_size": 20, "y_delt": 2, "y_size": 6},
    },
    # colors
    "fig_colors": {
        "AVISO": "k",
        "GFDL-ECDAv3.1": "k",
        "GODAS": "k",
        "ORAS5": "k",
    },
    # ticks
    "fig_ticks": {
        "panel_1": {
            "s_nam": "VARIABLE (UNITS)",
            "s_tic": list(range(-30, 31, 15)),
            "sha_cs": "cmo.balance",
            "x_lim": [100, 300],
            "x_nam": "",
            "x_tic": list(range(130, 301, 50)),
            "y_lim": [-30, 30],
            "y_nam": "",
            "y_tic": list(range(-30, 31, 15)),
            "central_longitude": 200,
        },
    },
    # titles
    "fig_titles": {},
    # panel parameters (to modify default values in enso_ocean_dynamics.plot.panels)
    "panel_param": {},
}
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Main
# ---------------------------------------------------------------------------------------------------------------------#
def f03_map_plot(
        data_datasets: list[str] | None = None,
        data_epoch: str | None = None,
        data_experiments: list[str] | None = None,
        data_projects: list[str] | None = None,
        data_type: str | None = None,
        data_variables: list[str] | None = None,
        proc_time: dict[str, int] | None = None,
        fig_colors: dict[str, str] | None = None,
        fig_format: Literal["eps", "pdf", "png", "svg"] | None = None,
        fig_panel_size: dict[str, dict[str, float | int]] | None = None,
        fig_ticks: dict[str, dict[str, dict[str, list[float | int] | None] | list[float | int] | None] |
                        list[float | int] | None] | None = None,
        fig_titles: dict[str, str] | None = None,
        panel_param: dict | None = None,
        **kwargs):
    #
    # -- Set to default
    #
    data_datasets = none_to_default(data_datasets, default["data_datasets"])
    data_epoch = none_to_default(data_epoch, default["data_epoch"])
    data_experiments = none_to_default(data_experiments, default["data_experiments"])
    data_projects = none_to_default(data_projects, default["data_projects"])
    data_type = none_to_default(data_type, default["data_type"])
    data_variables = none_to_default(data_variables, default["data_variables"])
    proc_time = none_to_default(proc_time, default["proc_time"])
    fig_colors = none_to_default(fig_colors, default["fig_colors"])
    fig_format = none_to_default(fig_format, default["fig_format"])
    fig_panel_size = none_to_default(fig_panel_size, default["fig_panel_size"])
    fig_ticks = none_to_default(fig_ticks, default["fig_ticks"])
    fig_titles = none_to_default(fig_titles, default["fig_titles"])
    panel_param = none_to_default(panel_param, default["panel_param"])
    #
    # -- Read data
    #
    values = {}
    tmp = reader(data_datasets=data_datasets, data_epoch=data_epoch, data_experiments=data_experiments,
                     data_projects=data_projects, data_type=data_type, data_variables=data_variables)
    for dia in list(tmp.keys()):
        for pro in list(tmp[dia].keys()):
            for exp in list(tmp[dia][pro].keys()):
                for dat in list(tmp[dia][pro][exp].keys()):
                    for mem in list(tmp[dia][pro][exp][dat].keys()):
                        put_in_dict(values, tmp[dia][pro][exp][dat][mem], dia, dat)
    #
    # -- Organize data for plot
    #
    figure_axes, plot_data = {}, {}
    # panel 1: map
    panel = "panel_1"
    for dia in list(values.keys()):
        for dat in list(values[dia].keys()):
            pan = copy__deepcopy(panel)
            grp = copy__deepcopy(dat)
            tuple_k = (dia, grp, pan)
            # arrays
            arr = values[dia][dat]
            # panel axes definition
            if panel in list(fig_ticks.keys()):
                for k1, k2 in fig_ticks[panel].items():
                    tmp = copy__deepcopy(k2)
                    if k1 in ["s_nam", "x_nam", "y_nam"]:
                        if "VARIABLE" in tmp and "short_name" in list(arr.attrs.keys()):
                            tmp = tmp.replace("VARIABLE", arr.attrs["short_name"])
                        if "UNITS" in tmp and "units" in list(arr.attrs.keys()):
                            tmp = tmp.replace("UNITS", arr.attrs["units"])
                            tmp = tmp.replace(" ()", "")
                    put_in_dict(figure_axes, tmp, *tuple_k + (k1,))
            dt = {}
            if dia in list(figure_axes.keys()) and grp in list(figure_axes[dia].keys()) and \
                    pan in list(figure_axes[dia][grp].keys()):
                dt = figure_axes[dia][grp][pan]
            # test possibility to plot text / legend
            do_legend = False
            if "x_lim" in list(dt.keys()) and "y_lim" in list(dt.keys()) and dt["x_lim"] is not None and \
                    dt["y_lim"] is not None:
                do_legend = True
            # -- shading
            plot_type = "sha_s"
            yy, mm = proc_time["year"], proc_time["month"]
            tmp = arr.sel(time=slice(str(yy) + "-" + str(mm) + "-01", str(yy) + "-" + str(mm) + "-28")).squeeze()
            put_in_dict(plot_data, tmp, *tuple_k + (str(plot_type),))
            put_in_dict(plot_data, ["pen1", "pes1"], *tuple_k + ("region",))
            # -- text
            plot_type = "text"
            if do_legend is True:
                x1, x2, y1, y2 = dt["x_lim"] + dt["y_lim"]
                lon = dt["central_longitude"] if "central_longitude" in list(dt.keys()) else 0
                x1 -= lon
                x2 -= lon
                dx, dy = (x2 - x1) / 100, (y2 - y1) / 100
                # dataset names
                lt, lc, lf, lh, lv, lx, ly = [], [], [], [], [], [], []
                for k1, k2, k3 in zip([dat], ["left"], ["bottom"]):
                    lt.append(k1)
                    lc.append(fig_colors[k1])
                    lf.append(15)
                    lh.append(k2)
                    lv.append(k3)
                    lx.append(x1 - 0 * dx)
                    ly.append(y2 + 3 * dy)
                put_in_dict(plot_data, copy__deepcopy(lt), *tuple_k + (plot_type,))
                put_in_dict(plot_data, copy__deepcopy(lc), *tuple_k + (str(plot_type) + "_c",))
                put_in_dict(plot_data, copy__deepcopy(lf), *tuple_k + (str(plot_type) + "_fs",))
                put_in_dict(plot_data, copy__deepcopy(lh), *tuple_k + (str(plot_type) + "_ha",))
                put_in_dict(plot_data, copy__deepcopy(lv), *tuple_k + (str(plot_type) + "_va",))
                put_in_dict(plot_data, copy__deepcopy(lx), *tuple_k + (str(plot_type) + "_x",))
                put_in_dict(plot_data, copy__deepcopy(ly), *tuple_k + (str(plot_type) + "_y",))
    #
    # -- Plot
    #
    # output plot directory (relative to current file directory)
    figure_directory = "/".join(os__path__dirname(__file__).split("/")[:-2]) + "/figures"
    # figure name
    figure_name = __file__.split("/")[-1].split(".")[0] + "_" + "_".join([data_epoch])
    figure_name = os__path__join(figure_directory, figure_name)
    for dia in list(plot_data.keys()):
        # list plot names
        list_names = list(plot_data[dia].keys())
        # output figure name
        fig_o = figure_name + "_" + "_".join([dia] + [str(proc_time[k]).zfill(2) for k in list(proc_time.keys())])
        # plot
        fig_basic_map(plot_data[dia], list_names, 2, figure_axes[dia], fig_format, fig_o, fig_panel_size,
                      panel_param=panel_param)
# ---------------------------------------------------------------------------------------------------------------------#
