# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Functions to plot panels
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy, deepcopy
from inspect import stack as inspect__stack
# cartopy
import cartopy.crs as ccrs
from cartopy.crs import PlateCarree
import cartopy.feature as cfeature
# cmocean
import cmocean
# matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon
import matplotlib.pyplot as pyplot
# numpy
from numpy import array as numpy__array
from numpy import linspace as numpy__linspace
from numpy import meshgrid as numpy__meshgrid
from numpy import zeros as numpy__zeros

# local package
from enso_ocean_dynamics.tools.default_tools import none_to_default
from enso_ocean_dynamics.tools.print_tools import print_fail
from enso_ocean_dynamics.wrapper.xarray_base import array_wrapper, convert_cf_dim_key
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Default parameters for plots
# ---------------------------------------------------------------------------------------------------------------------#
# general values
default_plot: dict[str, list[float] | bool | float | int | str] = {
    # horizontal and vertical size of the plot
    "size_x": 4.,
    "size_y": 4.,
    # axes ticks
    "lab": [-1., 1.],
    "lab_side": "both",
    "tic": [-1., 1.],
    "nbr_minor": 1,
    # reference color / line / marker
    "color": "k",
    "linestyle": "-",
    "linewidth": 1.,
    "marker": "D",
    "size": 100.,
    # text
    "fontsize": 15.,
    "horizontalalignment": "left",
    "rotation": 0.,
    "text": "",
    "verticalalignment": "center",
    # plot_ax_title
    "title_column_x": 50.,
    "title_column_y": 10.,
    "title_row_x": 35.,
    "title_row_y": 50.,
    "title_x": 4.05,
    "title_y": 4.1,
    # plot_boxplot
    "boxplot_flier": True,
    "boxplot_flier_size": 5.,
    "boxplot_linestyle": "-",
    "boxplot_linewidth": 2,
    "boxplot_mean": True,
    "boxplot_mean_size": 9.,
    "boxplot_vert": True,
    "boxplot_zorder": 1,
    # plot_curve
    "curve_zorder": 2,
    # plot_marker
    "marker_position": "bottom",
    "marker_zorder": 3,
    # plot_shading
    "shading_alpha": 1.,
    "shading_color": "r",
    "shading_zorder": 1,
}
# legend
default_legend = {
    "line": {
        "color": default_plot["color"],
        "linestyle": None,
        "linewidth": 2},
    "marker": {
        "edgecolor": default_plot["color"],
        "facecolor": default_plot["color"],
        "linestyle": default_plot["linestyle"],
        "linewidth": default_plot["linewidth"],
        "marker": None,
        "s": default_plot["size"]},
    "position": {
        "x": 5,
        "y": 95},
    "text": {
        "color": default_plot["color"],
        "fontsize": default_plot["fontsize"],
        "horizontalalignment": default_plot["horizontalalignment"],
        "verticalalignment": default_plot["verticalalignment"],
        "rotation": default_plot["rotation"]},
}
# map
default_map: dict[str, float | list[float] | PlateCarree | str] = {
    "colorbar": "cmo.balance",
    "color": "gainsboro",
    "lat_lim": [-20., 20.],
    "lat_tic": [-15, 0, 15],
    "lon_lim": [130., 290.],
    "lon_tic": [140, 180, 220, 260],
    "projection": ccrs.PlateCarree(central_longitude=0),
    "size_x": 7.2,
    "size_y": 3.6,
}
default_region = {
    "n30e": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [-5., 5.], "longitude": [210., 270.],
             "short_name": "N3"},
    "n34e": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [-5., 5.], "longitude": [190., 240.],
             "short_name": "N3.4"},
    "n40e": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [-5., 5.], "longitude": [160., 210.],
             "short_name": "N4"},
    "pen1": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [2., 6.], "longitude": [120., 280.],
             "short_name": "PEN1"},
    "pen3": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [6., 10.], "longitude": [120., 280.],
             "short_name": "PEN3"},
    "pes1": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [-6., -2.], "longitude": [120., 280.],
             "short_name": "PES1"},
    "pes3": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [-10., -6.], "longitude": [120., 280.],
             "short_name": "PES3"},
    "pequ": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [-5, 5], "longitude": [120., 280.],
             "short_name": "PNEQ"},
    "pneq": {"color": "k", "linestyle": "-", "linewidth": 3, "latitude": [-2.5, 2.5], "longitude": [120., 280.],
             "short_name": "PNEQ"},
}
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
def _ax_axis_x(
        ax,
        fontsize: float = None,
        x_lab: list = None,
        x_lab_rot: float = None,
        x_lim: list = None,
        x_nam: str = None,
        x_nbr_minor: int = None,
        x_tic: list = None,
        **kwarg):
    """
    Define x-axis

    Input:
    ------
    :param ax: matplotlib Axes object
    :param fontsize: float, optional
        Size of the font; e.g., fontsize = 15
    :param x_lab: list, optional
        List of labels to write on x-axis
    :param x_lab_rot: float, optional
        Label rotation on x-axis
    :param x_lim: list, optional
        Bounds of x-axis
    :param x_nam: str, optional
        Name of the x-axis
    :param x_nbr_minor: int, optional
        Number of minor x tick(s) between major ticks
    :param x_tic: list, optional
        Tick positions on x-axis
    """
    # set default values if not given
    fontsize = none_to_default(fontsize, default_plot["fontsize"])
    x_lab = none_to_default(x_lab, default_plot["lab"])
    x_lab_rot = none_to_default(x_lab_rot, default_plot["rotation"])
    x_nam = none_to_default(x_nam, default_plot["text"])
    x_nbr_minor = none_to_default(x_nbr_minor, default_plot["nbr_minor"])
    x_tic = none_to_default(x_tic, default_plot["tic"])
    x_lim = none_to_default(x_lim, [min(x_tic), max(x_tic)])
    # major x ticks
    ax.set_xticks(x_tic, minor=False)
    # minor x ticks
    if isinstance(x_nbr_minor, int) is True and x_nbr_minor > 0:
        # put lx_nbr_minor minor x tick between major ticks
        dx = (x_tic[1] - x_tic[0]) / (x_nbr_minor + 1)
        minor = list()
        xx = x_tic[0]
        while xx > min(x_lim):
            xx -= dx
        while xx < max(x_lim):
            if xx not in x_tic:
                minor.append(xx)
            xx += dx
        ax.set_xticks(minor, minor=True)
    # tick labels
    if 0 < x_lab_rot < 90:
        ax.set_xticklabels(x_lab, ha="right")
        pad = -3
    else:
        ax.set_xticklabels(x_lab)
        pad = 2
    # bounds of x-axis
    ax.set_xlim(x_lim)
    ax.set_xlabel(x_nam, fontsize=fontsize)
    # params of x-axis
    ax.tick_params(axis="x", direction="out", which="both", labelsize=fontsize, bottom=True, top=True, left=False,
                   right=False, labelrotation=x_lab_rot, pad=pad)
    ax.xaxis.set_tick_params(length=6, width=1.5, which="major")
    ax.xaxis.set_tick_params(length=3, width=1.0, which="minor")


def _ax_axis_y(
        ax,
        fontsize: float = None,
        y_axe: str = None,
        y_lab: list = None,
        y_lim: list = None,
        y_nam: str = None,
        y_nbr_minor: int = None,
        y_tic: list = None,
        **kwarg):
    """
    Define y-axis

    Input:
    ------
    :param ax: matplotlib Axes object
    :param fontsize: float, optional
        Size of the font; e.g., fontsize = 15
    :param y_axe: str, optional
        Side where to plot the y-axis ('both', 'left', 'right')
    :param y_lab: list, optional
        List of labels to write on y-axis
    :param y_lim: list, optional
        Bounds of y-axis
    :param y_nam: str, optional
        Name of the y-axis
    :param y_nbr_minor: int, optional
        Number of minor y tick(s) between major ticks
    :param y_tic: list, optional
        Tick positions on y-axis
    """
    # set default values if not given
    fontsize = none_to_default(fontsize, default_plot["fontsize"])
    y_axe = none_to_default(y_axe, default_plot["lab_side"])
    y_lab = none_to_default(y_lab, default_plot["lab"])
    y_nam = none_to_default(y_nam, default_plot["text"])
    y_nbr_minor = none_to_default(y_nbr_minor, default_plot["nbr_minor"])
    y_tic = none_to_default(y_tic, default_plot["tic"])
    y_lim = none_to_default(y_lim, [min(y_tic), max(y_tic)])
    # major y ticks
    ax.set_yticks(y_tic, minor=False)
    # minor y ticks
    if isinstance(y_nbr_minor, int) is True and y_nbr_minor > 0:
        # put y_nbr_minor minor y tick between major ticks
        dy = (y_tic[1] - y_tic[0]) / (y_nbr_minor + 1)
        minor = list()
        yy = y_tic[0]
        while yy > min(y_lim):
            yy -= dy
        while yy < max(y_lim):
            if yy not in y_tic:
                minor.append(yy)
            yy += dy
        ax.set_yticks(minor, minor=True)
    # tick labels
    ax.set_yticklabels(y_lab)
    # bounds of y-axis
    ax.set_ylim(y_lim)
    ax.set_ylabel(y_nam, fontsize=fontsize)
    # params of y-axis
    if y_axe == "both":
        ytic_left, ytic_right = True, True
        ylab_left, ylab_right = True, False
    elif y_axe == "left":
        ytic_left, ytic_right = True, False
        ylab_left, ylab_right = True, False
    else:
        ytic_left, ytic_right = False, True
        ylab_left, ylab_right = False, True
    ax.tick_params(axis="y", direction="out", which="both", labelsize=fontsize, bottom=False, top=False, left=ytic_left,
                   right=ytic_right, labelbottom=False, labeltop=False, labelleft=ylab_left, labelright=ylab_right)
    ax.yaxis.set_tick_params(length=6, width=1.5, which="major")
    ax.yaxis.set_tick_params(length=3, width=1.0, which="minor")


def _ax_axis_y2(
        ax2,
        fontsize: float = None,
        y2_axe: str = None,
        y2_lab: list = None,
        y2_lim: list = None,
        y2_nam: str = None,
        y2_nbr_minor: int = None,
        y2_tic: list = None,
        **kwarg):
    tmp_kwarg = deepcopy(kwarg)
    tmp_kwarg.update({"fontsize": fontsize, "y_axe": y2_axe, "y_lab": y2_lab, "y_lim": y2_lim, "y_nam": y2_nam,
                      "y_nbr_minor": y2_nbr_minor, "y_tic": y2_tic})
    _ax_axis_y(ax2, **tmp_kwarg)


def _ax_legend(
        ax,
        legend_param: dict = None,
        legend_txt: list = None,
        x_size: float = None,
        **kwarg):
    """
    Plot the legend if a keyword is used

    Input:
    ------
    :param ax: matplotlib Axes object
    :param legend_param, dict, optional
        Dictionary with two nested levels [legend_name, legend_keys1, legend_keys2]
        The first level (legend_name) must correspond to the values in legend_txt
        The second level (legend_keys1) must be made of keywords supported in this code:
            line: definition of the line (color, linestyle, linewidth)
            marker: definition of the line (edgecolor, facecolor, linestyle, linewidth, s)
            position: horizontal / vertical distance (percentage) from the left / bottom axis (x, y)
            text: definition of the line (color, fontsize, horizontalalignment, verticalalignment, rotation)
    :param legend_txt: list, optional
        List of string to write on the figure
    :param x_size: float, optional
        Horizontal size of the plot
    """
    # set default values if not given
    x_size = none_to_default(x_size, default_plot["size_x"])
    # plot dimensions
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    dx = (x2 - x1) / 100.
    dy = (y2 - y1) / 100.
    # legend
    if isinstance(legend_txt, list) is True:
        for txt in legend_txt:
            # get legend parameters
            if isinstance(legend_param, dict) is False or (
                    isinstance(legend_param, dict) is True and txt not in list(legend_param.keys())):
                dict_t = copy__deepcopy(default_legend)
            else:
                dict_t = dict((k, {}) for k in list(default_legend.keys()))
                for k1 in list(default_legend.keys()):
                    if k1 not in list(legend_param[txt].keys()):
                        dict_t[k1] = default_legend[k1]
                    else:
                        for k2 in list(default_legend[k1].keys()):
                            if k2 not in list(legend_param[txt][k1].keys()):
                                dict_t[k1][k2] = default_legend[k1][k2]
                            else:
                                dict_t[k1][k2] = legend_param[txt][k1][k2]
            # legend position
            xx1 = x1 + dict_t["position"]["x"] * dx
            xx2 = xx1 + 4 * dx * default_plot["size_x"] / x_size
            xx3 = xx2 + 4 * dx * default_plot["size_x"] / x_size
            xx4 = xx3 + 1 * dx * default_plot["size_x"] / x_size
            xx5 = copy__deepcopy(xx1)
            yy1 = y1 + dict_t["position"]["y"] * dy
            # plot line
            if dict_t["line"]["linestyle"] is not None:
                ax.plot([xx1, xx3], [yy1, yy1], clip_on=False, zorder=11, **dict_t["line"])
                xx5 = copy__deepcopy(xx4)
            # plot marker
            if dict_t["marker"]["marker"] is not None:
                ax.scatter([xx2], [yy1], clip_on=False, zorder=12, **dict_t["marker"])
                xx5 = copy__deepcopy(xx4)
            # plot txt
            ax.text(xx5, yy1, txt, **dict_t["text"])


def _ax_text(
        ax,
        text: list = None,
        text_c: list = None,
        text_fs: list = None,
        text_ha: list = None,
        text_r: list = None,
        text_va: list = None,
        text_x: list = None,
        text_y: list = None,
        **kwarg):
    """
    Plot text

    Input:
    ------
    :param ax: matplotlib Axes object
    :param text, list, optional
        List of text to write; e.g., text = ['a', 'b', 'c']
    :param text_c, list, optional
        List of color for text to write; e.g., text_c = ['k', 'r', 'limegreen']
    :param text_fs, list, optional
        List of fontsize for text to write; e.g., text_fs = [12, 15, 20]
    :param text_ha, list, optional
        List of horizontal alignment for text to write; e.g., text_ha = ['center', 'left', 'right']
    :param text_r, list, optional
        List of rotation for text to write; e.g., text_r = [0, 45, 90]
    :param text_va, list, optional
        List of vertical alignment for text to write; e.g., text_va = ['bottom', 'center', 'top']
    :param text_x, list, optional
        List of horizontal position for text to write (using horizontal axis); e.g., text_x = [1, 2, 3]
    :param text_y, list, optional
        List of vertical position for text to write (using left vertical axis); e.g., text_y = [1, 2, 3]
    """
    if isinstance(text, list) is True and len(text) > 0 and isinstance(text_x, list) is True and \
            len(text_x) == len(text) and isinstance(text_y, list) is True and len(text_y) == len(text):
        if isinstance(text_c, list) is False or (isinstance(text_c, list) is True and len(text_c) != len(text)):
            text_c = [default_plot["color"]] * len(text)
        if isinstance(text_fs, list) is False or (isinstance(text_fs, list) is True and len(text_fs) != len(text)):
            text_fs = [default_plot["fontsize"]] * len(text)
        if isinstance(text_ha, list) is False or (isinstance(text_ha, list) is True and len(text_ha) != len(text)):
            text_ha = [default_plot["horizontalalignment"]] * len(text)
        if isinstance(text_r, list) is False or (isinstance(text_r, list) is True and len(text_r) != len(text)):
            text_r = [default_plot["rotation"]] * len(text)
        if isinstance(text_va, list) is False or (isinstance(text_va, list) is True and len(text_va) != len(text)):
            text_va = [default_plot["verticalalignment"]] * len(text)
        for t, c, f, h, r, v, x, y in zip(text, text_c, text_fs, text_ha, text_r, text_va, text_x, text_y):
            ax.text(x, y, t, color=c, fontsize=f, ha=h, rotation=r, va=v)


def _ax_titles(
        ax,
        fontsize: float = None,
        title: str = None,
        title_col: str = None,
        title_col_x: float = None,
        title_col_y: float = None,
        title_row: str = None,
        title_row_x: float = None,
        title_row_y: float = None,
        title_x: float = None,
        title_y: float = None,
        x_size: float = None,
        y_size: float = None,
        **kwarg):
    """
    Plot column and row names

    Input:
    ------
    :param ax: matplotlib Axes object
    :param fontsize: float
        Size of the font
    :param title: str, optional
        Name of the panel; e.g. title = 'a'
    :param title_col: str, optional
        Name of the column; e.g. title_col = 'sst'
    :param title_col_x: float, optional
        Horizontal position of the column name (percentage of horizontal axis); e.g., title_col_x = 50
    :param title_col_y: float, optional
        Vertical position of the column name (percentage of vertical axis); e.g., title_col_y = 10
    :param title_row: str, optional
        Name of the row; e.g. title_row = 'sst'
    :param title_row_x: float, optional
        Horizontal position of the row name (percentage of horizontal axis); e.g., title_row_x = 30
    :param title_row_y: float, optional
        Vertical position of the row name (percentage of vertical axis); e.g., title_row_y = 50
    :param title_x: float, optional
        Horizontal position of the panel name (percentage of horizontal axis); e.g., title_x = 4
    :param title_y: float, optional
        Vertical position of the panel name (percentage of vertical axis); e.g., title_y = 4
    :param x_size: float, optional
        Horizontal size of the plot; e.g., x_size = 8
    :param y_size: float, optional; e.g., y_size = 8
        Vertical size of the plot
    """
    # set default values if not given
    fontsize = none_to_default(fontsize, default_plot["fontsize"])
    title = none_to_default(title, default_plot["text"])
    title_col = none_to_default(title_col, default_plot["text"])
    title_col_x = none_to_default(title_col_x, default_plot["title_column_x"])
    title_col_y = none_to_default(title_col_y, default_plot["title_column_y"])
    title_row = none_to_default(title_row, default_plot["text"])
    title_row_x = none_to_default(title_row_x, default_plot["title_row_x"])
    title_row_y = none_to_default(title_row_y, default_plot["title_row_y"])
    title_x = none_to_default(title_x, default_plot["title_x"])
    title_y = none_to_default(title_y, default_plot["title_y"])
    x_size = none_to_default(x_size, default_plot["size_x"])
    y_size = none_to_default(y_size, default_plot["size_y"])
    # plot dimensions
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    dx = (x2 - x1) / 100.
    dy = (y2 - y1) / 100.
    # panel title
    if len(title) in [1, 2]:
        yy = y2 - title_y * dy * default_plot["size_y"] / y_size
        for k in list(range(len(title))):
            xx = x1 + (1 + k) * title_x * dx * default_plot["size_x"] / x_size
            ax.plot([xx], [yy], markersize=17, color="darkgrey", marker="s", fillstyle="full",
                    markeredgecolor="darkgrey", markeredgewidth=0, zorder=11)
        xx = x1 + (1 + (len(title) - 1) / 2) * title_x * dx * default_plot["size_x"] / x_size
        ax.text(xx, yy, title, fontsize=fontsize, fontweight="bold", zorder=12, ha="center", va="center")
    else:
        ax.set_title(title, fontsize=fontsize, y=1 + 0.02 * default_plot["size_y"] / y_size, loc="right")
    # column title
    xx = x1 + title_col_x * dx
    yy = y2 + title_col_y * dy * default_plot["size_y"] / y_size
    ax.text(xx, yy, title_col, fontsize=fontsize * 1.3, color="k", weight="bold", ha="center", va="center")
    # row title
    xx = x1 - title_row_x * dx * default_plot["size_x"] / x_size
    yy = y1 + title_row_y * dy
    ax.text(xx, yy, title_row, fontsize=fontsize * 1.3, color="k", weight="bold", ha="center", va="center", rotation=90)


def _plot_boxplot(
        ax,
        box_c: list = None,
        box_flie: list = None,
        box_fs: list = None,
        box_ls: list = None,
        box_lw: list = None,
        box_mean: list = None,
        box_ms: list = None,
        box_vert: list = None,
        box_w: list = None,
        box_x: list = None,
        box_y: list = None,
        box_z: list = None,
        **kwarg):
    """
    Plot boxplots

    Input:
    ------
    :param ax: matplotlib Axes object
    :param box_c: list, optional
        List of color for boxplots; e.g., box_c = ['k', 'b']
    :param box_flie: list, optional
        List of boolean outliers for boxplots, if True, show the outliers; e.g., box_flie = [True, False]
    :param box_fs: list, optional
        List of size of outlier marker for boxplots; e.g., box_fs = [2, 3]
    :param box_mean: list, optional
        List of boolean means for boxplots, if True, show the means; e.g., box_mean = [True, False]
    :param box_ms: list, optional
        List of size of mean marker for boxplots; e.g., box_ms = [14, 15]
    :param box_vert: list, optional
        List of boolean orientations for boxplots, if True, draws vertical, else horizontal;
        e.g., box_vert = [True, False]
    :param box_w: list, optional
        List of width for boxplots; e.g., box_w = [0.5, 0.5]
    :param box_x: list, optional
        List of x values for boxplots; e.g., box_x = [1, 2]
    :param box_y: list, optional
        List of y values for boxplots; e.g., box_y = [ [1, 2, 3], [1, 2, 3] ]
    :param box_z: list, optional
        List of zorder for boxplots (drawing order); e.g., box_z = [2, 4]
    """
    if isinstance(box_x, list) is True and isinstance(box_y, list) is True and len(box_x) == len(box_y):
        # control boxplot parameters
        if isinstance(box_c, list) is False or (isinstance(box_c, list) is True and len(box_c) != len(box_x)):
            box_c = [default_plot["color"]] * len(box_x)
        if isinstance(box_flie, list) is False or (isinstance(box_flie, list) is True and len(box_flie) != len(box_x)):
            box_flie = [default_plot["boxplot_flier"]] * len(box_x)
        if isinstance(box_fs, list) is False or (isinstance(box_fs, list) is True and len(box_fs) != len(box_x)):
            box_fs = [default_plot["boxplot_flier_size"]] * len(box_x)
        if isinstance(box_ls, list) is False or (isinstance(box_ls, list) is True and len(box_ls) != len(box_x)):
            box_ls = [default_plot["boxplot_linestyle"]] * len(box_x)
        if isinstance(box_lw, list) is False or (isinstance(box_lw, list) is True and len(box_lw) != len(box_x)):
            box_lw = [default_plot["boxplot_linewidth"]] * len(box_x)
        if isinstance(box_mean, list) is False or (isinstance(box_mean, list) is True and len(box_mean) != len(box_x)):
            box_mean = [default_plot["boxplot_mean"]] * len(box_x)
        if isinstance(box_ms, list) is False or (isinstance(box_ms, list) is True and len(box_ms) != len(box_x)):
            box_ms = [default_plot["boxplot_mean_size"]] * len(box_x)
        if isinstance(box_vert, list) is False or (isinstance(box_vert, list) is True and len(box_vert) != len(box_x)):
            box_vert = [default_plot["boxplot_vert"]] * len(box_x)
        if isinstance(box_w, list) is False or (isinstance(box_w, list) is True and len(box_w) != len(box_x)):
            dx = 1e20
            for k1 in range(len(box_x)):
                for k2 in range(k1 + 1, len(box_x)):
                    if abs(box_x[k1] - box_x[k2]) < dx:
                        dx = abs(box_x[k1] - box_x[k2])
            if dx == 1e20:
                dx = 1
            box_w = [dx * 0.7] * len(box_vert)
        if isinstance(box_z, list) is False or (isinstance(box_z, list) is True and len(box_z) != len(box_x)):
            box_z = [default_plot["boxplot_zorder"]] * len(box_x)
        # plot boxplot
        for c, f, fs, ls, lw, m, ms, v, w, x, y, z in zip(
                box_c, box_flie, box_fs, box_ls, box_lw, box_mean, box_ms, box_vert, box_w, box_x, box_y, box_z):
            dict_t = {
                "boxprops": dict(linestyle=ls, linewidth=lw, color=c),
                "capprops": dict(linestyle=ls, linewidth=lw, color=c),
                "flierprops": dict(marker="o", markersize=fs, markeredgecolor=c, markerfacecolor=c, markeredgewidth=0),
                "meanprops": dict(marker="D", markersize=ms, markeredgecolor=c, markerfacecolor=c, markeredgewidth=0),
                "medianprops": dict(linestyle=ls, linewidth=lw, color=c),
                "whiskerprops": dict(linestyle=ls, linewidth=lw, color=c)}
            ax.boxplot(y, positions=[x], whis=[5, 95], widths=w, labels=[""], showmeans=m, showfliers=f, vert=v,
                       zorder=z, **dict_t)
        # redo axes, the boxplot function causes troubles
        _ax_axis_x(ax, **kwarg)
        _ax_axis_y(ax, **kwarg)


def _plot_curve(
        ax,
        cur_c: list = None,
        cur_ls: list = None,
        cur_lw: list = None,
        cur_x: list = None,
        cur_y: list = None,
        cur_z: list = None,
        **kwarg):
    """
    Plot curves

    Input:
    ------
    :param ax: matplotlib Axes object
    :param cur_c: list, optional
        List of colors for curves; e.g., cur_c = ['k', 'b']
    :param cur_ls: list, optional
        List of linestyle for curves; e.g., cur_ls = ['-', ':']
    :param cur_lw: list, optional
        List of linewidth for curves; e.g., cur_lw = [0.5, 0.5]
    :param cur_x: list, optional
        List of x values for curves; e.g., cur_x = [ [1, 2, 3], [1, 2, 3] ]
    :param cur_y: list, optional
        List of y values for curves; e.g., cur_y = [ [10, 2, 10], [3, 6, 9] ]
    :param cur_z: list, optional
        List of zorder for curves (drawing order); e.g., cur_z = [2, 4]
    """
    if isinstance(cur_x, list) is True and isinstance(cur_y, list) is True and len(cur_x) == len(cur_y):
        # control curve parameters
        if cur_c is None or (isinstance(cur_c, list) is True and len(cur_c) != len(cur_x)):
            cur_c = [default_plot["color"]] * len(cur_x)
        if cur_ls is None or (isinstance(cur_ls, list) is True and len(cur_ls) != len(cur_x)):
            cur_ls = [default_plot["linestyle"]] * len(cur_x)
        if cur_lw is None or (isinstance(cur_lw, list) is True and len(cur_lw) != len(cur_x)):
            cur_lw = [default_plot["linewidth"]] * len(cur_x)
        if cur_z is None or (isinstance(cur_z, list) is True and len(cur_z) != len(cur_x)):
            cur_z = [default_plot["curve_zorder"]] * len(cur_x)
        # plot curves
        for c, ls, lw, x, y, z in zip(cur_c, cur_ls, cur_lw, cur_x, cur_y, cur_z):
            ax.plot(x, y, color=c, ls=ls, lw=lw, zorder=z)


def _plot_curve2(
        ax2,
        cur2_c: list = None,
        cur2_ls: list = None,
        cur2_lw: list = None,
        cur2_x: list = None,
        cur2_y: list = None,
        cur2_z: list = None,
        **kwarg):
    tmp_kwargs = {}
    l1 = ["cur_c", "cur_ls", "cur_lw2", "cur_x", "cur_y", "cur_z"]
    l2 = [cur2_c, cur2_ls, cur2_lw, cur2_x, cur2_y, cur2_z]
    for k1, k2 in zip(l1, l2):
        if k2 is not None:
            tmp_kwargs[k1] = k2
    _plot_curve(ax2, **tmp_kwargs)


def _plot_marker(
        ax,
        mar_c: list = None,
        mar_lc: list = None,
        mar_ls: list = None,
        mar_lw: list = None,
        mar_m: list = None,
        mar_s: list = None,
        mar_x: list = None,
        mar_y: list = None,
        mar_z: list = None,
        mar_colorbar: str = None,
        mar_colorbar_lab: list = None,
        mar_colorbar_nam: str = None,
        mar_colorbar_position: str = None,
        mar_colorbar_tic: list = None,
        **kwarg):
    """
    Plot markers

    Input:
    ------
    :param ax: matplotlib Axes object
    :param mar_c: list, optional
        List of facecolor for markers; e.g., mar_c = ['k', 'b']
    :param mar_colorbar: str, optional
        Name of a mar_colorbar_tics; e.g., mar_colorbar = 'cmo.rain'
    :param mar_colorbar_lab: list, optional
        Tick labels for colorbar; e.g., mar_colorbar_lab = ['1', '2']
    :param mar_colorbar_nam: str, optional
        Label for the colorbar; e.g., mar_colorbar_nam = 'temperature'
        Default is ''
    :param mar_colorbar_position: str, optional
        Position of the colorbar; e.g., mar_colorbar_position = 'bottom'
        Two colorbar positions are accepted: 'bottom', 'right'
        Default is 'bottom'
    :param mar_colorbar_tic: list, optional
        Ticks for colorbar; e.g., mar_colorbar_tic = [1, 2]
    :param mar_ls: list, optional
        List of linestyle for markers; e.g., mar_ls = ['-', ':']
    :param mar_lc: list, optional
        List of edgecolor for markers; e.g., mar_lc = ['k', 'b']
    :param mar_lw: list, optional
        List of linewidth for markers; e.g., mar_lw = [0.5, 0.5]
    :param mar_m: list, optional
        List of marker for markers; e.g., mar_m = ['D', 'o']
    :param mar_s: list, optional
        List of size for markers; e.g., mar_s = [50, 100]
    :param mar_x: list, optional
        List of x values for markers; e.g., mar_x = [ [1, 2, 3], [1, 2, 3] ]
    :param mar_y: list, optional
        List of y values for markers; e.g., mar_y = [ [10, 2, 10], [3, 6, 9] ]
    :param mar_z: list, optional
        List of zorder for markers (drawing order); e.g., mar_z = [2, 4]
    """
    # set default values if not given
    mar_colorbar_lab = none_to_default(mar_colorbar_lab, default_plot["lab"])
    mar_colorbar_nam = none_to_default(mar_colorbar_nam, default_plot["text"])
    mar_colorbar_position = none_to_default(mar_colorbar_position, default_plot["marker_position"])
    mar_colorbar_tic = none_to_default(mar_colorbar_tic, default_plot["tic"])
    # left axis
    if isinstance(mar_x, list) is True and isinstance(mar_y, list) is True and len(mar_x) == len(mar_y):
        # control marker parameters
        if mar_c is None or (isinstance(mar_c, list) is True and len(mar_c) != len(mar_x)):
            mar_c = [default_plot["color"]] * len(mar_x)
        if mar_lc is None or (isinstance(mar_lc, list) is True and len(mar_lc) != len(mar_x)):
            mar_lc = [default_plot["color"]] * len(mar_x)
        if mar_ls is None or (isinstance(mar_ls, list) is True and len(mar_ls) != len(mar_x)):
            mar_ls = [default_plot["linestyle"]] * len(mar_x)
        if mar_lw is None or (isinstance(mar_lw, list) is True and len(mar_lw) != len(mar_x)):
            mar_lw = [default_plot["linewidth"]] * len(mar_x)
        if mar_m is None or (isinstance(mar_m, list) is True and len(mar_m) != len(mar_x)):
            mar_m = [default_plot["marker"]] * len(mar_x)
        if mar_s is None or (isinstance(mar_s, list) is True and len(mar_s) != len(mar_x)):
            mar_s = [default_plot["size"]] * len(mar_x)
        if mar_z is None or (isinstance(mar_z, list) is True and len(mar_z) != len(mar_x)):
            mar_z = [default_plot["marker_zorder"]] * len(mar_x)
        # put parameters that are unique in one dictionary, the ones that are not in another
        ll1 = [mar_c, mar_lc, mar_ls, mar_lw, mar_m, mar_s, mar_z]
        ll2 = ["facecolors", "edgecolors", "linestyle", "linewidths", "marker", "s", "zorder"]
        # ll1 = [mar_ls, mar_m, mar_z]
        # ll2 = ["linestyle", "marker", "zorder"]
        dict_sca, dict_lis = {}, {}
        for k1, k2 in zip(ll1, ll2):
            ll3 = list(set(k1))
            if len(ll3) == 1:
                dict_sca[k2] = ll3[0]
            else:
                dict_lis[k2] = {"full": k1, "unique": ll3}
        scat = None
        if isinstance(mar_colorbar, str) is True and all([isinstance(k, str) is False for k in mar_c]):
            dif, mul = round(float(mar_colorbar_tic[1] - mar_colorbar_tic[0]), 5), 5
            delta = round(dif / mul, 5)
            lev = [round(k1 + k2 * delta, 5) for k1 in mar_colorbar_tic[:-1]
                   for k2 in range(mul)] + [mar_colorbar_tic[-1]]
            if mar_colorbar_nam == "correlation":
                cmap = pyplot.get_cmap("cmo.balance")
                cmap = cmap(numpy__linspace(0.15, 0.85, len(lev) - 1))
                cmap = ListedColormap(cmap)
            else:
                cmap = pyplot.get_cmap(mar_colorbar)
                cmap = cmap(numpy__linspace(0, 1, len(lev) - 1))
            cmap.set_bad("black")
            norm = BoundaryNorm(lev, ncolors=cmap.N, clip=True)
            dict_sca.update({"cmap": cmap, "norm": norm})
        else:
            mar_colorbar = None
        # plot markers
        if len(list(dict_lis.keys())) == 0:
            # all markers have the same parameters
            print("_plot_marker all markers have the same parameters")
            scat = ax.scatter(mar_x, mar_y, clip_on=False, **dict_sca)
        elif (len(mar_x) < 101 or len(list(dict_lis.keys())) > 3) and mar_colorbar is None:
            # few markers to plot --> plot them one by one
            # or
            # too few unique parameters --> plot markers one by one
            print("_plot_marker few markers to plot")
            for c, lc, ls, lw, m, s, x, y, z in zip(mar_c, mar_lc, mar_ls, mar_lw, mar_m, mar_s, mar_x, mar_y, mar_z):
                scat = ax.scatter(x, y, marker=m, s=s, facecolors=c, edgecolors=lc, linestyle=ls, linewidths=lw,
                                  clip_on=False, zorder=z)
        else:
            # there are several non unique parameters
            print("_plot_marker several non unique parameters")
            list_keys = list(dict_lis.keys())
            n1 = list_keys[0]
            f1, u1 = dict_lis[n1]["full"], dict_lis[n1]["unique"]
            if len(list_keys) == 1:
                for k1 in u1:
                    dict_t = {**dict_sca, **{n1: k1}}
                    list_ec, list_fc, list_lw, list_si, list_x, list_y = [], [], [], [], [], []
                    for i1, fc, ec, lw, si, xx, yy in zip(f1, mar_c, mar_lc, mar_lw, mar_s, mar_x, mar_y):
                        if i1 == k1:
                            list_ec.append(ec)
                            list_fc.append(fc)
                            list_lw.append(lw)
                            list_si.append(si)
                            list_x.append(xx)
                            list_y.append(yy)
                    scat = ax.scatter(list_x, list_y, c=list_fc, s=list_si, edgecolors=list_ec, linewidths=list_lw,
                                      clip_on=False, **dict_t)
            elif len(list_keys) == 2:
                n2 = list_keys[1]
                f2, u2 = dict_lis[n2]["full"], dict_lis[n2]["unique"]
                for k1 in u1:
                    for k2 in u2:
                        dict_t = {**dict_sca, **{n1: k1, n2: k2}}
                        list_ec, list_fc, list_lw, list_si, list_x, list_y = [], [], [], [], [], []
                        for i1, i2, fc, ec, lw, si, xx, yy in zip(
                                f1, f2, mar_c, mar_lc, mar_lw, mar_s, mar_x, mar_y):
                            if i1 == k1 and i2 == k2:
                                list_ec.append(ec)
                                list_fc.append(fc)
                                list_lw.append(lw)
                                list_si.append(si)
                                list_x.append(xx)
                                list_y.append(yy)
                        scat = ax.scatter(list_x, list_y, c=list_fc, s=list_si, edgecolors=list_ec, linewidths=list_lw,
                                          clip_on=False, **dict_t)
            elif len(list_keys) == 3:
                n2 = list_keys[1]
                f2, u2 = dict_lis[n2]["full"], dict_lis[n2]["unique"]
                n3 = list_keys[2]
                f3, u3 = dict_lis[n3]["full"], dict_lis[n3]["unique"]
                for k1 in u1:
                    for k2 in u2:
                        for k3 in u3:
                            dict_t = {**dict_sca, **{n1: k1, n2: k2, n3: k3}}
                            list_ec, list_fc, list_lw, list_si, list_x, list_y = [], [], [], [], [], []
                            for i1, i2, i3, fc, ec, lw, si, xx, yy in zip(
                                    f1, f2, f3, mar_c, mar_lc, mar_lw, mar_s, mar_x, mar_y):
                                if i1 == k1 and i2 == k2 and i3 == k3:
                                    list_ec.append(ec)
                                    list_fc.append(fc)
                                    list_lw.append(lw)
                                    list_si.append(si)
                                    list_x.append(xx)
                                    list_y.append(yy)
                            scat = ax.scatter(list_x, list_y, c=list_fc, s=list_si, edgecolors=list_ec,
                                              linewidths=list_lw, clip_on=False, **dict_t)
            else:
                print_fail(inspect__stack(), "too many non unique parameters for scatter")
        # colorbar
        fontsize = kwarg["fontsize"] if "fontsize" in list(kwarg.keys()) else default_plot["fontsize"]
        if mar_colorbar is not None and mar_colorbar_position == "bottom":
            x1, x2 = ax.get_position().x0, ax.get_position().x1
            y1, y2 = ax.get_position().y0, ax.get_position().y1
            cax = pyplot.axes([x1, y1 - (y2 - y1) / 3.5, x2 - x1, (y2 - y1) / 15])
            cbar = pyplot.colorbar(
                scat, cax=cax, orientation="horizontal", ticks=mar_colorbar_tic, label=mar_colorbar_lab, pad=0.3)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(mar_colorbar_nam, fontsize=fontsize, labelpad=2)
        elif mar_colorbar is not None and mar_colorbar_position == "right":
            x1, x2 = ax.get_position().x0, ax.get_position().x1
            y1, y2 = ax.get_position().y0, ax.get_position().y1
            cax = pyplot.axes([x2 + (x2 - x1) / 20, y1, (x2 - x1) / 30, y2 - y1])
            cbar = pyplot.colorbar(
                scat, cax=cax, orientation="vertical", ticks=mar_colorbar_tic, label=mar_colorbar_lab, pad=0.35)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(mar_colorbar_nam, fontsize=fontsize, rotation=90)
    return


def _plot_shading(
        ax,
        sha_a=None,
        sha_c=None,
        sha_x=None,
        sha_x2=None,
        sha_y1=None,
        sha_y2=None,
        sha_z=None,
        **kwarg):
    """
    Plot shading between y1 and y2

    Input:
    ------
    :param ax: matplotlib Axes object
    :param sha_a: list[float | int], optional
        List of alpha for shadings; e.g., sha_c = [0.5, 1]
    :param sha_c: list, optional
        List of color for shadings; e.g., sha_c = ['k', 'b']
    :param sha_x: list, optional
        List of x values for shadings; e.g., sha_x = [ [1, 2, 3], [1, 2, 3] ]
    :param sha_y1: list, optional
        List of lowest y values for shadings; e.g., sha_y1 = [ [1, 2, 3], [3, 4, 5] ]
    :param sha_y2: list, optional
        List of highest y values for shadings; e.g., sha_y2 = [ [2, 3, 4], [3, 6, 9] ]
    :param sha_z: list, optional
        List of zorder for shadings (drawing order); e.g., sha_z = [2, 4]
    """
    # left axis
    if isinstance(sha_x, list) is True and isinstance(sha_y1, list) is True and isinstance(sha_y2, list) is True \
            and len(sha_x) == len(sha_y1) and len(sha_x) == len(sha_y2):
        # control shading parameters
        if sha_a is None or (isinstance(sha_a, list) is True and len(sha_a) != len(sha_x)):
            sha_a = [default_plot["shading_alpha"]] * len(sha_x)
        if sha_c is None or (isinstance(sha_c, list) is True and len(sha_c) != len(sha_x)):
            sha_c = [default_plot["shading_color"]] * len(sha_x)
        if sha_z is None or (isinstance(sha_z, list) is True and len(sha_z) != len(sha_x)):
            sha_z = [default_plot["shading_zorder"]] * len(sha_x)
        if sha_x2 is None or (isinstance(sha_x2, list) is True and len(sha_x2) != len(sha_x)):
            sha_x2 = copy__deepcopy(sha_x)
        # plot shadings
        for a, c, x, y1, y2, z in zip(sha_a, sha_c, sha_x, sha_y1, sha_y2, sha_z):
            ax.fill_between(x, y1, y2=y2, alpha=a, color=c, linewidth=0.0, zorder=z)


def plot_hovmoeller(
        ax,
        contours1_kwargs: dict = None,
        contours2_kwargs: dict = None,
        fontsize: float = None,
        l_tic1: list = None,
        l_tic2: list = None,
        legend_position: str = "bottom",
        s_lab: list = None,
        s_nam: str = "",
        s_tic: list = None,
        sha_cs: str = None,
        sha_l1: array_wrapper = None,
        sha_l2: array_wrapper = None,
        sha_s: array_wrapper = None,
        **kwarg):
    """
    Plot hovmoeller

    :param ax: matplotlib Axes object
    :param contours1_kwargs: dict, optional
        kwargs for the first contours (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
    :param contours2_kwargs: dict, optional
        kwargs for the second contours (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
    :param fontsize: float, optional
        Size of the font; e.g., fontsize = 15.
    :param l_tic1: list[float], optional
        Levels for first contours; e.g., l_tic1 = [-1, 1]
    :param l_tic2: list[float], optional
        Levels for second contours; e.g., l_tic2 = [-1, 1]
    :param legend_position: str, optional
        Position of the legend; e.g., fig_legend_position = 'bottom'
        Two legend positions are accepted: 'bottom', 'right'
        Default is 'bottom'
    :param s_lab: list, optional
        Tick labels for colorbar; e.g., s_lab = ['1', '2']
    :param s_nam: str, optional
        Label for the colorbar
        Default is ''
    :param s_tic: list, optional
        Ticks for colorbar; e.g., s_lab = [1, 2]
    :param sha_cs: str, optional
        Name of a colorbar; e.g., sha_cs = 'cmo.rain'
    :param sha_l1: array_like
        Array for first contours
    :param sha_l2: array_like
        Array for second contours
    :param sha_s: array_like
        Array for shading
    :param kwarg: dict, optional
        Extra keywords (for title), see _ax_titles
    """
    # set default values if not given
    contours1_kwargs = none_to_default(contours1_kwargs, {})
    contours2_kwargs = none_to_default(contours2_kwargs, {})
    fontsize = none_to_default(fontsize, default_plot["fontsize"])
    s_lab = none_to_default(s_lab, default_plot["lab"])
    l_tic1 = none_to_default(l_tic1, default_plot["tic"])
    l_tic2 = none_to_default(l_tic2, default_plot["tic"])
    s_tic = none_to_default(s_tic, default_plot["tic"])
    sha_cs = none_to_default(sha_cs, default_map["colorbar"])
    if sha_s is not None:
        # x-y ticks and labels
        ax.tick_params(axis="both", direction="out", which="both", labelsize=fontsize, bottom=True, top=False,
                       left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        # colorbar levels
        dif, mul = round(float(s_tic[1] - s_tic[0]), 5), 5
        delta = round(dif / mul, 5)
        lev = [round(k1 + k2 * delta, 5) for k1 in s_tic[:-1] for k2 in range(mul)] + [s_tic[-1]]
        # shading
        dim_x, dim_y = convert_cf_dim_key(sha_s, "X"), convert_cf_dim_key(sha_s, "T")
        times, longitudes = sha_s[dim_y], sha_s[dim_x]
        xx, yy = numpy__meshgrid(longitudes, times)
        lc = ax.contourf(xx, yy, sha_s, lev, extend="both", cmap=sha_cs)
        if sha_l1 is not None:
            # contours
            times = sha_s[convert_cf_dim_key(sha_s, "T")]
            longitudes = sha_s[convert_cf_dim_key(sha_s, "X")]
            xx, yy = numpy__meshgrid(longitudes, times)
            ll = ax.contour(xx, yy, sha_l1, l_tic1, transform=ccrs.PlateCarree(), **contours1_kwargs)
            # ax.clabel(ll, fontsize=fontsize, inline=True)
        if sha_l2 is not None:
            # contours
            times = sha_s[convert_cf_dim_key(sha_s, "T")]
            longitudes = sha_s[convert_cf_dim_key(sha_s, "X")]
            xx, yy = numpy__meshgrid(longitudes, times)
            ll = ax.contour(xx, yy, sha_l2, l_tic2, transform=ccrs.PlateCarree(), **contours2_kwargs)
            # ax.clabel(ll, fontsize=fontsize, inline=True)
        # title
        _ax_titles(ax, **kwarg)
        _ax_axis_x(ax, **kwarg)
        _ax_axis_y(ax, **kwarg)
        _ax_text(ax, **kwarg)
        # colorbar
        if legend_position == "bottom":
            x1, x2 = ax.get_position().x0, ax.get_position().x1
            y1, y2 = ax.get_position().y0, ax.get_position().y1
            dy = default_map["size_y"] / kwarg["y_size"]
            cax = pyplot.axes([x1, y1 - (y2 - y1) * dy / 5, x2 - x1, (y2 - y1) * dy / 15])
            cbar = pyplot.colorbar(lc, cax=cax, orientation="horizontal", ticks=s_tic, label=s_lab, pad=0.3)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(s_nam, fontsize=fontsize, labelpad=2)
        elif legend_position == "right":
            x1, x2 = ax.get_position().x0, ax.get_position().x1
            y1, y2 = ax.get_position().y0, ax.get_position().y1
            dx = default_map["size_x"] / kwarg["x_size"]
            cax = pyplot.axes([x2 + (x2 - x1) * dx / 20, y1, (x2 - x1) * dx / 30, y2 - y1])
            cbar = pyplot.colorbar(lc, cax=cax, orientation="vertical", ticks=s_tic, label=s_lab, pad=0.35)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(s_nam, fontsize=fontsize, rotation=90)


def plot_main(ax, **kwarg):
    """
    Plot given boxplots, curves, histograms, markers and shadings

    Input:
    ------
    :param ax: matplotlib Axes object

    """
    # x-axis
    _ax_axis_x(ax, **kwarg)
    # y-axis
    _ax_axis_y(ax, **kwarg)
    # title
    _ax_titles(ax, **kwarg)
    # shading
    _plot_shading(ax, **kwarg)
    # boxplot
    _plot_boxplot(ax, **kwarg)
    # curve
    _plot_curve(ax, **kwarg)
    if "cur2_x" in list(kwarg.keys()) and "cur2_y" in list(kwarg.keys()):
        ax2 = ax.twinx()
        if "y2_lim" in list(kwarg.keys()) and "y2_tic" in list(kwarg.keys()):
            print("_ax_axis_y2")
            _ax_axis_y2(ax2, **kwarg)
        _plot_curve2(ax2, **kwarg)
    # marker
    _plot_marker(ax, **kwarg)
    # text
    _ax_text(ax, **kwarg)
    # legend
    _ax_legend(ax, **kwarg)


def plot_map(
        ax,
        contours1_kwargs: dict = None,
        contours2_kwargs: dict = None,
        fontsize: float = None,
        l_tic1: list = None,
        l_tic2: list = None,
        legend_position: str = "bottom",
        projection: object = None,
        region: list = None,
        s_lab: list = None,
        s_nam: str = "",
        s_tic: list = None,
        sha_cl: str = None,
        sha_cs: str = None,
        sha_l1: array_wrapper = None,
        sha_l2: array_wrapper = None,
        sha_s: array_wrapper = None,
        **kwarg):
    """
    Plot map

    :param ax: matplotlib Axes object
    :param contours1_kwargs: dict, optional
        kwargs for the first contours (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
    :param contours2_kwargs: dict, optional
        kwargs for the second contours (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html)
    :param fontsize: float, optional
        Size of the font; e.g., fontsize = 15.
    :param l_tic1: list[float], optional
        Levels for first contours; e.g., l_tic1 = [-1, 1]
    :param l_tic2: list[float], optional
        Levels for second contours; e.g., l_tic2 = [-1, 1]
    :param legend_position: str, optional
        Position of the legend; e.g., fig_legend_position = 'bottom'
        Two legend positions are accepted: 'bottom', 'right'
        Default is 'bottom'
    :param projection: object, optional
        Output from a cartopy projection (see https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html);
        e.g., projection = cartopy.crs.PlateCarree()
    :param region: list, optional
        Names of region to draw on the map, must be defined in default_region; e.g., region = ['n30e']
        Default is None (no region)
    :param s_lab: list, optional
        Tick labels for colorbar; e.g., s_lab = ['1', '2']
    :param s_nam: str, optional
        Label for the colorbar
        Default is ''
    :param s_tic: list, optional
        Ticks for colorbar; e.g., s_lab = [1, 2]
    :param sha_cl: str, optional
        Name of a color for the land; e.g., sha_cl = 'grey'
    :param sha_cs: str, optional
        Name of a colorbar; e.g., sha_cs = 'cmo.rain'
    :param sha_l1: array_like
        Array for first contours
    :param sha_l2: array_like
        Array for second contours
    :param sha_s: array_like
        Array for shading
    :param kwarg: dict, optional
        Extra keywords (for title), see _ax_titles
    """
    # set default values if not given
    contours1_kwargs = none_to_default(contours1_kwargs, {})
    contours2_kwargs = none_to_default(contours2_kwargs, {})
    fontsize = none_to_default(fontsize, default_plot["fontsize"])
    projection = none_to_default(projection, default_map["projection"])
    s_lab = none_to_default(s_lab, default_plot["lab"])
    l_tic1 = none_to_default(l_tic1, default_plot["tic"])
    l_tic2 = none_to_default(l_tic2, default_plot["tic"])
    s_tic = none_to_default(s_tic, default_plot["tic"])
    sha_cl = none_to_default(sha_cl, default_map["color"])
    sha_cs = none_to_default(sha_cs, default_map["colorbar"])
    if sha_s is not None:
        # select region
        ax.coastlines()
        # ax.set_extent(x_lim + y_lim, crs=projection)
        # draw coastlines
        ax.coastlines()
        # fill continents
        ax.add_feature(cfeature.NaturalEarthFeature(
            "physical", "land", "110m", edgecolor="face", facecolor=sha_cl))
        # x-y ticks and labels
        ax.tick_params(axis="both", direction="out", which="both", labelsize=fontsize, bottom=True, top=False,
                       left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        # colorbar levels
        dif, mul = round(float(s_tic[1] - s_tic[0]), 5), 5
        delta = round(dif / mul, 5)
        lev = [round(k1 + k2 * delta, 5) for k1 in s_tic[:-1] for k2 in range(mul)] + [s_tic[-1]]
        # shading
        latitudes = sha_s[convert_cf_dim_key(sha_s, "Y")]
        longitudes = sha_s[convert_cf_dim_key(sha_s, "X")]
        xx, yy = numpy__meshgrid(longitudes, latitudes)
        lc = ax.contourf(xx, yy, sha_s, lev, extend="both", transform=projection, cmap=sha_cs)
        if sha_l1 is not None:
            # contours
            latitudes = sha_l1[convert_cf_dim_key(sha_l1, "Y")]
            longitudes = sha_l1[convert_cf_dim_key(sha_l1, "X")]
            xx, yy = numpy__meshgrid(longitudes, latitudes)
            ll = ax.contour(xx, yy, sha_l1, l_tic1, transform=ccrs.PlateCarree(), **contours1_kwargs)
            # ax.clabel(ll, fontsize=fontsize, inline=True)
        if sha_l2 is not None:
            # contours
            latitudes = sha_l2[convert_cf_dim_key(sha_l2, "Y")]
            longitudes = sha_l2[convert_cf_dim_key(sha_l2, "X")]
            xx, yy = numpy__meshgrid(longitudes, latitudes)
            ll = ax.contour(xx, yy, sha_l2, l_tic2, transform=ccrs.PlateCarree(), **contours2_kwargs)
            # ax.clabel(ll, fontsize=fontsize, inline=True)
        # title
        _ax_titles(ax, **kwarg)
        _ax_axis_x(ax, **kwarg)
        _ax_axis_y(ax, **kwarg)
        _ax_text(ax, **kwarg)
        # region delimitation
        if isinstance(region, list) is True or isinstance(region, str) is True:
            list_reg = deepcopy(region) if isinstance(region, list) is True else [region]
            for k in list_reg:
                lat_corners = numpy__array(default_region[k]["latitude"] +
                                           list(reversed(default_region[k]["latitude"])))
                lon_corners = numpy__array([default_region[k]["longitude"][0]] * 2 +
                                           [default_region[k]["longitude"][1]] * 2)
                poly_corners = numpy__zeros((len(lat_corners), 2))
                poly_corners[:, 0] = lon_corners
                poly_corners[:, 1] = lat_corners
                pol = Polygon(poly_corners, linestyle=default_region[k]["linestyle"],
                              linewidth=default_region[k]["linewidth"], edgecolor=default_region[k]["color"],
                              facecolor="none", zorder=5, transform=projection)
                ax.add_patch(pol)
        # colorbar
        if legend_position == "bottom":
            x1, x2 = ax.get_position().x0, ax.get_position().x1
            y1, y2 = ax.get_position().y0, ax.get_position().y1
            dy = default_map["size_y"] / kwarg["y_size"]
            cax = pyplot.axes([x1, y1 - (y2 - y1) * dy / 5, x2 - x1, (y2 - y1) * dy / 15])
            cbar = pyplot.colorbar(lc, cax=cax, orientation="horizontal", ticks=s_tic, label=s_lab, pad=0.3)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(s_nam, fontsize=fontsize, labelpad=2)
        elif legend_position == "right":
            x1, x2 = ax.get_position().x0, ax.get_position().x1
            y1, y2 = ax.get_position().y0, ax.get_position().y1
            dx = default_map["size_x"] / kwarg["x_size"]
            cax = pyplot.axes([x2 + (x2 - x1) * dx / 20, y1, (x2 - x1) * dx / 30, y2 - y1])
            cbar = pyplot.colorbar(lc, cax=cax, orientation="vertical", ticks=s_tic, label=s_lab, pad=0.35)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(s_nam, fontsize=fontsize, rotation=90)
# ---------------------------------------------------------------------------------------------------------------------#
