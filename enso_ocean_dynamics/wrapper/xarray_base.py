# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# N-dimensional array with labeled coordinates and dimensions are needed for the package.
# In xarray:
#     - A dataset resembles an in-memory representation of a NetCDF file, and consists of variables, coordinates and
#       attributes which together form a self describing dataset.
#     - A DataArray provides a wrapper around numpy ndarrays that uses labeled dimensions and coordinates to support
#       metadata aware operations.
# This file builds on xarray functions.
# https://docs.xarray.dev/en/latest/
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy
from inspect import stack as inspect__stack
from re import split as re__split
from typing import Hashable, Literal, Union
# numpy
from numpy import array as numpy__array
# xarray
import xarray

# local functions
from enso_ocean_dynamics.tools.print_tools import print_fail
from enso_ocean_dynamics.wrapper.calendar_handler import *
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
array_wrapper = xarray.DataArray


def check_time_bounds(
        ds: array_wrapper,
        dim: Union[Hashable, str],
        time_bounds: Union[tuple[str, str]],
        side: Literal["lower", "upper"],
        **kwargs) -> array_wrapper:
    # bound position (0 or 1)
    position = 0 if side == "lower" else 1
    # split time bound in ["year", "month", "day"]
    split_ta = re__split("-| |:", str(get_time_bounds(ds)[position]))
    split_td = re__split("-| |:", time_bounds[position])
    while (side == "lower" and all([float(k1) >= float(k2) for k1, k2 in zip(split_ta, split_td)]) is False) or \
            (side == "upper" and all([float(k1) <= float(k2) for k1, k2 in zip(split_ta, split_td)]) is False):
        # if side == "lower" & one of available("year", "month", "day") is smaller than desired ("year", "month", "day")
        # or
        # if side == "upper" & one of available("year", "month", "day") is larger than desired ("year", "month", "day")
        # remove the first (last) time step
        time_slice = slice(1, int(1e20)) if side == "lower" else slice(0, -1)
        ds.isel(indexers={dim: time_slice})
        split_ta = re__split("-| |:", str(get_time_bounds(ds)[position]))
    return ds


def convert_cf_dim_key(ds: array_wrapper, cf_dim: Literal["T", "X", "Y", "Z"], **kwargs) -> str:
    """
    Return dimension name corresponding to CF dimension name (T is time, X is longitude, Y is latitude, Z is depth or
    level).

    Input:
    ------
    :param ds: xarray.DataArray
    :param cf_dim: {"X", "Y", "T", "Z"}
        Name of a CF dimension
    **kwargs - Discarded

    Output:
    -------
    :return: str
        Name of given CF dimension in input xarray.DataArray
    """
    # list dimension
    list_dim = list(ds.dims)
    # dimension to find
    dim_to_find = {
        "T": ["time"],
        "X": ["lon"],
        "Y": ["lat"],
        "Z": ["depth", "height", "level", "pressure", "vertical", "lev"]}
    # find the name in the xarray.DataArray
    dim_o = ""
    for k1 in dim_to_find[cf_dim]:
        for k2 in list_dim:
            if k1 in k2:
                dim_o = copy__deepcopy(k2)
                break
        if dim_o != "":
            break
    if dim_o == "":
        error = str().ljust(5) + "cannot find " + str(cf_dim)
        error += "\n" + str().ljust(5) + "dimension(s): " + ", ".join([repr(k) for k in list_dim])
        print_fail(inspect__stack(), error)
    return dim_o


def convert_dim_keys(
        ds: array_wrapper,
        dim: Union[Hashable, str, list[Hashable], list[str], tuple[Hashable], tuple[str], None],
        **kwargs) -> Union[str, list[str]]:
    """
    Return dimension name(s) from input xarray.DataArray

    Input:
    ------
    :param ds: xarray.DataArray
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None
        Name(s) of dimension or CF dimension
    **kwargs - Discarded

    Output:
    -------
    :return: str
        Name(s) of dimension in input xarray.DataArray
    """
    dim_o = None
    if dim is not None:
        # input dimension to list
        dimensions_asked = copy__deepcopy(dim)
        if isinstance(dim, (Hashable, str)) is True:
            dimensions_asked = [dim]
        # dimensions in input xarray.DataArray
        dimensions_available = list(ds.dims)
        # match asked and available dimensions
        dim_o = []
        for k in dimensions_asked:
            if k in dimensions_available:
                dim_o.append(k)
            elif isinstance(k, (Hashable, str)) is True and k in ["T", "X", "Y", "Z"]:
                dim_o.append(convert_cf_dim_key(ds, k))
            else:
                error = str().ljust(5) + "cannot find " + str(k)
                error += "\n" + str().ljust(5) + "dimension(s): " + ", ".join([repr(k) for k in dimensions_available])
                print_fail(inspect__stack(), error)
        # list to str if needed
        if isinstance(dim, (Hashable, str)) is True and len(dim_o) == 1:
            dim_o = dim_o[0]
    return dim_o


def correlation(
        ds_a: array_wrapper,
        ds_b: array_wrapper,
        data_var_a: str = None,
        data_var_b: str = None,
        dim: Union[Hashable, str, list[Hashable], list[str], tuple[Hashable], tuple[str], None] = None,
        weights: Union[array_wrapper, None] = None,
        **kwargs) -> array_wrapper:
    """
    Compute the Pearson correlation coefficient between two DataArray objects along shared dimension(s).
    https://docs.xarray.dev/en/latest/generated/xarray.corr.html

    Input:
    ------
    :param ds_a: xarray.DataArray
    :param ds_b: xarray.DataArray
    :param data_var_a: str, optional
        Data variable in ‘ds_a‘; e.g., data_var_a = "ts".
        Default is None
    :param data_var_b: str, optional
        Data variable in ‘ds_b‘; e.g., data_var_b = "ts".
        Default is None
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension[s] along which to apply var; e.g., dim="x" or dim=["x", "y"].
        If None, will reduce over all dimensions.
        Default is None
    :param weights: xarray.DataArray, optional
        Array of weights.
        Default is None
     **kwargs - Discarded

    Output:
    -------
    :return: xarray.DataArray
        New object with dimensions, attributes, coordinates, name, encoding, with correlation applied to given data and
        the indicated dimension(s) removed.
    """
    # get dimension(s) as named in xarray.DataArray
    dim_name = convert_dim_keys(ds_a, dim)
    # correlation value
    ds_o = xarray.corr(ds_a, ds_b, dim=dim_name, weights=weights)
    # attributes
    att_a, att_b = ds_a.attrs, ds_b.attrs
    for k in ["freq", "mode", "operation", "units", "weighted"]:
        if k in list(att_a.keys()):
            del att_a[k]
        if k in list(att_b.keys()):
            del att_b[k]
    attributes = copy__deepcopy(att_a)
    for k1, k2 in att_b.items():
        if k1 in list(attributes.keys()) and k1 == "long_name":
            attributes[k1] = str(att_a[k1]).split(" time series")[0] + " & " + str(k2).split(" time series")[0] + \
                             " correlation"
        elif k1 in list(attributes.keys()) and k1 == "short_name":
            attributes[k1] = "corr (" +str(att_a[k1]) + " & " + str(k2) + ")"
        elif k1 in list(attributes.keys()) and att_a[k1] != k2:
            attributes[k1] = str(data_var_a) + " -- " + str(att_a[k1]) + "\n" + str(data_var_b) + " -- " + str(k2)
        else:
            attributes[k1] = k2
    attributes["units"] = ""
    ds_o.attrs.update(**attributes)
    return ds_o


def create_array(
        ds: array_wrapper,
        data_var: str = "",
        value: Union[float, int, None] = 0,
        **kwargs) -> array_wrapper:
    """
    Return a new DataArray of ’value’ with the same shape, axes, coordinates, attributes,... as input DataArray.
    https://docs.xarray.dev/en/latest/generated/xarray.zeros_like.html

    Input:
    ------
    :param ds: xarray.DataArray
    :param data_var: str, optional
        Name of the output data variable.
        Default is ""
    :param value: float or int or None, optional
        Value to fill the array with; e.g., value = 1.
        If None, array filled with NaN.
        Default is 0
    **kwargs - Discarded

    Output:
    -------
    :return: xarray.DataArray
        New DataArray of ones with the same shape and type as ds.
    """
    # create an array of zeros with the same shape, axes, coordinates, attributes,... as input DataArray
    da_o = xarray.zeros_like(ds)
    # name this new array
    da_o.name = data_var
    if value is None:
        # fill with NaN
        da_o = da_o.where(da_o == 1)
    elif isinstance(value, (float, int)) is True and value != 0:
        # fill with value
        da_o = da_o.where(da_o == value, other=value)
    return da_o


def get_axis_dim(
        ds: array_wrapper,
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[Hashable], list[str], tuple[Hashable], tuple[str], None] = None,
        **kwargs) -> Union[tuple[int, str], tuple[int, Hashable], tuple[list[int], list[str]]]:
    """
    Return axis(es) corresponding to given dimension name(s) and conversely.

    Input:
    ------
    :param ds: xarray.DataArray
    :param axis: int or list[int] or tuple[int], optional
        Axis(es) for which the dimension name(s) must be found; e.g., axis = 0
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str], optional
        Dimension(s) for which the axis number(s) must be found; e.g., dim = "time"
        Only one of the 'axis' and 'dim' arguments can be supplied

    Outputs:
    --------
    :return axis: int or list[int] or tuple[int]
        Axis(es) corresponding to given dimension name(s) or input axis(es)
    :return dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str]
        Dimension(s) corresponding to given axis(es) or input dimension(s)
    """
    # get dimension listed as in xarray.DataArray
    list_dim = list(ds.dims)
    if dim is not None:
        # get axis(es) corresponding to dimension name(s)
        if isinstance(dim, str) is True:
            axis = list_dim.index(dim)
        else:
            axis = [list_dim.index(k) for k in dim]
    elif axis is not None:
        # get dimension name(s) corresponding to axis(es)
        if isinstance(axis, int) is True:
            dim = list_dim[axis]
        else:
            dim = [list_dim[k] for k in axis]
    return axis, dim



def get_season(
        ds: array_wrapper,
        season: Literal["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"],
        **kwargs):
    """
    Get values for given season.

    Input:
    ------
    :param ds: xarray.DataArray
    :param season: {"DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"}
        Name of a season
    **kwargs - Discarded
    :return:
    """
    # time dimension name
    dim_name = convert_cf_dim_key(ds, "T")
    # Use .groupby('time.month') to organize the data into months
    # then use .groups to extract the indices for each month
    month_idxs = ds.groupby(group="time.month").groups
    # Extract the time indices corresponding to all 'season'
    centered_month_per_season = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11,
        "NDJ": 12,
    }
    idx = month_idxs[centered_month_per_season[season]]
    # Extract the 'season' months by selecting the relevant indices
    ds_o = ds.isel(indexers={dim_name: idx})
    # get time index
    time = ds[dim_name].to_index()
    years = time.year
    if season == "DJF":
        years += 1
    # change dimension
    ds_o = ds_o.assign_coords(coords={dim_name: numpy__array(list(years))})
    ds_o = ds_o.rename({dim_name: "year"})
    # remove first or last time step depending on the case
    # rolling computes the first time step as DJF and last time step as NDJ, fills them with nans
    if season == "DJF":
        ds_o = ds_o[1:]
    elif season == "NDJ":
        ds_o = ds_o[:-1]
    return ds_o


def get_time_bounds(ds: array_wrapper, **kwargs) -> list[str]:
    """
    Return first and last time values of given xarray.DataArray

    Input:
    ------
    :param ds: xarray.DataArray
    **kwargs - Discarded

    Output:
    -------
    :return: list[str]
        First and last time values.
    """
    dim_name = convert_cf_dim_key(ds, "T")
    time_initial = ds[dim_name][0].values.item().isoformat().split("T")[0]
    time_final = ds[dim_name][-1].values.item().isoformat().split("T")[0]
    return [time_initial, time_final]


def time_weights(ds: array_wrapper) -> array_wrapper:
    # get time array, index, calendar
    dim_name = convert_cf_dim_key(ds, "T")
    time_arr = ds[dim_name]
    time_index = time_arr.to_index()
    calendar = ds.dt.calendar
    # get month length
    month_length = array_wrapper(
        get_days_per_month(time_index, calendar=calendar), coords=[time_arr], name="month_length")
    return month_length
# ---------------------------------------------------------------------------------------------------------------------#
