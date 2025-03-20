# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Processors built over xarray
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy
from glob import iglob as glob__iglob
from os.path import dirname as os__path__dirname
from os.path import join as os__path__join
from typing import Hashable, Literal, Union
# numpy
from numpy import array as numpy__array
from numpy import zeros as numpy__zeros
# xarray
import xarray

# local functions
from enso_ocean_dynamics.tools.default_tools import none_to_default, set_instance
from enso_ocean_dynamics.tools.dictionary_tools import put_in_dict, tuple_for_dict
from enso_ocean_dynamics.wrapper import wrapper_tools
from enso_ocean_dynamics.wrapper import xarray_base
from enso_ocean_dynamics.wrapper.xarray_base import array_wrapper
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
def _modify_metadata_short_name(name_i):
    name_o = name_i.replace("N30", "N3").replace("N34", "N3.4").replace("N40", "N4")
    for k1 in ["N3", "N3.4", "N4"]:
        for k2 in ["E", "N", "S"]:
            if str(k1) + str(k2) in name_o:
                if k2 == "E":
                    name_o = name_o.replace(str(k1) + str(k2), k1)
                else:
                    name_o = name_o.replace(str(k1) + str(k2), str(k1) + "$_{" + str(k2) + "}$")
    name_o = name_o.replace("PEQU", "EP").replace("PEQE", "EEP").replace("PEQW", "WEP")
    return name_o


def _modify_metadata_units(name_i):
    name_o = name_i.replace("degC", "$^\circ$C")
    for k in range(-20, 21):
        name_o = name_o.replace("**" + str(k), "$^{" + str(k) + "}$")
    return name_o


def _read_netcdf(
        file_name: str,
        data_var: str = None,
        **kwargs):
    """
    Read neCDF file

    Input:
    ------
    :param file_name: str
        Name of the file to open
    :param data_var: str, optional
        The key of the data variable to keep in the Dataset; e.g., data_var = "ts".
        Default is None
    **kwargs – Additional keyword arguments passed on to xarray.open_mfdataset.

    Output:
    -------
    :return: xarray.Dataset
        Newly created dataset.
    :return dataset: str
        Name of the dataset, as specified in the file
    :return member: str
        Name of the member, as specified in the file
    """
    # open file
    ds = xarray.open_mfdataset(file_name, **kwargs)
    # available variables
    list_variables = sorted(list(ds.keys()), key=lambda v: v.lower())
    if data_var in list_variables:
        # read variable dataset -> dataArray
        ds = ds[data_var]
        # read variable attributes
        att_i = ds.attrs
        att_o = {}
        for k1, k2 in att_i.items():
            if k1 == "short_name":
                k2 = _modify_metadata_short_name(k2)
            elif k1 == "units":
                k2 = _modify_metadata_units(k2)
            # update attribute
            att_o[k1] = copy__deepcopy(k2)
        # update variable attributes
        ds.attrs.update(**att_o)
    else:
        ds = None
    return ds


def confidence_interval(
        dict_x: Union[dict, array_wrapper],
        dict_y: Union[dict, array_wrapper],
        stat: str,
        confidence_level: float = 95,
        method: Literal["identical", "random"] = "identical",
        pseudo_sample_size: int = 10000,
        random_eel: bool = True,
        sample_size: Union[int, None] = None,
        dict_o: dict = None,
        tuple_k: tuple = None,
        tuple_k_last: tuple = None,
        **kwargs) -> (dict, tuple, tuple):
    # set None input to it's default value
    dict_o = none_to_default(dict_o, {})
    tuple_k = none_to_default(tuple_k, ())
    tuple_k_last = none_to_default(tuple_k_last, ())
    # loop through nested levels
    if isinstance(dict_x, dict) is True:
        list_keys = sorted(list(dict_x.keys()), key=lambda v: v.lower())
        for k in list_keys:
            dict_o, tuple_k, tuple_k_last = confidence_interval(
                dict_x[k], dict_y[k], stat, confidence_level=confidence_level, method=method,
                pseudo_sample_size=pseudo_sample_size, random_eel=random_eel, sample_size=sample_size, dict_o=dict_o,
                tuple_k=tuple_k + (k,), tuple_k_last=tuple_k_last + (list_keys[-1],), **kwargs)
    else:
        # compute correlation using resample array
        ds_o = wrapper_tools.statistic_resampled(
            dict_x, arr_i2=dict_y, data_var_i1=tuple_k[0], data_var_i2=tuple_k[0], dim="year", method=method,
            pseudo_sample_size=pseudo_sample_size, random_eel=random_eel, sample_size=sample_size, stat=stat, **kwargs)
        # lower and higher percentile of the confidence interval
        threshold = [50 + k * confidence_level / 2 for k in [-1, 1]]
        # compute confidence interval
        ds_o = wrapper_tools.statistic(ds_o, dim="resample", stat="per", per=threshold, keep_attrs=True)
        # save values
        put_in_dict(dict_o, ds_o, *tuple_k)
        # remove relevant keys from the tuples of keys
        tuple_k, tuple_k_last = tuple_for_dict(tuple_k, tuple_k_last)
    return dict_o, tuple_k, tuple_k_last


def correlation(
        dict_x: Union[dict, array_wrapper],
        dict_y: Union[dict, array_wrapper],
        dict_o: dict = None,
        tuple_k: tuple = None,
        tuple_k_last: tuple = None,
        **kwargs) -> (dict, tuple, tuple):
    # set None input to it's default value
    dict_o = none_to_default(dict_o, {})
    tuple_k = none_to_default(tuple_k, ())
    tuple_k_last = none_to_default(tuple_k_last, ())
    # loop through nested levels
    if isinstance(dict_x, dict) is True:
        list_keys = sorted(list(dict_x.keys()), key=lambda v: v.lower())
        for k in list_keys:
            dict_o, tuple_k, tuple_k_last = correlation(
                dict_x[k], dict_y[k], dict_o=dict_o, tuple_k=tuple_k + (k,),
                tuple_k_last=tuple_k_last + (list_keys[-1],), **kwargs)
    else:
        # compute correlation
        ds_o = xarray_base.correlation(dict_x, dict_y, data_var_a=tuple_k[0], data_var_b=tuple_k[0], dim="year")
        # save values
        put_in_dict(dict_o, ds_o, *tuple_k)
        # remove relevant keys from the tuples of keys
        tuple_k, tuple_k_last = tuple_for_dict(tuple_k, tuple_k_last)
    return dict_o, tuple_k, tuple_k_last


def reader(
        data_datasets: list = None,
        data_epoch: str = None,
        data_experiments: list = None,
        data_members: list = None,
        data_projects: list = None,
        data_type: str = None,
        data_variables: list = None,
        **kwargs) -> dict:
    # set None input to it's default value
    data_datasets = set_instance(data_datasets, list, False, ["*"])
    data_experiments = set_instance(data_experiments, list, False, ["*"])
    data_members = set_instance(data_members, list, False, ["*"])
    data_projects = set_instance(data_projects, list, False, ["*"])
    # plot directory (relative to current file directory)
    data_directory = "/".join(os__path__dirname(__file__).split("/")[:-2]) + "/data"
    # get data
    dict_o = {}
    for pro in data_projects:  # loop on projects
        for dat in data_datasets:  # loop on datasets
            for exp in data_experiments:  # loop on experiments
                for mem in data_members:  # loop on members
                    for var in data_variables:  # loop on variables
                        # construct filename
                        file_name = os__path__join(
                            data_directory, "_".join([pro, dat, exp, mem, data_epoch, data_type]) + ".nc")
                        file_names = sorted(list(glob__iglob(file_name)), key=lambda s: s.lower())
                        for fil in file_names:  # loop on files
                            # read dataset
                            arr = _read_netcdf(fil, data_var=var, **kwargs)
                            if arr is None:
                                continue
                            # get names
                            project, dataset, experiment, member = fil.split("/")[-1].split("_")[:4]
                            # save value
                            put_in_dict(dict_o, arr, var, project, experiment, dataset, member)
    return dict_o


def reorder_time_series(
        dict_i: Union[dict, array_wrapper],
        window: int,
        remove_eel: bool = False,
        dict_o: dict = None,
        tuple_k: tuple = None,
        tuple_k_last: tuple = None,
        **kwargs) -> (dict, tuple, tuple):
    # set None input to it's default value
    dict_o = none_to_default(dict_o, {})
    tuple_k = none_to_default(tuple_k, ())
    tuple_k_last = none_to_default(tuple_k_last, ())
    # loop through nested levels
    if isinstance(dict_i, dict) is True:
        list_keys = sorted(list(dict_i.keys()), key=lambda v: v.lower())
        for k in list_keys:
            dict_o, tuple_k, tuple_k_last = reorder_time_series(
                dict_i[k], window, remove_eel=remove_eel, dict_o=dict_o, tuple_k=tuple_k + (k,),
                tuple_k_last=tuple_k_last + (list_keys[-1],), **kwargs)
    else:
        # get time array, index, calendar
        dim_name = xarray_base.convert_cf_dim_key(dict_i, "T")
        time_index = dict_i[dim_name].to_index()
        # get numpy array
        arr_i = dict_i.to_numpy()
        # reorganize time series
        arr_o = []
        half = window // 2
        h1 = half - 12
        h2 = half + 12
        for k in range(-h1, len(arr_i) - h2 + 1, 12):
            # 'window' sized array
            arr_t = numpy__zeros(window)
            # time steps taken from input array
            i1, i2 = max(0, k), min(k + window, len(arr_i))
            # time steps where to place input slice in output array
            o1, o2 = 0, copy__deepcopy(window)
            if k < 0:
                # the first 'h1' time steps are not centered, they need to be handled properly
                o1 = i1 - k
            elif k + window > len(arr_i):
                # the last 'window' time steps are not centered, they need to be handled properly
                o2 = o1 + i2 - i1
            arr_t[o1: o2] = arr_i[i1: i2]
            if remove_eel is True:
                # extreme el nino
                list_eel = [1972, 1982, 1997, 2015]
                # get time axis of given window and
                for i1, t1 in zip(range(o1, o2), time_index[i1: i2]):
                    # if time between May0 and Apr3 (May 7 months before peak, April 27 months after peak), set to 0
                    if (t1.year in list_eel and t1.month in list(range(5, 13))) or (t1.year - 1 in list_eel) or (
                            t1.year - 2 in list_eel) or (t1.year - 3 in list_eel and t1.month in list(range(1, 5))):
                        arr_t[i1] = 0
            arr_o.append(arr_t)
        arr_o = numpy__array(arr_o)
        # recreate array
        year = int(xarray_base.get_time_bounds(dict_i)[0].split("-")[0])
        coordinates = {
            "year": numpy__array(list(range(year, year + arr_o.shape[0]))),
            "month": numpy__array(list(range(arr_o.shape[1])))}
        dim_name = xarray_base.convert_cf_dim_key(dict_i, "T")
        arr_o = wrapper_tools.recreate_array(
            arr_o, dict_i, axis_added=[0, 1], coords_added=coordinates, data_var=tuple_k[0], data_var_o=tuple_k[0],
            dim_added=["year", "month"], dim_removed=[dim_name])
        # mask where data = 0 (not defined)
        arr_o = arr_o.where(arr_o.to_array() != 0)
        # save values
        put_in_dict(dict_o, arr_o, *tuple_k)
        # remove relevant keys from the tuples of keys
        tuple_k, tuple_k_last = tuple_for_dict(tuple_k, tuple_k_last)
    return dict_o, tuple_k, tuple_k_last


def season_mean(
        dict_i: Union[dict, array_wrapper],
        season: Literal["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"],
        dict_o: dict = None,
        tuple_k: tuple = None,
        tuple_k_last: tuple = None,
        **kwargs) -> (dict, tuple, tuple):
    # set None input to it's default value
    dict_o = none_to_default(dict_o, {})
    tuple_k = none_to_default(tuple_k, ())
    tuple_k_last = none_to_default(tuple_k_last, ())
    # loop through nested levels
    if isinstance(dict_i, dict) is True:
        list_keys = sorted(list(dict_i.keys()), key=lambda v: v.lower())
        for k in list_keys:
            dict_o, tuple_k, tuple_k_last = season_mean(dict_i[k], season, dict_o=dict_o, tuple_k=tuple_k + (k,),
                                                        tuple_k_last=tuple_k_last + (list_keys[-1],), **kwargs)
    else:
        # compute month length
        month_length = xarray_base.time_weights(dict_i)
        # mask month_length where input data is not available
        month_length = month_length.where(dict_i.notnull())
        # input times month_length
        ds_o = dict_i * month_length
        # compute rolling (moving) 3 months window sum of ds_o
        ds_o = ds_o.rolling(center=True, time=3).sum(keep_attrs=True)
        # divide by rolling (moving) 3 months window sum of month_length
        ds_o /= month_length.rolling(center=True, time=3).sum()
        # add input attributes
        ds_o.attrs.update(**dict_i.attrs)
        # get season
        ds_o = xarray_base.get_season(ds_o, season)
        # save values
        put_in_dict(dict_o, ds_o, *tuple_k)
        # remove relevant keys from the tuples of keys
        tuple_k, tuple_k_last = tuple_for_dict(tuple_k, tuple_k_last)
    # get numpy array
    return dict_o, tuple_k, tuple_k_last


def stat_1_arr(
        dict_i: Union[dict, array_wrapper],
        stat: str,
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[Hashable], list[str], tuple[Hashable], tuple[str], None] = None,
        dict_o: dict = None,
        tuple_k: tuple = None,
        tuple_k_last: tuple = None,
        **kwargs) -> (dict, tuple, tuple):
    # set None input to it's default value
    dict_o = none_to_default(dict_o, {})
    tuple_k = none_to_default(tuple_k, ())
    tuple_k_last = none_to_default(tuple_k_last, ())
    # loop through nested levels
    if isinstance(dict_i, dict) is True:
        list_keys = sorted(list(dict_i.keys()), key=lambda v: v.lower())
        for k in list_keys:
            dict_o, tuple_k, tuple_k_last = stat_1_arr(
                dict_i[k], stat, axis=axis, dim=dim, dict_o=dict_o, tuple_k=tuple_k + (k,),
                tuple_k_last=tuple_k_last + (list_keys[-1],), **kwargs)
    else:
        # compute correlation
        ds_o = wrapper_tools.statistic(dict_i, axis=axis, dim=dim, data_var_i1=tuple_k[0], stat=stat, **kwargs)
        # save values
        put_in_dict(dict_o, ds_o, *tuple_k)
        # remove relevant keys from the tuples of keys
        tuple_k, tuple_k_last = tuple_for_dict(tuple_k, tuple_k_last)
    return dict_o, tuple_k, tuple_k_last


def stat_2_arr(
        dict_x: Union[dict, array_wrapper],
        dict_y: Union[dict, array_wrapper],
        stat: str,
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[Hashable], list[str], tuple[Hashable], tuple[str], None] = None,
        dict_o: dict = None,
        tuple_k: tuple = None,
        tuple_k_last: tuple = None,
        **kwargs) -> (dict, tuple, tuple):
    # set None input to it's default value
    dict_o = none_to_default(dict_o, {})
    tuple_k = none_to_default(tuple_k, ())
    tuple_k_last = none_to_default(tuple_k_last, ())
    # loop through nested levels
    if isinstance(dict_x, dict) is True:
        list_keys = sorted(list(dict_x.keys()), key=lambda v: v.lower())
        for k in list_keys:
            dict_o, tuple_k, tuple_k_last = stat_2_arr(
                dict_x[k], dict_y[k], stat, axis=axis, dim=dim, dict_o=dict_o, tuple_k=tuple_k + (k,),
                tuple_k_last=tuple_k_last + (list_keys[-1],), **kwargs)
    else:
        # compute correlation
        ds_o = wrapper_tools.statistic(dict_x, arr_i2=dict_y, axis=axis, dim=dim, data_var_i1=tuple_k[0], stat=stat,
                                       **kwargs)
        # save values
        put_in_dict(dict_o, ds_o, *tuple_k)
        # remove relevant keys from the tuples of keys
        tuple_k, tuple_k_last = tuple_for_dict(tuple_k, tuple_k_last)
    return dict_o, tuple_k, tuple_k_last
# ---------------------------------------------------------------------------------------------------------------------#
