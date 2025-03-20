# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Basic tools built over xarray
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy
from dataclasses import dataclass
from typing import Hashable, Literal, Union
# numpy
from numpy import array as numpy__array
from numpy import ndarray as numpy__ndarray
from numpy import where as numpy__where

# local functions
from enso_ocean_dynamics.tools.default_tools import none_to_default
from enso_ocean_dynamics.wrapper import array_handler
from enso_ocean_dynamics.wrapper import statistics
from enso_ocean_dynamics.wrapper import xarray_base
from enso_ocean_dynamics.wrapper.xarray_base import array_wrapper
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Classes
# ---------------------------------------------------------------------------------------------------------------------#
@dataclass
class ValueRange:
    min: float
    max: float
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
def recreate_array(
        arr: numpy__ndarray,
        ds: array_wrapper,
        attrs_added: dict[str, str] = None,
        attrs_removed: list[str] = None,
        axis_added: Union[list[int], tuple[int], None] = None,
        coords_added: dict[str, Union[numpy__ndarray, array_wrapper]] = None,
        data_var: str = None,
        dim_added: Union[list[Hashable], list[str], tuple[Hashable], tuple[str], None] = None,
        dim_removed: Union[list[Hashable], list[str], tuple[Hashable], tuple[str], None] = None,
        **kwargs) -> array_wrapper:
    """
    Recreate output xarray.DataArray from input xarray.DataArray.

    Input:
    ------
    :param arr: numpy.ndarray
        Array derived from ‘ds‘ (e.g., a statistic was computed) that was transformed into a numpy.ndarray in the
        process
    :param ds: xarray.DataArray
        Original xarray.DataArray from which ‘arr‘ is derived
    :param attrs_added: dict[str, str] or None, optional
        Variable attributes to add to the DataArray; e.g, attrs_added = {"attr_name": "new attribute"}.
        Default is None (no attribute added)
    :param attrs_removed: list[str] or None, optional
        Variable attributes to remove from the DataArray; e.g, attrs_removed = ["attr_name"].
        Default is None (no attribute removed)
    :param axis_added: list[int] or tuple[int] or None
        Position(s) of added dimension(s) (if any); e.g, axis_added = [0] or axis_added = [0, 1].
        If given, both ‘axis_added‘ and ‘dim_added‘ must be provided.
        Default is None (no dimension has been added)
    :param coords_added: dict[str, numpy.ndarray or xarray.DataArray] or None, optional
        Coordinates (tick labels) to use for indexing along each dimension.
        If ‘dim_added‘ is not in ‘coords_added‘, coordinates will be a sequence of numbers.
        Default is None
    :param data_var: str, optional
        Name of the output data variable.
        Default is ""
    :param dim_added: list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Dimension name(s) that has been added from ‘ds‘ to ‘arr‘ (if any);
        e.g, dim_added = ["x"] or dim_added = ["x", "y"].
        If given, both ‘axis_added‘ and ‘dim_added‘ must be provided.
        Default is None (no dimension has been added)
    :param dim_removed: list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Dimension name(s) that has been removed from ‘ds‘ to ‘arr‘ (e.g., to compute a statistic);
        e.g., dim_removed = ["x"] or dim_removed = ["x", "y"].
        Default is None (no dimension was removed)

    Output:
    -------
    :return: xarray.DataArray
        Input ‘arr‘ wrapped in a xarray.DataArray.
    """
    dim_removed = none_to_default(dim_removed, [])
    # list dimensions
    dimensions = list(ds.dims)
    # delete removed dimension(s)
    for k in dim_removed:
        if isinstance(k, str) is True and k in dimensions:
            dimensions.remove(k)
    # get coordinates corresponding to dimensions
    coordinates = dict((k, ds[k]) for k in dimensions)
    # add given dimension(s)
    if isinstance(axis_added, (list, tuple)) is True and isinstance(dim_added, (list, tuple)) is True and \
            len(axis_added) == len(dim_added):
        for k1, k2 in zip(axis_added, dim_added):
            # add given dimension at given position
            dimensions.insert(k1, k2)
            # add coordinates in dictionary
            if isinstance(coords_added, dict) is True and k2 in coords_added.keys():
                coordinates[k2] = coords_added[k2]
            else:
                coordinates[k2] = numpy__array(list(range(arr.shape[k1])))
    # get input attributes
    attributes = ds.attrs
    # remove attributes
    if isinstance(attrs_removed, list) is True:
        for k in attrs_removed:
            if k in list(attributes.keys()):
                del attributes[k]
    # add given attribute(s)
    if isinstance(attrs_added, dict) is True:
        attributes.update(attrs_added)
    attributes = dict((k, attributes[k]) for k in sorted(list(attributes.keys()), key=lambda v: v.lower()))
    # numpy.ndarray to xarray.DataArray
    ds_o = array_wrapper(attrs=attributes, coords=coordinates, data=arr, dims=dimensions, name=data_var)
    return ds_o


def statistic_resampled(
        arr_i1: array_wrapper,
        arr_i2: Union[array_wrapper, None] = None,
        axis: Union[int, None] = 0,
        data_var_i1: str = None,
        data_var_i2: str = None,
        dim: Union[Hashable, str, None] = None,
        method: Literal["identical", "random"] = "identical",
        pseudo_sample_size: int = 10000,
        random_eel: bool = True,
        sample_size: int = None,
        stat: str = "mea",
        **kwargs) -> array_wrapper:
    # numpy arrays are manipulated in this function
    arr_t1 = arr_i1.to_numpy()
    arr_t2 = None
    if isinstance(arr_i2, array_wrapper) is True:
        arr_t2 = arr_i2.to_numpy()
    # sample size
    if isinstance(arr_t1, numpy__ndarray) is True and sample_size is None:
        sample_size = arr_t1.shape[axis]
    # set axis and dimension (if axis is given find dimension name and conversely)
    axis, dim = xarray_base.get_axis_dim(arr_i1, axis=axis, dim=dim)
    # create random indices
    idx1 = array_handler.resampling_indices(arr_t1.shape[axis], pseudo_sample_size, sample_size)
    # transpose arrays (put axis 'axis' in first position)
    arr_t1 = array_handler.transpose_axis_to_first_dimension(arr_t1, axis)
    # select random indices
    sample_1 = arr_t1[idx1]
    # do the same for the second array
    sample_2 = None
    if isinstance(arr_i2, array_wrapper) is True:
        if method == "random":
            # get random indices for arr_t2, not the same as for arr_t1
            idx2 = array_handler.resampling_indices(arr_t2.shape[axis], pseudo_sample_size, sample_size)
            if random_eel is False:
                # get first year
                dim_arr = arr_i1.to_numpy()
                y0 = dim_arr[0]
                # if an extreme El Niño was selected in idx1, keep it
                list_eel = [1972, 1982, 1997, 2015]
                for yy in list_eel:
                    idx2 = numpy__where(idx1 == yy - y0, idx1, idx2)
        else:
            idx2 = copy__deepcopy(idx1)
        arr_t2 = array_handler.transpose_axis_to_first_dimension(arr_t2, axis)
        sample_2 = arr_t2[idx2]
    # recreate DataArray
    dim_removed = [dim] if isinstance(dim, (Hashable, str)) is True else copy__deepcopy(dim)
    coordinates = {
        "resample": numpy__array(list(range(pseudo_sample_size))),
        dim: numpy__array(list(range(sample_size)))}
    sample_1 = recreate_array(sample_1, arr_i1, axis_added=[0, 1], coords_added=coordinates, data_var=data_var_i1,
                              dim_added=["resample", dim], dim_removed=dim_removed)
    if isinstance(arr_i2, array_wrapper) is True:
        sample_2 = recreate_array(sample_2, arr_i2, axis_added=[0, 1], coords_added=coordinates, data_var=data_var_i2,
                                  dim_added=["resample", dim], dim_removed=dim_removed)
    # compute statistic
    if sample_2 is not None:
        kwargs.update({"arr_i2": sample_2})
    return statistic(sample_1, dim=dim, stat=stat, **kwargs)


def statistic(
        arr_i1: Union[array_wrapper, list, numpy__ndarray],
        arr_i2: Union[array_wrapper, list, numpy__ndarray] = None,
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[Hashable], list[str], tuple[Hashable], tuple[str], None] = None,
        stat: str = "mea",
        **kwargs) -> Union[float, array_wrapper, array_wrapper, numpy__ndarray]:
    """
    Compute given statistic on input numpy.ndarray or xarray.DataArray

    Input:
    ------
    :param arr_i1: array_like
    :param arr_i2: array_like, optional
        Needed for two array statistics:
            correlation (statistic = 'cor'),
        Default is None
    :param axis: int or list[int] or tuple[int] or or None, optional
        Axis(es) (numpy) along which to compute statistic; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute statistic of the flattened array).
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute statistic; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute statistic of the flattened array)
    :param stat: str, optional
        Nickname of the desired statistic (see statistics)
    **kwargs: dict
        The following options are available for statistics using standard deviation or variance (see specific function
        for details):
            'ddof': divisor used in the calculation is N - ddof, where N represents the number of elements
        The following options are available for statistics using kurtosis or skewness (see specific function for
        details):
            'bias': False to corrected for statistical bias
        The following options are available for percentile (statistic = 'per', see stat_percentile for details):
            'per': Percentile(s) at which to extract score. Values should be in range [0, 100]
        The following options are sometimes available for xarray (see specific function for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one
            'method' (stat_percentile): Interpolation method to use when the desired quantile lies between two data
                points
            'weights' (stat_correlation_pearson): array of weights

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        The desired statistical value(s).
    """
    arr_i1, arr_i2 = array_handler.do_list_to_array(arr_i1), array_handler.do_list_to_array(arr_i2)
    # set axis and dimension (if axis is given find dimension name and conversely)
    if isinstance(arr_i1, array_wrapper) is True:
        axis, dim = xarray_base.get_axis_dim(arr_i1, axis=axis, dim=dim)
    elif isinstance(arr_i2, array_wrapper) is True:
        axis, dim = xarray_base.get_axis_dim(arr_i2, axis=axis, dim=dim)
    # known statistics
    stat_1_arr = {
        "kur": statistics.kurtosis, "mea": statistics.mean, "med": statistics.median, "per": statistics.percentile,
        "ske": statistics.skewness, "std": statistics.standard_deviation, "tin": statistics.student_confidence_interval,
        "var": statistics.variance}
    stat_2_arr = {
        "cor": statistics.correlation_pearson}
    # compute statistic
    if stat in list(stat_1_arr.keys()):
        arr_o = stat_1_arr[stat](arr_i1, axis=axis, dim=dim, **kwargs)
    else:
        arr_o = stat_2_arr[stat](arr_i1, arr_i2, axis=axis, dim=dim, **kwargs)
    return arr_o
# ---------------------------------------------------------------------------------------------------------------------#
