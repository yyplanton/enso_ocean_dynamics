# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Functions to compute elementary statistics
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy, deepcopy
from typing import Hashable, Union, Literal
# numpy
from numpy import array as numpy__array
from numpy import clip as numpy__clip
from numpy import mean as numpy__mean
from numpy import median as numpy__median
from numpy import ndarray as numpy__ndarray
from numpy import ndindex as numpy__ndindex
from numpy import std as numpy__std
from numpy import var as numpy__var
from numpy import zeros as numpy__zeros
# scipy
from scipy.stats import kurtosis as scipy__stats__kurtosis
from scipy.stats import linregress as scipy__stats__linregress
from scipy.stats import scoreatpercentile as scipy__stats__scoreatpercentile
from scipy.stats import skew as scipy__stats__skew
from scipy.stats import t as scipy__stats__t

# local package
from enso_ocean_dynamics.tools.default_tools import none_to_default
from enso_ocean_dynamics.wrapper.array_handler import transpose_axis_to_last_dimension
from enso_ocean_dynamics.wrapper import xarray_base
from enso_ocean_dynamics.wrapper.xarray_base import array_wrapper
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
def correlation_pearson(
        arr_i1: Union[array_wrapper, numpy__ndarray],
        arr_i2: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        data_var_i1: str = None,
        data_var_i2: str = None,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the Pearson correlation between the two given arrays using my own code (neither numpy nor scipy allow to
    compute the Pearson correlation along a given axis)

    Input:
    ------
    :param arr_i1: array_like
    :param arr_i2: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute Pearson correlation; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute Pearson correlation of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute Pearson correlation; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute Pearson correlation of the flattened array)
    :param data_var_i1: str, optional
        Data variable in ‘arr_i1‘; e.g., data_var_i1 = "ts".
        Default is None
    :param data_var_i2: str, optional
        Data variable in ‘arr_i2‘; e.g., data_var_i2 = "ts".
        Default is None
    **kwargs: dict
        Additional keyword arguments passed on to xarray.
        The following options are available (see xarray_wrapper.xarray_corr for details):
            'weights': array of weights.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        The Pearson correlation coefficient(s)
    """
    if isinstance(arr_i1, array_wrapper) is True:
        # arguments for corr
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["weights"])
        # compute xarray's corr
        arr_o = xarray_base.correlation(
            arr_i1, arr_i2, dim=dim, data_var_a=data_var_i1, data_var_b=data_var_i2, **xarray_kwargs)
    else:
        # input to numpy.ndarray
        arr_t1 = numpy__array(arr_i1) if isinstance(arr_i1, numpy__ndarray) is False else copy__deepcopy(arr_i1)
        arr_t2 = numpy__array(arr_i2) if isinstance(arr_i2, numpy__ndarray) is False else copy__deepcopy(arr_i2)
        # transpose or flatten arrays
        if axis is None:
            arr_t1, arr_t2 = arr_t1.flatten(), arr_t2.flatten()
        else:
            # transpose arrays (put axis 'axis' in first position)
            axis_p = copy__deepcopy(axis) if axis >= 0 else len(arr_i1.shape) + axis
            ordered_axes_tuple = tuple([axis_p] + [k for k in range(len(arr_t1.shape)) if k != axis_p])
            arr_t1, arr_t2 = arr_t1.transpose(ordered_axes_tuple), arr_t2.transpose(ordered_axes_tuple)
        # remove mean
        xm, ym = arr_t1 - arr_t1.mean(axis=0), arr_t2 - arr_t2.mean(axis=0)
        # covariance
        covariance = xm * ym
        covariance = covariance.sum(axis=0) / covariance.shape[0]
        # correlation
        arr_o = numpy__clip(covariance / (xm.std(axis=0) * ym.std(axis=0)), -1.0, 1.0)
    return arr_o


def kurtosis(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        bias: bool = True,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the kurtosis (Fisher) along the given axis using scipy.
    g2 (G2 if bias is False) as in Wright and Herrington 2011, https://doi.org/10.3758/s13428-010-0044-x

    Input:
    ------
    :param arr_i: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute kurtosis; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute kurtosis of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute kurtosis; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute kurtosis of the flattened array)
    :param bias: bool, optional
        If False, then the calculations are corrected for statistical bias.
    **kwargs: dict
        Additional keyword arguments for xarray.
        The following options are available (see xarray_wrapper.xarray_base.reduce for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        Array containing the kurtosis values.
    """
    # compute kurtosis
    if isinstance(arr_i, array_wrapper) is True:
        # arguments for reduce
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["keep_attrs"])
        # compute scipy.stats.kurtosis using xarray's reduce
        arr_o = arr_i.reduce(scipy__stats__kurtosis, dim=dim, bias=bias, **xarray_kwargs)
    else:
        arr_o = scipy__stats__kurtosis(numpy__array(arr_i), axis=axis, bias=bias)
    return arr_o


def linear_regression(arr_i1, arr_i2, **kwargs) -> dict[str, float]:
    """
    Compute the linear least-squares regression between the two given arrays using scipy

    Input:
    ------
    :param arr_i1: array_like
    :param arr_i2: array_like
    **kwargs: dict

    Output:
    -------
    :return slope: float
        Slope of the regression line
    :return intercept: float
        Intercept of the regression line
    :return correlation: float
        The Pearson correlation coefficient
    :return p_value: float
        The p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald Test with
        t-distribution of the test statistic
    """
    slope, intercept, correlation, p_value, _ = scipy__stats__linregress(arr_i1, arr_i2)
    return {"i": intercept, "p": p_value, "r": correlation, "s": slope}


def mean(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the mean along the given axis using numpy or xarray.

    Input:
    ------
    :param arr_i: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute mean; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute mean of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute mean; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute mean of the flattened array)
    **kwargs: dict
        Additional keyword arguments for xarray.
        The following options are available (see xarray_wrapper.xarray_mean for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        Array containing the mean values.
    """
    if isinstance(arr_i, array_wrapper) is True:
        # arguments for mean
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["keep_attrs"])
        # compute xarray's mean
        arr_o = arr_i.mean(dim=dim, **xarray_kwargs)
    else:
        arr_o = numpy__mean(arr_i, axis=axis)
    return arr_o


def median(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the median along the given axis using numpy or xarray.

    Input:
    ------
    :param arr_i: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute median; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute median of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute median; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute median of the flattened array)
    **kwargs: dict
        Additional keyword arguments for xarray.
        The following options are available (see xarray_wrapper.xarray_median for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        Array containing the median values.
    """
    if isinstance(arr_i, array_wrapper) is True:
        # arguments for median
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["keep_attrs"])
        # compute xarray's median
        arr_o = arr_i.median(dim=dim, **xarray_kwargs)
    else:
        arr_o = numpy__median(arr_i, axis=axis)
    return arr_o


def percentile(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        per: Union[float, list[float], tuple[float] ]= None,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the per th percentile(s) of the data along the specified dimension using scipy or xarray.

    Input:
    ------
    :param arr_i: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute percentile; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute percentile of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute percentile; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute percentile of the flattened array)
    :param per: float, list[float], tuple[float], optional
        Percentile(s) at which to extract score. Values should be in range [0, 100].
        Default is [0, 100] (full range returned)
    **kwargs: dict
        Additional keyword arguments for xarray.
        The following options are available (see xarray_wrapper.xarray_quantile for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one.
            'method': Interpolation method to use when the desired quantile lies between two data points.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        Score at percentile(s).
    """
    per = none_to_default(per, [0, 100])
    # compute percentile
    if isinstance(arr_i, array_wrapper) is True:
        # arguments for quantile
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["keep_attrs", "method"])
        # xarray quantile requires quantiles not percentiles
        quantile = per / 100 if isinstance(per, (float, int)) is True else [k / 100 for k in per]
        # compute xarray's quantile
        arr_o = arr_i.quantile(quantile, dim=dim, **xarray_kwargs)
    else:
        arr_o = scipy__stats__scoreatpercentile(arr_i, per, axis=axis)
    return arr_o


def skewness(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        bias: bool = True,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the skewness (Fisher-Pearson) along the given axis using scipy.
    g1 (G1 if bias is False) as in Wright and Herrington 2011, https://doi.org/10.3758/s13428-010-0044-x

    Input:
    ------
    :param arr_i: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute skewness; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute skewness of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute skewness; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute skewness of the flattened array)
    :param bias: bool, optional
        If False, then the calculations are corrected for statistical bias.
    **kwargs: dict
        Additional keyword arguments for xarray.
        The following options are available (see xarray_wrapper.xarray_base.reduce for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        Array containing the skewness values.
    """
    # compute skewness
    if isinstance(arr_i, array_wrapper) is True:
        # arguments for reduce
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["keep_attrs"])
        # compute scipy.stats.skew using xarray's reduce
        arr_o = arr_i.reduce(scipy__stats__skew, dim=dim, bias=bias, **xarray_kwargs)
    else:
        arr_o = scipy__stats__skew(numpy__array(arr_i), axis=axis, bias=bias)
    return arr_o


def standard_deviation(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        ddof: int = 0,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the standard deviation along the given axis using numpy or xarray.

    Input:
    ------
    :param arr_i: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute standard_deviation; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute standard_deviation of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute standard_deviation; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute standard_deviation of the flattened array)
    :param ddof: int, optional
        Delta Degrees of Freedom: the divisor used in the calculation is N - ddof, where N represents the number of
        elements.
        Default is 0
    **kwargs: dict
        Additional keyword arguments for xarray.
        The following options are available (see xarray_wrapper.xarray_std for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        Array containing the standard deviation values.
    """
    # compute standard deviation
    if isinstance(arr_i, array_wrapper) is True:
        # arguments for std
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["keep_attrs"])
        # compute xarray's std
        arr_o = arr_i.standard_deviation(ddof=ddof, dim=dim, **xarray_kwargs)
    else:
        arr_o = numpy__std(arr_i, axis=axis, ddof=ddof)
    return arr_o


def student_confidence_interval(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: int = None,
        dim: Union[Hashable, str] = None,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    # compute standard deviation
    arr_t = deepcopy(arr_i)
    if isinstance(arr_i, array_wrapper) is True:
        arr_t = arr_i.to_numpy()
    # transpose array (put axis 'axis' in last position)
    arr_t = transpose_axis_to_last_dimension(arr_t, axis)
    # initialize output matrix
    matrix_shape = arr_t.shape[:-1]
    arr_o = numpy__zeros(matrix_shape + (2,))
    # iterate over all dimensions but given one
    for index in numpy__ndindex(matrix_shape):
        arr_o[index+ (0,)] = student_interval(arr_t[index], bound="lower", **kwargs)
        arr_o[index + (1,)] = student_interval(arr_t[index], bound="upper", **kwargs)
    # recreate array
    if isinstance(arr_i, array_wrapper) is True:
        data_var = kwargs["data_var"] if "data_var" in list(kwargs.keys()) else "unknown"
        # list dimensions
        dimensions = list(arr_i.dims)
        # delete removed dimension
        if isinstance(dim, (Hashable, str)) is True and dim in dimensions:
            dimensions.remove(dim)
        # get coordinates corresponding to dimensions
        coordinates = dict((k, arr_i[k]) for k in dimensions)
        # add given dimension at given position
        dimensions.append("interval")
        coordinates["interval"] = numpy__array(list(range(2)))
        # get input attributes
        attributes = arr_i.attrs
        attributes = dict((k, attributes[k]) for k in sorted(list(attributes.keys()), key=lambda v: v.lower()))
        # numpy.ndarray to xarray.DataArray
        arr_o =array_wrapper(attrs=attributes, coords=coordinates, data=arr_o, dims=dimensions, name=data_var)
    return arr_o


def student_interval(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: int = None,
        dim: Union[Hashable, str] = None,
        bound: Literal["lower", "upper"] = "lower",
        confidence_level: float = 95,
        **kwargs) -> float:
    # array mean
    loc = mean(arr_i, axis=axis, dim=dim, **kwargs)
    # standard deviation
    scale = variance(arr_i, axis=axis, dim=dim, ddof=1, **kwargs)
    # array size
    size = arr_i.size
    # number of standard deviations needed to obtain given significance_level
    alpha = 0.5 + confidence_level / 200
    zscore = scipy__stats__t.ppf(alpha, size - 1)
    # standard error
    se = scale ** 0.5 / (size - 1) ** 0.5
    # theoretical uncertainty of the sample mean
    arr_o = [loc - zscore * se, loc + zscore * se]
    if bound == "lower":
        arr_o = arr_o[0]
    else:
        arr_o = arr_o[1]
    return arr_o


def variance(
        arr_i: Union[array_wrapper, numpy__ndarray],
        axis: Union[int, list[int], tuple[int], None] = None,
        dim: Union[Hashable, str, list[str], list[Hashable], tuple[Hashable], tuple[str], None] = None,
        ddof: int = 0,
        **kwargs) -> Union[float, array_wrapper, numpy__ndarray]:
    """
    Compute the variance along the given axis using numpy or xarray.

    Input:
    ------
    :param arr_i: array_like
    :param axis: int or list[int] or tuple[int] or None, optional
        Axis(es) (numpy) along which to compute variance; e.g., axis=0.
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute variance of the flattened array)
    :param dim: Hashable or str or list[Hashable] or list[str] or tuple[Hashable] or tuple[str] or None, optional
        Name of dimension(s) (xarray) along which to compute variance; e.g., dim="time".
        Only one of the 'axis' and 'dim' arguments can be supplied.
        Default is None (compute variance of the flattened array)
    :param ddof: int, optional
        Delta Degrees of Freedom: the divisor used in the calculation is N - ddof, where N represents the number of
        elements.
        Default is 0
    **kwargs: dict
        Additional keyword arguments for xarray.
        The following options are available (see xarray_wrapper.xarray_var for details):
            'keep_attrs': True to copy the dataset's attributes (attrs) from the original object to the new one.

    Output:
    -------
    :return: float or xarray.DataArray or numpy.ndarray
        Array containing the variance values.
    """
    # compute variance
    if isinstance(arr_i, array_wrapper) is True:
        # arguments for var
        xarray_kwargs = dict((ii, jj) for ii, jj in kwargs.items() if ii in ["keep_attrs"])
        # compute xarray's var
        arr_o = arr_i.variance(ddof=ddof, dim=dim, **xarray_kwargs)
    else:
        arr_o = numpy__var(arr_i, axis=axis, ddof=ddof)
    return arr_o
# ---------------------------------------------------------------------------------------------------------------------#
