# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Functions to handle arrays
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy
from typing import Any
# numpy
from numpy import array as numpy__array
from numpy import ndarray as numpy__ndarray
from numpy.random import randint as numpy__random__randint
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
def do_list_to_array(input_value: Any) -> Any:
    """
    Make a numpy.ndarray from a list (if a list is given).

    Input:
    ------
    :param input_value: Any
        Anything

    Output:
    -------
    :return: Any
        If the input value is a list, returns a numpy.ndarray of this list, else returns input unchanged
    """
    if isinstance(input_value, list) is True:
        input_value = numpy__array(input_value)
    return input_value


def resampling_indices(
        population_size: int,
        nbr_pseudo_samples: int,
        sample_size: int) -> numpy__ndarray:
    """
    Create array of indices for resampling

    Input:
    ------
    :param population_size: int
        Number of values in the population; e.g., population_size = 100
    :param nbr_pseudo_samples: int
        Number of samples to generate; e.g., nbr_pseudo_samples = 1000
        If method = 'combination', this is the maximum number of combinations to use
    :param sample_size: int
        Number of values in each sample; e.g., sample_size = 10

    Output:
    -------
    :return: ndarray
        Indices to select from the population
    """
    # create resamples
    return numpy__random__randint(0, population_size, (nbr_pseudo_samples, sample_size))


def transpose_axis_to_first_dimension(input_array: numpy__ndarray, axis: int) -> numpy__ndarray:
    """
    Transpose given axis to the first dimension.
    E.g., input_array.shape = (2, 3, 4) and axis = 1, output_array.shape = (3, 2, 4)

    Input:
    ------
    :param input_array: numpy.ndarray
        numpy.ndarray to transpose
    :param axis: int
        axis number to transpose at the first dimension

    Output:
    -------
    :return: numpy.ndarray
        Transposed array, with given axis number now at the first dimension
    """
    # transpose arrays (put axis 'axis' in first position)
    axis_p = copy__deepcopy(axis) if axis >= 0 else len(input_array.shape) + axis
    ordered_axes_tuple = tuple([axis_p] + [k for k in range(len(input_array.shape)) if k != axis_p])
    return input_array.transpose(ordered_axes_tuple)


def transpose_axis_to_last_dimension(input_array: numpy__ndarray, axis: int) -> numpy__ndarray:
    """
    Transpose given axis to the last dimension.
    E.g., input_array.shape = (2, 3, 4) and axis = 1, output_array.shape = (2, 4, 3)

    Input:
    ------
    :param input_array: numpy.ndarray
        numpy.ndarray to transpose
    :param axis: int
        axis number to transpose at the last dimension

    Output:
    -------
    :return: numpy.ndarray
        Transposed array, with given axis number now at the last dimension
    """
    # transpose arrays (put axis 'axis' in first position)
    axis_p = copy__deepcopy(axis) if axis >= 0 else len(input_array.shape) + axis
    ordered_axes_tuple = tuple([k for k in range(len(input_array.shape)) if k != axis_p] + [axis_p])
    return input_array.transpose(ordered_axes_tuple)
# ---------------------------------------------------------------------------------------------------------------------#
