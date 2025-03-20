# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Simple tests on input value to return a valid value
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
from copy import deepcopy as copy__deepcopy
from typing import Any

# local functions
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
def none_to_default(input_value: Any, default_value: Any, **kwargs) -> Any:
    """
    Return default_value if input is None, else input_value

    Input:
    ------
    :param input_value: Any
        Input variable, can be anything
    :param default_value: Any
        Default value for input_value, can be anything
    **kwargs - Discarded

    Output:
    -------
    :return: Any
        If input_value is None, default_value, else input_value
    """
    # test if input is None
    if input_value is None:
        input_value = copy__deepcopy(default_value)
    return input_value


def set_default_str(input_value: str, defined_values: list[str], optional_default: str, **kwargs) -> str:
    """
    Check if the input string is a known value (i.e., it is in defined_values)
    if not return given value (optional_default)
    if given value is a known value, return the first value of defined_values
    
    Input:
    ------
    :param input_value: string
        Input value to check; e.g., input_value = 'test'
    :param defined_values: list[str]
        List of known values; e.g., defined_values = ['value1', 'value2']
    :param optional_default: string
        Value to return if input is not in defined_values; e.g., optional_default = 'value2'
    **kwargs - Discarded
    
    Output:
    -------
    :return: string
        Same as input_value or a value defined in defined_values
    """
    # test if input value is a known value, if not, change it
    if input_value not in defined_values:
        if optional_default in defined_values:
            # if the given default value is a known value, return it
            input_value = copy__deepcopy(optional_default)
        else:
            # if the given default value is not a known value, return the first defined value
            input_value = copy__deepcopy(defined_values[0])
    return input_value


def set_instance(input_value: Any, test_class: type, test_bool: bool, default_value: Any, **kwargs) -> Any:
    """
    Test instance of input and change and return it or another given value depending on the test
    
    Input:
    ------
    :param input_value: Any
        Input value to check; e.g., input_value = 'test'
    :param test_class: type
        Object class; e.g., test_class = dict
    :param test_bool: bool
        Boolean for the test (if True input_value changed if the test true, if False input_value changed if the test
        false); e.g., test_bool = True
    :param default_value: Any
        Value to return depending on the test
    **kwargs - Discarded
    
    Output:
    -------
    :return: Any
        input_value or default_value depending on the test
    """
    # return input_value or default_value depending on the test
    if isinstance(input_value, test_class) is test_bool:
        input_value = copy__deepcopy(default_value)
    return input_value
# ---------------------------------------------------------------------------------------------------------------------#
