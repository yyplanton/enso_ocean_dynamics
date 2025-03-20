# -*- coding:UTF-8 -*-
# ---------------------------------------------------------------------------------------------------------------------#
# Functions to handle calendar (time axes)
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------#
# Import packages
# ---------------------------------------------------#
# basic python package
# numpy
from numpy import integer as numpy__integer
from numpy import ndarray as numpy__ndarray
from numpy import zeros as numpy__zeros

# pandas
from pandas import Index
# ---------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
# Functions
# ---------------------------------------------------------------------------------------------------------------------#
def _leap_year(year: int, calendar="standard") -> bool:
    """
    Determine if year is a leap year
    """
    leap = False
    if calendar in ["standard", "gregorian", "proleptic_gregorian", "julian"] and year % 4 == 0:
        leap = True
        if calendar == "proleptic_gregorian" and year % 100 == 0 and year % 400 != 0:
            leap = False
        elif calendar in ["standard", "gregorian"] and year % 100 == 0 and year % 400 != 0 and year < 1583:
            leap = False
    return leap


def get_days_per_month(time: Index, calendar: str = "standard") -> numpy__ndarray:
    """
    Return an array of days per month corresponding to the months provided in `months`
    """
    # create output array
    month_length = numpy__zeros(len(time), dtype=numpy__integer)
    # get the number of days for each month according to the calendar type
    days_per_month = {
        "noleap": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "365_day": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "standard": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "proleptic_gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "all_leap": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "366_day": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "360_day": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    }
    nbr_of_days_for_each_month = days_per_month[calendar]
    # number of days for each month of the given time axis
    for k, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[k] = nbr_of_days_for_each_month[month]
        if month == 2 and _leap_year(year, calendar=calendar):
            month_length[k] += 1
    return month_length
# ---------------------------------------------------------------------------------------------------------------------#
