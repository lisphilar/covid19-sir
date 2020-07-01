#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd


class Word(object):
    """
    Word definition.
    """
    # Variables of SIR-like model
    N = "Population"
    S = "Susceptible"
    C = "Confirmed"
    CI = "Infected"
    F = "Fatal"
    R = "Recovered"
    FR = "Fatal or Recovered"
    V = "Vaccinated"
    E = "Exposed"
    W = "Waiting"
    # Column names
    DATE = "Date"
    START = "Start"
    END = "End"
    T = "Elapsed"
    TS = "t"
    TAU = "tau"
    COUNTRY = "Country"
    ISO3 = "ISO3"
    PROVINCE = "Province"
    AREA_COLUMNS = [COUNTRY, PROVINCE]
    STR_COLUMNS = [DATE, *AREA_COLUMNS]
    COLUMNS = [*STR_COLUMNS, C, CI, F, R]
    NLOC_COLUMNS = [DATE, C, CI, F, R]
    VALUE_COLUMNS = [C, CI, F, R]
    FIG_COLUMNS = [CI, F, R, FR, V, E, W]
    # Date format: 22Jan2020 etc.
    DATE_FORMAT = "%d%b%Y"
    # Separator of country and province
    SEP = "/"
    # EDA
    RATE_COLUMNS = [
        "Fatal per Confirmed",
        "Recovered per Confirmed",
        "Fatal per (Fatal or Recovered)"
    ]
    # Optimization
    A = "_actual"
    P = "_predicted"
    # Phase name
    SUFFIX_DICT = defaultdict(lambda: "th")
    SUFFIX_DICT.update({1: "st", 2: "nd", 3: "rd"})
    TENSE = "Type"
    PAST = "Past"
    FUTURE = "Future"
    INITIAL = "Initial"
    ODE = "ODE"
    RT = "Rt"
    # Scenario analysis
    PHASE = "Phase"
    SERIES = "Scenario"
    MAIN = "Main"
    # Flag
    UNKNOWN = "-"

    @classmethod
    def num2str(cls, num):
        """
        Convert numbers to 1st, 2nd etc.

        Args:
        @num (int): number

        Returns:
            (str)
        """
        if not isinstance(num, int):
            raise TypeError("@num must be an integer.")
        q, mod = divmod(num, 10)
        suffix = "th" if q == 1 else cls.SUFFIX_DICT[mod]
        return f"{num}{suffix}"

    @staticmethod
    def negative_exp(x, a, b):
        """
        Negative exponential function f(x)=A exp(-Bx).

        Args:
            x (float): x values
            a (float): the first parameters of the function
            b (float): the second parameters of the function
        """
        return a * np.exp(-b * x)

    @classmethod
    def date_obj(cls, date_str):
        """
        Convert a string to a datetime object.

        Args:
            date_str (str): date, like 22Jan2020

        Returns:
            (datetime.datetime)
        """
        obj = datetime.strptime(date_str, cls.DATE_FORMAT)
        return obj

    @staticmethod
    def flatten(nested_list, unique=True):
        """
        Flatten the nested list.

        Args:
            nested_list (list[list[object]]): nested list
            unique (bool): if True, only unique values will remain

        Returns:
            (list[object])
        """
        flattened = sum(nested_list, list())
        if unique:
            return list(set(flattened))
        return flattened

    @staticmethod
    def validate_dataframe(target, name="df", time_index=False, columns=None):
        """
        Validate the dataframe has the columns.

        Args:
            target (pandas.DataFrame): the dataframe to validate
            name (str): argument name of the dataframe
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have
            df (pandas.DataFrame): as-is the target
        """
        df = target.copy()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"@{name} must be a instance of (pandas.DataFrame).")
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if columns is None:
            return df
        if not set(columns).issubset(set(df.columns)):
            cols_str = ', '.join(
                [col for col in columns if col not in df.columns]
            )
            raise KeyError(
                f"Expected columns were not included in {name}. {cols_str} must be included."
            )
        return df

    @staticmethod
    def validate_natural_int(target, name="number", include_zero=False):
        """
        Validate the natural (non-negative) number.
        If the value is natural number and the type was float,
        will be converted to an integer.

        Args:
            target (int or float or str): value to validate
            name (str): argument name of the value
            include_zero (bool): include 0 or not

        Returns:
            (int): as-is the target
        """
        s = f"@{name} must be a natural number, but {target} was applied"
        try:
            number = int(target)
        except TypeError:
            raise TypeError(f"{s} and not converted to integer.")
        if number != target:
            raise ValueError(f"{s}. |{target} - {number}| > 0")
        min_value = 0 if include_zero else 1
        if number < min_value:
            raise ValueError(f"{s}. This value is under {min_value}")
        return number

    @staticmethod
    def validate_subclass(target, parent, name="target"):
        """
        Validate the target is a subclass of the parent class.

        Args:
            target (object): target to validate
            parent (object): parent class
            name (str): argument name of the target

        Returns:
            (int): as-is the target
        """
        s = f"@{name} must be an sub class of {type(parent)}, but {type(target)} was applied."
        if not issubclass(target, parent):
            raise TypeError(s)
        return target

    @staticmethod
    def validate_instance(target, class_obj, name="target"):
        """
        Validate the target is a instance of the class object.

        Args:
            target (instance): target to validate
            parent (class): class object
            name (str): argument name of the target

        Returns:
            (instance): as-is target
        """
        s = f"@{name} must be an instance of {type(class_obj)}, but {type(target)} was applied."
        if not isinstance(target, class_obj):
            raise TypeError(s)
        return target

    @classmethod
    def divisors(cls, value):
        """
        Return the list of divisors of the value.

        Args:
            value (int): target value

        Returns:
            (list[int]): the list of divisors
        """
        value = cls.validate_natural_int(value)
        divisors = [
            i for i in range(1, value + 1) if value % i == 0
        ]
        return divisors

    @classmethod
    def to_date_obj(cls, date_str=None, default=None):
        """
        Convert a string to a datatime object.

        Args:
            date_str (str or None, optional): string, like 22Jan2020
            default (datetime.datetime or None, optional): default value to return

        Returns:
            (datetime.datetime or None)

        Notes:
            If @date_str is None, returns @default value
        """
        if date_str is None:
            return default
        date_obj = datetime.strptime(date_str, cls.DATE_FORMAT)
        return date_obj
