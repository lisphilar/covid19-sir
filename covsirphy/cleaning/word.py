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
    STR_COLUMNS = [DATE, COUNTRY, PROVINCE]
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
        @num <int>: number
        @return <str>
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
        @x <float>: x values
        parameters of the function
            - a <float>
            - b <float>
        """
        return a * np.exp(-b * x)

    @classmethod
    def date_obj(cls, date_str):
        """
        Convert a string to a datetime object.
        @date_str <str>: date, like 22Jan2020
        @return <datetime.datetime>
        """
        obj = datetime.strptime(date_str, cls.DATE_FORMAT)
        return obj

    @staticmethod
    def flatten(nested_list, unique=True):
        """
        Flatten the nested list.
        @nested_list <list[list[object]]>: nested list
        @unique <bool>: if True, only unique values will remain
        @return <list[object]>
        """
        flattened = sum(nested_list, list())
        if unique:
            return list(set(flattened))
        return flattened

    @staticmethod
    def validate_dataframe(target, name="df", time_index=False, columns=None):
        """
        Validate the dataframe has the columns.
        @target <pd.DataFrame>: the dataframe to validate
        @name <str>: argument name of the dataframe
        @time_index <bool>: if True, the dataframe must has DatetimeIndex
        @columns <list[str]/None>: the columns the dataframe must have
        @df <pd.DataFrame>: as-is the target
        """
        df = target.copy()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"@{name} must be a instance of <pd.DataFrame>.")
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if columns is None:
            return df
        if not set(columns).issubset(set(df.columns)):
            cols_str = ', '.join(
                [col for col in columns if col not in df.columns]
            )
            raise KeyError(f"@{name} must have {cols_str}, but not included.")
        return df

    @staticmethod
    def validate_natural_int(target, name="number"):
        """
        Validate the natural (non-negative) number.
        If the value is natural number and the type was float,
         will be converted to an integer.
        @target <int/float/str>: value to validate
        @name <str>: argument name of the value
        @return <int>: as-is the target
        """
        s = f"@{name} must be a natural number, but {target} was applied"
        try:
            number = int(target)
        except TypeError:
            raise TypeError(f"{s} and not converted to integer.")
        if number != target:
            raise ValueError(f"{s}. |{target} - {number}| > 0")
        if number < 1:
            raise ValueError(f"{s}. This value is under 1")
        return number

    @staticmethod
    def validate_subclass(target, parent, name="target"):
        """
        Validate the target is a subclass of the parent class.
        @target <object>: target to validate
        @parent <object>: parent class
        @name <str>: argument name of the target
        @return <int>: as-is the target
        """
        s = f"@{name} must be an sub class of {type(parent)}, but {type(target)} was applied."
        if not issubclass(target, parent):
            raise TypeError(s)
        return target

    @staticmethod
    def validate_instance(target, class_obj, name="target"):
        """
        Validate the target is a instance of the class object.
        @target <instance>: target to validate
        @parent <class>: class object
        @name <str>: argument name of the target
        @return <instance>: as-is target
        """
        s = f"@{name} must be an instance of {type(class_obj)}, but {type(target)} was applied."
        if not isinstance(target, class_obj):
            raise TypeError(s)
        return target

    @classmethod
    def divisors(cls, value):
        """
        Return the list of divisors of the value.
        @value <int>: target value
        @return <list[int]>: the list of divisors
        """
        value = cls.validate_natural_int(value)
        divisors = [
            i for i in range(1, value + 1) if value % i == 0
        ]
        return divisors
