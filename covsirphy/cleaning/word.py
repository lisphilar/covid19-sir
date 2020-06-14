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
    def validate_dataframe(df, name="df", time_index=False, columns=None):
        """
        Validate the dataframe has the columns.
        @df <pd.DataFrame>: the dataframe to validate
        @name <str>: argument name of the dataframe
        @time_index <bool>: if True, the dataframe must has DatetimeIndex
        @columns <list[str]/None>: the columns the dataframe must have
        @df <pd.DataFrame>: as-is the dataframe
        """
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
