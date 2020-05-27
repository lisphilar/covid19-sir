#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime
import numpy as np


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
    V = "Vacctinated"
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
    PROVINCE = "Province"
    COLUMNS = [DATE, COUNTRY, PROVINCE, C, CI, F, R]
    NLOC_COLUMNS = [DATE, C, CI, F, R]
    VALUE_COLUMNS = [C, CI, F, R]
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
    # Flag
    UNKNOWN = "-"

    @classmethod
    def num2str(cls, num):
        """
        Convert numbers to 1st, 2nd etc.
        @num <int>: number
        @return <str>
        """
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
