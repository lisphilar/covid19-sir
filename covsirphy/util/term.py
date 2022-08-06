#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime, timedelta
import logging
import math
import warnings
import country_converter as coco
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.util.validator import Validator


class Term(object):
    """
    Term definition.
    """
    # Variables of SIR-derived model
    N = "Population"
    S = "Susceptible"
    C = "Confirmed"
    CI = "Infected"
    F = "Fatal"
    R = "Recovered"
    FR = "Fatal or Recovered"
    E = "Exposed"
    W = "Waiting"
    # PCR tests
    TESTS = "Tests"
    TESTS_DIFF = "Tests_diff"
    # Severity
    MODERATE = "Moderate"
    SEVERE = "Severe"
    # Vaccination
    VAC = "Vaccinations"
    V = "Vaccinated"
    V_ONCE = f"{V}_once"
    V_FULL = f"{V}_full"
    VAC_BOOSTERS = f"{VAC}_boosters"
    PRODUCT = "Product"
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
    CITY = "City"
    STEP_N = "step_n"
    Y0_DICT = "y0_dict"
    PARAM_DICT = "param_dict"
    ID = "ID"
    _PH = "Phase_ID"
    _SIRF = [S, CI, R, F]
    AREA_COLUMNS = [COUNTRY, PROVINCE]
    STR_COLUMNS = [DATE, *AREA_COLUMNS]
    COLUMNS = [*STR_COLUMNS, C, CI, F, R]
    NLOC_COLUMNS = [DATE, C, CI, F, R]
    SUB_COLUMNS = [DATE, C, CI, F, R, S]
    VALUE_COLUMNS = [C, CI, F, R]
    FIG_COLUMNS = [CI, F, R, FR, V, E, W]
    MONO_COLUMNS = [C, F, R]
    AREA_ABBR_COLS = [ISO3, *AREA_COLUMNS]
    DSIFR_COLUMNS = [DATE, S, CI, F, R]
    # Date format: 22Jan2020 etc.
    DATE_FORMAT = "%d%b%Y"
    DATE_FORMAT_DESC = "DDMmmYYYY"
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
    ACTUAL = "Actual"
    FITTED = "Fitted"
    # Phase name
    SUFFIX_DICT = defaultdict(lambda: "th")
    SUFFIX_DICT.update({1: "st", 2: "nd", 3: "rd"})
    # Summary of phases
    TENSE = "Type"
    PAST = "Past"
    FUTURE = "Future"
    INITIAL = "Initial"
    ODE = "ODE"
    RT = "Rt"
    RT_FULL = "Reproduction number"
    TRIALS = "Trials"
    RUNTIME = "Runtime"
    # Scenario analysis
    PHASE = "Phase"
    SERIES = "Scenario"
    MAIN = "Main"
    # Flag
    NA = "-"
    OTHERS = "Others"

    @classmethod
    def num2str(cls, num):
        """
        Convert numbers to 1st, 2nd etc.

        Args:
            num (int): number

        Returns:
            str
        """
        num = Validator(num, "num").int(value_range=(0, None))
        q, mod = divmod(num, 10)
        suffix = "th" if q % 10 == 1 else cls.SUFFIX_DICT[mod]
        return f"{num}{suffix}"

    @staticmethod
    def str2num(string, name="phase names"):
        """
        Convert 1st to 1 and so on.

        Args:
            string (str): like 1st, 2nd, 3rd,...
            name (str): name of the string

        Returns:
            int
        """
        try:
            return int(string[:-2])
        except ValueError as e:
            raise ValueError(
                f"Examples of {name} are 0th, 1st, 2nd..., but {string} was applied."
            ) from e

    @staticmethod
    @deprecate(".negative_exp()", version="2.25.0-mu")
    def negative_exp(x, a, b):
        """
        Negative exponential function f(x) = A exp(-Bx).

        Args:
            x (float): x values
            a (float): the first parameters of the function
            b (float): the second parameters of the function

        Returns:
            float
        """
        return a * np.exp(-b * x)

    @staticmethod
    @deprecate(".linear()", version="2.25.0-mu")
    def linear(x, a, b):
        """
        Linear function f(x) = A x + b.

        Args:
            x (float): x values
            a (float): the first parameter of the function
            b (float): the second parameter of the function

        Returns:
            float
        """
        return a * x + b

    @staticmethod
    @deprecate(".flatten()", new="Validator.sequence(flatten=True)", version="2.25.0-mu")
    def flatten(nested_list, unique=True):
        """
        Flatten the nested list.

        Args:
            nested_list (list[list[object]]): nested list
            unique (bool): if True, only unique values will remain

        Returns:
            list[object]
        """
        flattened = sum(nested_list, [])
        return list(set(flattened)) if unique else flattened

    @classmethod
    @deprecate(".divisors()", version="2.25.0-mu")
    def divisors(cls, value):
        """
        Return the list of divisors of the value.

        Args:
            value (int): target value

        Returns:
            list[int]: the list of divisors
        """
        value = Validator(value).int(value_range=(1, None))
        return [i for i in range(1, value + 1) if value % i == 0]

    @classmethod
    @deprecate(".date_obj()", version="2.19.1-gamma-fu5")
    def date_obj(cls, date_str=None, default=None):
        """
        Convert a string to a datetime object.

        Args:
            date_str (str or None, optional): string, like 22Jan2020
            default (datetime.datetime or None, optional): default value to return

        Returns:
            datetime.datetime or None

        Note:
            If @date_str is None, returns @default value
        """
        if date_str is None:
            return default
        return datetime.strptime(date_str, cls.DATE_FORMAT)

    @classmethod
    @deprecate(".date_change()", version="2.25.0-nu")
    def _ensure_date(cls, target, name="date", default=None):
        """
        Ensure the format of the string.

        Args:
            target (str or pandas.Timestamp): string to ensure
            name (str): argument name of the string
            default (pandas.Timestamp or None): default value to return

        Returns:
            pandas.Timestamp or None: as-is the target or default value
        """
        if target is None:
            return default
        if isinstance(target, pd.Timestamp):
            return target.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            return pd.to_datetime(target).replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError as e:
            raise ValueError(f"{name} was not recognized as a date, {target} was applied.") from e

    @classmethod
    @deprecate(".date_change()", version="2.25.0-mu")
    def date_change(cls, date_str, days=0):
        """
        Return @days days ago or @days days later.

        Args:
            date_str (str): today
            days (int): (negative) days ago or (positive) days later

        Returns:
            str: the date
        """
        if not isinstance(days, int):
            raise TypeError(
                f"@days must be integer, but {type(days)} was applied.")
        date = cls._ensure_date(date_str) + timedelta(days=days)
        return date.strftime(cls.DATE_FORMAT)

    @classmethod
    @deprecate(".tomorrow()", version="2.25.0-mu")
    def tomorrow(cls, date_str):
        """
        Tomorrow of the date.

        Args:
            date_str (str): today

        Returns:
            str: tomorrow
        """
        return cls.date_change(date_str, days=1)

    @classmethod
    @deprecate(".yesterday()", version="2.25.0-mu")
    def yesterday(cls, date_str):
        """
        Yesterday of the date.

        Args:
            date_str (str): today

        Returns:
            str: yesterday
        """
        return cls.date_change(date_str, days=-1)

    @classmethod
    @deprecate(".steps()", version="2.25.0-mu")
    def steps(cls, start_date, end_date, tau):
        """
        Return the number of days (round up).

        Args:
            start_date (str): start date, like 01Jan2020
            end_date (str): end date, like 01Jan2020
            tau (int): tau value [min]
        """
        sta = cls._ensure_date(start_date)
        end = cls._ensure_date(end_date)
        tau = Validator(tau, "tau").tau(default=None)
        return math.ceil((end - sta) / timedelta(minutes=tau))

    @classmethod
    def _to_iso3(cls, name):
        """Convert country name(s) to ISO3 codes.

        Args:
            name (str or list[str] or None): country name(s)

        Returns:
            list[str]: ISO3 code(s)

        Note:
            "UK" will be converted to "GBR".

        Note:
            When the country was not found or None, it will not be converted.

        Examples:
            >>> Term._to_iso3("Japan")
            ['JPN']
            >>> Term._to_iso3("UK")
            ['GBR']
            >>> Term._to_iso3("Moon")
            ['Moon']
            >>> Term._to_iso3(None)
            ['---']
            >>> Term._to_iso3(["Japan", "UK", "Moon", None])
            ['JPN', 'GBR', 'Moon', '---']
        """
        logging.basicConfig(level=logging.CRITICAL)
        warnings.simplefilter("ignore", FutureWarning)
        names = [name] if (isinstance(name, str) or name is None) else name
        code_dict = {"UK": "GBR", None: cls.NA * 3, }
        code_dict.update({elem: coco.convert(elem, to="ISO3", not_found=elem) for elem in set(names) - set(code_dict)})
        return [code_dict[elem] for elem in names]

    def _country_information(self):
        """Return the raw data of country_converter library raw data as a dataframe.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - name_short: standard or short names
                    - ISO2: ISO2 codes
                    - ISO3: ISO3 codes
                    - Continent: continent names
                    - the other columns listed in country_converter library homepage.

        Note:
            Refer to https://github.com/konstantinstadler/country_converter
        """
        return coco.CountryConverter().data


class Word(Term):
    @deprecate(old="Word()", new="Term()")
    def __init__(self):
        super().__init__()
