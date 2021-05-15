#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate, UnExpectedValueError


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
    TESTS = "Tests"
    # Vaccination
    VAC = "Vaccinations"
    V = "Vaccinated"
    V_ONCE = f"{V}_once"
    V_FULL = f"{V}_full"
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
    STEP_N = "step_n"
    Y0_DICT = "y0_dict"
    PARAM_DICT = "param_dict"
    ID = "ID"
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
    UNKNOWN = "-"
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
        num = cls._ensure_natural_int(num, include_zero=True)
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
        except ValueError:
            raise ValueError(
                f"Examples of {name} are 0th, 1st, 2nd..., but {string} was applied.")

    @staticmethod
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
    def flatten(nested_list, unique=True):
        """
        Flatten the nested list.

        Args:
            nested_list (list[list[object]]): nested list
            unique (bool): if True, only unique values will remain

        Returns:
            list[object]
        """
        flattened = sum(nested_list, list())
        if unique:
            return list(set(flattened))
        return flattened

    @staticmethod
    def _ensure_dataframe(target, name="df", time_index=False, columns=None):
        """
        Ensure the dataframe has the columns.

        Args:
            target (pandas.DataFrame): the dataframe to ensure
            name (str): argument name of the dataframe
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have

        Returns:
            pandas.DataFrame:
                Index
                    as-is
                Columns:
                    columns specified with @columns or all columns of @target (when @columns is None)
        """
        if not isinstance(target, pd.DataFrame):
            raise TypeError(f"@{name} must be a instance of (pandas.DataFrame).")
        df = target.copy()
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if columns is None:
            return df
        if not set(columns).issubset(df.columns):
            cols_str = ", ".join(col for col in columns if col not in df.columns)
            included = ", ".join(df.columns.tolist())
            s1 = f"Expected columns were not included in {name} with {included}."
            raise KeyError(f"{s1} {cols_str} must be included.")
        return df.loc[:, columns]

    @staticmethod
    def _ensure_natural_int(target, name="number", include_zero=False, none_ok=False):
        """
        Ensure a natural (non-negative) number.

        Args:
            target (int or float or str or None): value to ensure
            name (str): argument name of the value
            include_zero (bool): include 0 or not
            none_ok (bool): None value can be applied or not.

        Returns:
            int: as-is the target

        Note:
            When @target is None and @none_ok is True, None will be returned.
            If the value is a natural number and the type was float or string,
            it will be converted to an integer.
        """
        if target is None and none_ok:
            return None
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

    def _ensure_int_range(self, target, name="number", value_range=(0, None)):
        """
        Ensure the number is an integer and in the specified range.

        Args:
            target (int or float or str): value to ensure
            name (str): argument name of the value
            value_range(tuple(int or None, int or None)): value range, None means un-specified

        Returns:
            int: as-is the target
        """
        number = self._ensure_natural_int(target=target, name=name, include_zero=True, none_ok=False)
        # Minimum
        if value_range[0] is not None and number < value_range[0]:
            raise ValueError(f"{name} must be over or equal to {value_range[0]}, but {number} was applied.")
        # Maximum
        if value_range[1] is not None and number > value_range[1]:
            raise ValueError(f"{name} must be under or equal to {value_range[1]}, but {number} was applied.")
        return number

    @classmethod
    def _ensure_tau(cls, tau, accept_none=True):
        """
        Ensure that the value can be used as tau value [min].

        Args:
            tau (int or None): value to use [min] or None (when @accept_none is True)
            accept_none (bool): whether accept None or not

        Returns:
            int or None: as-is
        """
        if tau is None and accept_none:
            return None
        tau = cls._ensure_natural_int(tau, name="tau")
        if tau in set(cls.divisors(1440)):
            return tau
        raise ValueError(f"@tau must be a divisor of 1440 [min], but {tau} was applied.")

    @classmethod
    def _ensure_population(cls, population):
        """
        Ensure that the population value is valid.

        Args:
            population (int or float or str): population value

        Returns:
            int: as-is
        """
        return cls._ensure_natural_int(
            population, name="population", include_zero=False, none_ok=False
        )

    @staticmethod
    def _ensure_float(target, name="value"):
        """
        Ensure a float value.
        If the value is a float value and the type was string,
        it will be converted to a float.

        Args:
            target (float or str): value to ensure
            name (str): argument name of the value

        Returns:
            float: as-is the target
        """
        s = f"@{name} must be a float value, but {target} was applied"
        try:
            return float(target)
        except ValueError:
            raise ValueError(f"{s} and not converted to float.") from None

    @classmethod
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
        except ValueError:
            raise ValueError(f"{name} was not recognized as a date, {target} was applied.")

    @staticmethod
    def _ensure_subclass(target, parent, name="target"):
        """
        Ensure the target is a subclass of the parent class.

        Args:
            target (object): target to ensure
            parent (object): parent class
            name (str): argument name of the target

        Returns:
            int: as-is the target
        """
        s = f"@{name} must be an sub class of {type(parent)}, but {type(target)} was applied."
        if not issubclass(target, parent):
            raise TypeError(s)
        return target

    @staticmethod
    def _ensure_instance(target, class_obj, name="target"):
        """
        Ensure the target is a instance of the class object.

        Args:
            target (instance): target to ensure
            parent (class): class object
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        s = f"@{name} must be an instance of {class_obj}, but {type(target)} was applied."
        if not isinstance(target, class_obj):
            raise TypeError(s)
        return target

    @staticmethod
    def _ensure_list(target, candidates=None, name="target"):
        """
        Ensure the target is a sub-list of the candidates.

        Args:
            target (list[object]): target to ensure
            candidates (list[object] or None): list of candidates, if we have
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        if not isinstance(target, (list, tuple)):
            raise TypeError(f"@{name} must be a list or tuple, but {type(target)} was applied.")
        if candidates is None:
            return target
        # Check the target is a sub-list of candidates
        try:
            strings = [str(candidate) for candidate in candidates]
        except TypeError:
            raise TypeError(f"@candidates must be a list, but {candidates} was applied.") from None
        ok_list = [element in candidates for element in target]
        if all(ok_list):
            return target
        candidate_str = ", ".join(strings)
        raise KeyError(f"@{name} must be a sub-list of [{candidate_str}], but {target} was applied.") from None

    @classmethod
    def divisors(cls, value):
        """
        Return the list of divisors of the value.

        Args:
            value (int): target value

        Returns:
            list[int]: the list of divisors
        """
        value = cls._ensure_natural_int(value)
        return [i for i in range(1, value + 1) if value % i == 0]

    @deprecate(".date_obj()", version="2.19.1-gamma-fu5")
    @classmethod
    def date_obj(cls, date_str=None, default=None):
        """
        Convert a string to a datatime object.

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
        tau = cls._ensure_tau(tau)
        return math.ceil((end - sta) / timedelta(minutes=tau))

    @classmethod
    def _ensure_date_order(cls, previous_date, following_date, name="following_date"):
        """
        Ensure that the order of dates.

        Args:
            previous_date (str or pandas.Timestamp): previous date
            following_date (str or pandas.Timestamp): following date
            name (str): name of @following_date

        Raises:
            ValueError: @previous_date > @following_date
        """
        previous_date = cls._ensure_date(previous_date)
        following_date = cls._ensure_date(following_date)
        p_str = previous_date.strftime(cls.DATE_FORMAT)
        f_str = following_date.strftime(cls.DATE_FORMAT)
        if previous_date <= following_date:
            return None
        raise ValueError(f"@{name} must be the same as/over {p_str}, but {f_str} was applied.")

    def _ensure_selectable(self, target, candidates, name="target"):
        """
        Ensure that the target can be selectable.

        Args:
            target (object): target to check
            candidates (list[object]): list of candidates
            name (str): name of the target
        """
        self._ensure_list(candidates, name="candidates")
        if target in candidates:
            return target
        raise UnExpectedValueError(name=name, value=target, candidates=candidates)

    def _ensure_kwargs(self, arg_list, value_type, **kwargs):
        """
        Ensure the all expected arguments are specified.

        Args:
            arg_list (list[str]): list of argument names
            value_type (object): type of the values
            kwargs: keyword arguments of values

        Returns:
            dict[str, int]: dictionary of values
        """
        for param in arg_list:
            if param not in kwargs:
                raise KeyError(f"Value of {param} was not specified with keyword arguments.")
            self._ensure_instance(kwargs[param], value_type, name=f"{param} value")
        return {param: kwargs[param] for param in arg_list}


class Word(Term):
    @ deprecate(old="Word()", new="Term()")
    def __init__(self):
        super().__init__()
