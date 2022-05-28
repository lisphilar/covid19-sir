#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class ODEModel(Term):
    """Basic class of ordinary differential equation (ODE) model.

    Args:
        date_range (tuple(str, str)): start date and end date of simulation
        tau (int): tau value [min]
        initial_dict (dict of {str: int}): initial values
        param_dict (dict of {str: float}): non-dimensional parameter values
    """
    _SIFR = [Term.S, Term.CI, Term.F, Term.R]
    # Name of ODE model
    _NAME = "ODE Model"
    # Variables
    _VARIABLES = []
    # Non-dimensional parameters
    _PARAMETERS = []
    # Dimensional parameters
    _DAY_PARAMETERS = []
    # Weights of variables in parameter estimation error function
    _WEIGHTS = np.array([])
    # Variables that increases monotonically
    _VARS_INCREASE = []
    # Sample data
    _SAMPLE_DICT = {
        "initial_dict": dict.fromkeys(_VARIABLES),
        "param_dict": dict.fromkeys(_PARAMETERS)
    }

    def __init__(self, date_range, tau, initial_dict, param_dict):
        start_date, end_date = Validator(date_range, "date_range", accept_none=False).sequence(length=2)
        self._start = Validator(start_date, name="the first value of @date_range", accept_none=False).date()
        self._end = Validator(
            end_date, name="the second date of @date_range", accept_none=False).date(value_range=(self._start, None))
        self._tau = Validator(tau, "tau", accept_none=False).tau()
        self._initial_dict = Validator(initial_dict, "initial_dict", accept_none=False).dict(
            required_keys=self._VARIABLES, errors="raise")
        self._param_dict = Validator(param_dict, "param_dict", accept_none=False).dict(
            required_keys=self._PARAMETERS, errors="raise")
        # Total population
        self._population = sum(list(initial_dict.values()))

    def __str__(self):
        return self._NAME

    def __repr__(self):
        _dict = self.to_dict()
        return f"{type(self).__name__}({', '.join([f'{k}={v}' for k, v in _dict.items()])})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def to_dict(self):
        """Return conditions as a dictionary.

        Returns:
            dict of {str: object}:
                - date_range (tuple(str, str)): start date and end date of simulation
                - tau (int): tau value [min]
                - initial_dict (dict of {str: int}): initial values
                - param_dict (dict of {str: float}): non-dimensional parameter values
        """
        return {
            "date_range": (self._start.strftime(self.DATE_FORMAT), self._end.strftime(self.DATE_FORMAT)),
            "tau": self._tau,
            "initial_dict": self._initial_dict,
            "param_dict": self._param_dict,
        }

    @classmethod
    def from_sample(cls, date_range=None, tau=1440):
        """Initialize model with sample data.

        Args:
            date_range (tuple(str or None, str or None) or None): start date and end date of simulation
            tau (int): tau value [min]

        Returns:
            covsirphy.ODEModel: initialized model

        Note:
            When @date_range or the first value of @date_range is None, today when executed will be set as start date.

        Note:
            When @date_range or the second value of @date_range is None, 180 days after start date will be used as end date.
        """
        start_date, end_date = Validator(date_range, "date_range").sequence(default=(None, None), length=2)
        start = Validator(start_date, name="the first value of @date_range").date(default=datetime.now())
        end = Validator(
            end_date, name="the second date of @date_range").date(value_range=(start, None), default=start + timedelta(days=180))
        return cls(date_range=(start, end), tau=tau, **cls._SAMPLE_DICT)

    def _discretize(self, t, X):
        """Discretize the ODE.

        Args:
            t (int): discrete time-steps
            X (numpy.array): the current values of the model

        Returns:
            numpy.array: the next values of the model
        """
        raise NotImplementedError

    def solve(self):
        """Solve an initial value problem.

        Return:
            pandas.DataFrame: analytical solution
                Index
                    Date (pandas.Timestamp): dates from start date to end date
                Columns
                    (pandas.Int64): model-specific dimensional variables of the model
        """
        step_n = math.ceil((self._end - self._start) / timedelta(minutes=self._tau))
        sol = solve_ivp(
            fun=self._discretize,
            t_span=[0, step_n],
            y0=np.array([self._initial_dict[variable] for variable in self._VARIABLES]),
            t_eval=np.arange(0, step_n + 1, 1),
            dense_output=False
        )
        df = pd.DataFrame(data=sol["y"].T.copy(), columns=self._VARIABLES)
        df = self._non_dim_to_date(data=df, tau=self._tau, start_date=self._start)
        return df.round().convert_dtypes()

    @staticmethod
    def _date_to_non_dim(series, tau):
        """Convert date information (TIME) to time(x) = (TIME(x) - TIME(0)) / tau

        Args:
            series (pandas.DatetimeIndex): date information
            tau (int or None): tau value [min]

        Returns:
            pandas.DatetimeIndex: as-is @series when tau is None else converted time information without series name
        """
        Validator(series, "index of data").instance(pd.DatetimeIndex)
        if tau is None:
            return series
        Validator(tau, "tau", accept_none=False).tau()
        converted = (series - series.min()) / np.timedelta64(tau, "m")
        return converted.rename(None).astype("Int64")

    @classmethod
    def _non_dim_to_date(cls, data, tau, start_date):
        """Convert non-dimensional date information (time) to TIME(x) = TIME(0) + tau * time(x) and resample with dates.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index or pandas.DatetimeIndex (when @tau is not None)
                Columns
                    any columns
            tau (int or None): tau value [min]
            start_date (str or pandas.Timestamp or None): start date of records ie. TIME(0)

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.DatetimeIndex) or as-is @data (when either @tau or @start_date are None the index @data is date)
                Columns
                    any columns of @data
            pandas.DatetimeIndex: as-is @series when tau is None else converted time information without series name

        Note:
            The first values on date will be selected when resampling.
        """
        df = Validator(data, "data").dataframe()
        if tau is None or start_date is None or isinstance(df.index, pd.DatetimeIndex):
            return data
        Validator(tau, "tau", accept_none=False).tau()
        start = Validator(start_date, "start_date", accept_none=False).date()
        df[cls.DATE] = pd.date_range(start=start, periods=len(data), freq=f"{tau}min")
        return df.set_index(cls.DATE).resample("D").first()

    @classmethod
    def transform(cls, data, tau=None):
        """Transform a dataframe, converting Susceptible/Infected/Fatal/Recovered to model-specific variables.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index or pandas.DatetimeIndex (when tau is not None)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int or None): tau value [min]


        Returns:
            pandas.DataFrame:
                Index
                    as the same as index of @data when @tau is None else converted to time(x) = (TIME(x) - TIME(0)) / tau
                Columns
                    model-specific variables

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    @classmethod
    def inverse_transform(cls, data, tau=None, start_date=None):
        """Transform a dataframe, converting model-specific variables to Susceptible/Infected/Fatal/Recovered.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    model-specific variables
            tau (int or None): tau value [min]
            start_date (str or pandas.Timestamp or None): start date of records ie. TIME(0)

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.DatetimeIndex) or as-is @data (when either @tau or @start_date are None)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    def r0(self):
        """Calculate basic reproduction number.

        Returns:
            float: reproduction number of the ODE model and parameters

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    def dimensional_parameters(self):
        """Calculate dimensional parameter values.

        Returns:
            dict of {str: int or float}: dictionary of dimensional parameter values

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    @classmethod
    def from_data_with_quantile(cls, data, tau=1440, q=0.5, digits=None):
        """Initialize model with data, estimating ODE parameters with quantiles.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int): tau value [min]
            q (float): the quantiles to compute, values between (0, 1)
            digits (int or None): effective digits of ODE parameter values or None (skip rounding)

        Returns:
            covsirphy.ODEModel: initialized model
        """
        Validator(data, "data", accept_none=False).dataframe(columns=[cls.DATE, *cls._SIFR], empty_ok=False)
        Validator(tau, "tau", accept_none=False).tau()
        Validator(q, "q", accept_none=False).float(value_range=(0, 1))
        start, end = data[cls.DATE].min(), data[cls.DATE].max()
        trans_df = cls.transform(data=data.set_index(cls.DATE), tau=tau)
        initial_dict = trans_df.iloc[0].to_dict()
        param_dict = cls._param_quantile(data=trans_df, q=q)
        if digits is not None:
            param_dict = {k: Validator(v, k).float(value_range=(0, 1), digits=digits) for k, v in param_dict.items()}
        return cls(date_range=(start, end), tau=tau, initial_dict=initial_dict, param_dict=param_dict)

    @classmethod
    def _param_quantile(cls, data, q=0.5):
        """With combinations (X, dX/dt) for variables, calculate quantile values of ODE parameters.

        Args:
            data (pandas.DataFrame): transformed data with covsirphy.ODEModel.transform(data=data, tau=tau)
            q (float or array-like): the quantile(s) to compute, value(s) between (0, 1)

        Returns:
            dict of {str: float or pandas.Series}: parameter values at the quantile(s)

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    @classmethod
    def _clip(cls, values, lower, upper):
        """
        Trim values at input threshold.

        Args:
            values (float or array-like): values to trim
            lower (float): minimum threshold
            upper (float): maximum threshold

        Returns:
            float or pandas.Series: clipped array
        """
        return min(max(values, lower), upper) if isinstance(values, float) else pd.Series(values).clip()
