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
    """Basic class of ordinary differential equation (ODE) models which solves initial value problem.

    Args:
        date_range (tuple(str, str)): start date and end date of simulation
        tau (int): tau value [min]
        initial_dict (dict[str, int]): initial values
        param_dict (dict[str, float]): parameter values
    """
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
        # Information regarding parameter estimation

    def __str__(self):
        return self._NAME

    def __repr__(self):
        _dict = {
            "date_range": (self._start.strftime(self.DATE_FORMAT), self._end.strftime(self.DATE_FORMAT)),
            "tau": self._tau,
            "initial_dict": self._initial_dict,
            "param_dict": self._param_dict,
        }
        return f"{type(self).__name__}({', '.join([f'{k}={v}' for k, v in _dict.items()])})"

    @classmethod
    def from_sample(cls, date_range=None, tau=1440):
        """Initialize model with sample data.

        Args:
            date_range (tuple(str or None, str or None) or None): start date and end date of simulation
            tau (int): tau value [min]

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
            X (numpy.array): current values of the model

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    def solve(self):
        """Solve an initial value problem.

        Return:
            pandas.DataFrame: numerical solution
                Index
                    reset index: time steps
                Columns
                    (pandas.Int64): dimensional variables of the model
        """
        step_n = math.ceil((self._end - self._start) / timedelta(minutes=self._tau))
        sol = solve_ivp(
            fun=self._discretize,
            t_span=[0, step_n],
            y0=np.array([self._initial_dict[variable] for variable in self._VARIABLES]),
            t_eval=np.arrange(0, step_n + 1, 1),
            dense_output=False
        )
        df = pd.DataFrame(data=sol["y"].T.copy(), columns=self._VARIABLES)
        return df.round().convert_dtypes()

    @classmethod
    def transform(cls, data):
        """Transform a dataframe, converting Susceptible/Infected/Fatal/Recovered to model-specific variables.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Returns:
            pandas.DataFrame:
                Index
                    as the same as index if @data
                Columns
                    model-specific variables

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    @classmethod
    def inverse_transform(cls, data):
        """Transform a dataframe, converting model-specific variables to Susceptible/Infected/Fatal/Recovered.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    model-specific variables

        Returns:
            pandas.DataFrame:
                Index
                    as the same as index if @data
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
