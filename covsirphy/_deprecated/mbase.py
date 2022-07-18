#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate, NAFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class ModelBase(Term):
    """
    Base class of ODE models.
    """
    # Model name
    NAME = "ModelBase"
    # names of parameters
    PARAMETERS = list()
    DAY_PARAMETERS = list()
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = dict()
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array(list())
    # Variables that increases monotonically
    VARS_INCREASE = list()
    # Example set of parameters and initial values
    EXAMPLE = {
        Term.STEP_N: 180,
        Term.N.lower(): 1_000_000,
        Term.PARAM_DICT: dict(),
        Term.Y0_DICT: dict(),
    }

    @deprecate(old="ModelBase", new="ODEModel", version="2.24.0-xi")
    def __init__(self, population):
        """
        This method should be overwritten in subclass.

        Args:
            population (int): total population
        """
        # Total population
        self.population = Validator(population, "population").int(value_range=(1, None))
        # Dictionary of non-dim parameters: {name: value}
        self.non_param_dict = {}

    def __str__(self):
        param_str = ", ".join(f"{p}={v}" for (
            p, v) in self.non_param_dict.items())
        return f"{self.NAME} model with {param_str}"

    def __getitem__(self, key):
        """
        Args:
            key (str): parameter name
        """
        if key not in self.non_param_dict.keys():
            raise KeyError(f"key must be in {', '.join(self.PARAMETERS)}")
        return self.non_param_dict[key]

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.
        This method should be overwritten in subclass.

        Returns:
            np.array
        """
        raise NotImplementedError

    @classmethod
    @deprecate(".param_range()", new=".guess()", version="2.19.1-zeta-fu1")
    def param_range(cls, taufree_df, population, quantiles=(0.3, 0.7)):
        """
        Deprecated. Define the value range of ODE parameters using (X, dX/dt) points.
        This method should be overwritten in subclass.

        Args:
            taufree_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - t (int): time steps (tau-free)
                    - columns with dimensional variables
            population (int): total population
            quantiles (tuple(int, int)): quantiles to cut, like confidence interval

        Returns:
            dict(str, tuple(float, float)): minimum/maximum values
        """
        raise NotImplementedError

    @classmethod
    @deprecate(".specialize()", new=".convert()", version="2.19.1-zeta-fu1")
    def specialize(cls, data_df, population):
        """
        Deprecated. Specialize the dataset for this model.
        This method should be overwritten in subclass.

        Args:
            data_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): total population in the place

        Returns:
            (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - any columns @data_df has
                    - columns with dimensional variables
        """
        raise NotImplementedError

    @classmethod
    @deprecate(".restore()", new=".convert_reverse()", version="2.19.1-zeta-fu1")
    def restore(cls, specialized_df):
        """
        Deprecated. Restore Confirmed/Infected/Recovered/Fatal using a dataframe with the variables of the model.
        This method should be overwritten in subclass.

        Args:
            specialized_df (pandas.DataFrame): dataframe with the variables

                Index
                    (object):
                Columns
                    - variables of the models (int)
                    - any columns

        Returns:
            (pandas.DataFrame):
                Index
                    (object): as-is
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - the other columns @specialized_df has
        """
        raise NotImplementedError

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.
        This method should be overwritten in subclass.

        Returns:
            float
        """
        raise NotImplementedError

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        This method should be overwritten in subclass.

        Args:
            param tau (int): tau value [min]

        Returns:
            dict[str, int]
        """
        raise NotImplementedError

    @classmethod
    @deprecate(".taufree()", new=".convert()", version="2.19.1-zeta-fu1")
    def tau_free(cls, subset_df, population, tau=None):
        """
        Deprecated. Create a dataframe specialized to the model.
        If tau is not None, Date column will be converted to '(Date - start date) / tau'
        and saved in t column.

        Args:
            subset_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): population value
            tau (int or None): tau value [min], 0 <= tau <= 1440

        Returns:
            (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - t (int): if tau is not None
                    - columns with dimensional variables
        """
        raise NotImplementedError

    @classmethod
    def _convert(cls, data, tau):
        """
        Divide dates by tau value [min].

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
            tau (int): tau value [min] or None (skip division by tau values)

        Returns:
            pandas.DataFrame:
                Index
                    - Date (pd.Timestamp): Observation date (available when @tau is None)
                    - t (int): time steps (available when @tau is not None)
                Columns
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = Validator(data, "data").dataframe(columns=cls.DSIFR_COLUMNS)
        if tau is None:
            return df.set_index(cls.DATE)
        # Convert to tau-free
        tau = Validator(tau, "tau").tau(default=None)
        if tau is None:
            raise NAFoundError("tau", None)
        time_series = (pd.to_datetime(df[cls.DATE]) - df[cls.DATE].min()).dt.total_seconds() // 60
        df.index = (time_series / tau).astype(np.int64)
        df.index.name = cls.TS
        return df.drop(cls.DATE, axis=1)

    @classmethod
    def _convert_reverse(cls, converted_df, start, tau):
        """
        Calculate date with tau and start date.

        Args:
            converted_df (pandas.DataFrame):
                Index
                    t: Dates divided by tau value (time steps)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal or Recovered (int): the number of fatal/recovered cases
            start (pd.Timestamp): start date of simulation, like 14Apr2021
            tau (int): tau value [min]

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = converted_df.copy()
        # Calculate date with tau and start date
        tau = Validator(tau, "tau").tau(default=None)
        if tau is None:
            raise NAFoundError("tau", None)
        elapsed = pd.Series(df.index * tau)
        df[cls.DATE] = start + elapsed.apply(lambda x: timedelta(minutes=x))
        # Select the last records for dates
        df = df.set_index(cls.DATE).resample("D").last().reset_index()
        return df

    @classmethod
    def convert(cls, data, tau):
        """
        Divide dates by tau value [min] and convert variables to model-specialized variables.
        This will be overwritten by child classes.

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
            tau (int): tau value [min] or None (skip division by tau values)

        Returns:
            pandas.DataFrame:
                Index
                    time steps: Dates divided by tau value
                Columns
                    - model-specialized variables
        """
        raise NotImplementedError

    @classmethod
    def convert_reverse(cls, converted_df, start, tau):
        """
        Calculate date with tau and start date, and restore Susceptible/Infected/"Fatal or Recovered".
        This will be overwritten by child classes.

        Args:
            converted_df (pandas.DataFrame):
                Index
                    time steps: Dates divided by tau value
                Columns
                    - model-specialized variables
            start (pd.Timestamp): start date of simulation, like 14Apr2021
            tau (int): tau value [min]

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        raise NotImplementedError

    @classmethod
    def guess(cls, data, tau, q=0.5):
        """
        With (X, dX/dt) for X=S, I, R and so on, guess parameter values.
        This will be overwritten by child classes.

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
            q (float or tuple(float,)): the quantile(s) to compute, value(s) between (0, 1)

        Returns:
            dict(str, float or pandas.Series): guessed parameter values with the quantile(s)
        """
        raise NotImplementedError

    @classmethod
    def _clip(cls, values, lower, upper):
        """
        Trim values at input threshold.

        Args:
            values (float or pandas.Series): values to trim
            lower (float): minimum threshold
            upper (float): maximum threshold
        """
        if isinstance(values, float):
            return min(max(values, lower), upper)
        Validator(values, "values").instance(pd.Series)
        return values.clip()
