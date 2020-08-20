#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.cleaning.term import Term


class ModelBase(Term):
    """
    Base class of ODE models.
    """
    # Quartile range of the parametes when setting initial values
    QUANTILE_RANGE = [0.3, 0.7]
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
    VARS_INCLEASE = list()
    # Example set of parameters and initial values
    EXAMPLE = {
        "step_n": 180,
        "population": 1_000_000,
        "param_dict": dict(),
        "y0_dict": dict(),
    }

    def __init__(self, population):
        """
        This method should be overwritten in subclass.

        Args:
            population (int): total population
        """
        # Total population
        self.population = self.ensure_natural_int(
            population, name="population"
        )
        # Dictionary of non-dim parameters: {name: value}
        self.non_param_dict = {}

    def __str__(self):
        param_str = ", ".join(
            [f"{p}={v}" for (p, v) in self.non_param_dict.items()]
        )
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
    def param_range(cls, taufree_df, population):
        """
        Define the range of parameters (not including tau value).
        This method should be overwritten in subclass.

        Args:
            taufree_df (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - t (int): time steps (tau-free)
                    - columns with dimensional variables
            population (int): total population

        Returns:
            (dict)
                - key (str): parameter name
                - value (tuple(float, float)): min value and max value
        """
        raise NotImplementedError

    @classmethod
    def specialize(cls, data_df, population):
        """
        Specialize the dataset for this model.
        This method should be overwritten in subclass.

        Args:
            data_df (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): total population in the place

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - any columns @data_df has
                    - columns with dimensional variables
        """
        raise NotImplementedError

    @classmethod
    def restore(cls, specialized_df):
        """
        Restore Confirmed/Infected/Recovered/Fatal using a dataframe with the variables of the model.
        This method should be overwritten in subclass.

        Args:
            specialized_df (pandas.DataFrame): dataframe with the variables

                Index:
                    (object):
                Columns:
                    - variables of the models (int)
                    - any columns

        Returns:
            (pandas.DataFrame):
                Index:
                    (object): as-is
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - the other columns @specialzed_df has
        """
        df = specialized_df.copy()
        other_cols = list(set(df.columns) - set(cls.VALUE_COLUMNS))
        df[cls.C] = None
        df[cls.CI] = None
        df[cls.F] = None
        df[cls.R] = None
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

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
    def tau_free(cls, subset_df, population, tau=None):
        """
        Create a dataframe specialized to the model.
        If tau is not None, Date column will be converted to '(Date - start date) / tau'
        and saved in t column.

        Args:
            subset_df (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): population value
            tau (int or None): tau value [min], 0 <= tau <= 1440

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - t (int): if tau is not None
                    - columns with dimensional variables
        """
        df = subset_df.copy()
        if tau is None:
            df = df.drop(cls.DATE, axis=1)
            return cls.specialize(df, population=population).reset_index(drop=True)
        tau = cls.ensure_tau(tau)
        df = cls.ensure_dataframe(
            df, name="data_df", columns=cls.NLOC_COLUMNS)
        # Calculate elapsed time from the first date [min]
        df[cls.T] = (df[cls.DATE] - df[cls.DATE].min()).dt.total_seconds()
        df[cls.T] = df[cls.T] // 60
        # Convert to tau-free
        df[cls.TS] = (df[cls.T] / tau).astype(np.int64)
        df = df.drop([cls.T, cls.DATE], axis=1)
        return cls.specialize(df, population).reset_index(drop=True)
