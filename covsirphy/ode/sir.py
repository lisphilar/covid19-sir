#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIR(ModelBase):
    """
    SIR model.

    Args:
        population (int): total population
        rho (float)
        sigma (float)
    """
    # Model name
    NAME = "SIR"
    # names of parameters
    PARAMETERS = ["rho", "sigma"]
    DAY_PARAMETERS = ["1/beta [day]", "1/gamma [day]"]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.FR
    }
    VARIABLES = list(VAR_DICT.values())
    # Priorities of the variables when optimization
    PRIORITIES = np.array([1, 1, 1])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.FR]

    def __init__(self, population, rho, sigma):
        # Total population
        if not isinstance(population, int):
            raise TypeError("@population must be an integer.")
        self.population = population
        # Non-dim parameters
        self.rho = rho
        self.sigma = sigma

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, i, *_ = X
        dsdt = 0 - self.rho * s * i / n
        drdt = self.sigma * i
        didt = 0 - dsdt - drdt
        return np.array([dsdt, didt, drdt])

    @classmethod
    def param_range(cls, taufree_df, population):
        """
        Define the range of parameters (not including tau value).

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
        df = cls.validate_dataframe(
            taufree_df, name="taufree_df", columns=[cls.TS, *cls.VARIABLES]
        )
        n, t, s, i, r = population, df[cls.TS], df[cls.S], df[cls.CI], df[cls.FR]
        # rho = - n * (dS/dt) / S / I
        rho_series = 0 - n * s.diff() / t.diff() / s / i
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # Calculate quantile
        _dict = {
            k: v.quantile(cls.QUANTILE_RANGE)
            for (k, v) in zip(["rho", "sigma"], [rho_series, sigma_series])
        }
        return _dict

    @classmethod
    def specialize(cls, data_df, population):
        """
        Specialize the dataset for this model.

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
                        - Susceptible (int): the number of susceptible cases
                        - Fatal or Recovered (int): total number of fatal/recovered cases
        """
        df = super().specialize(data_df, population)
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        df[cls.FR] = df[cls.F] + df[cls.R]
        return df

    @classmethod
    def restore(cls, specialized_df):
        """
        Restore Confirmed/Infected/Recovered/Fatal.
         using a dataframe with the variables of the model.

        Args:
            specialized_df (pandas.DataFrame): dataframe with the variables

                Index:
                    reset index
                Columns:
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal or Recovered (int): the number of fatal/recovered cases
                    - any columns

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - the other columns @specialzed_df has
        """
        df = specialized_df.copy()
        other_cols = list(set(df.columns) - set(cls.VALUE_COLUMNS))
        df[cls.C] = df[cls.CI] + df[cls.FR]
        df[cls.F] = 0
        df[cls.R] = df[cls.FR]
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.
        """
        rt = self.rho / self.sigma
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.

        Args:
            param tau (int): tau value [min]
        """
        _dict = {
            "1/beta [day]": int(tau / 24 / 60 / self.rho),
            "1/gamma [day]": int(tau / 24 / 60 / self.sigma)
        }
        return _dict
