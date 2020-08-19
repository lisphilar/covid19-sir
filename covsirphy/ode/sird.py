#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIRD(ModelBase):
    """
    SIR-D model.

    Args:
        population (int): total population
        kappa (float)
        rho (float)
        sigma (float)
    """
    # Model name
    NAME = "SIR-D"
    # names of parameters
    PARAMETERS = ["kappa", "rho", "sigma"]
    DAY_PARAMETERS = ["1/alpha2 [day]", "1/beta [day]", "1/gamma [day]"]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([1, 10, 10, 2])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.R, ModelBase.F]
    # Example set of parameters and initial values
    EXAMPLE = {
        "step_n": 180,
        "population": 1_000_000,
        "param_dict": {
            "kappa": 0.005, "rho": 0.2, "sigma": 0.075,
        },
        "y0_dict": {
            ModelBase.S: 999_000, ModelBase.CI: 1000, ModelBase.R: 0, ModelBase.F: 0,
        },
    }

    def __init__(self, population, kappa, rho, sigma):
        # Total population
        self.population = self.ensure_natural_int(
            population, name="population"
        )
        # Non-dim parameters
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma
        self.non_param_dict = {"kappa": kappa, "rho": rho, "sigma": sigma}

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
        dfdt = self.kappa * i
        didt = 0 - dsdt - drdt - dfdt
        return np.array([dsdt, didt, drdt, dfdt])

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
        df = cls.ensure_dataframe(
            taufree_df, name="taufree_df", columns=[cls.TS, *cls.VARIABLES]
        )
        n, t = population, df[cls.TS]
        s, i, r, d = df[cls.S], df[cls.CI], df[cls.R], df[cls.F]
        # kappa = (dD/dt) / I
        kappa_series = d.diff() / t.diff() / i
        # rho = - n * (dS/dt) / S / I
        rho_series = 0 - n * s.diff() / t.diff() / s / i
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # Calculate quantile
        _dict = {
            k: v.quantile(cls.QUANTILE_RANGE)
            for (k, v) in zip(
                ["kappa", "rho", "sigma"],
                [kappa_series, rho_series, sigma_series]
            )
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
        """
        df = cls.ensure_dataframe(
            data_df, name="data_df", columns=cls.VALUE_COLUMNS)
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        return df

    @classmethod
    def restore(cls, specialized_df):
        """
        Restore Confirmed/Infected/Recovered/Fatal using a dataframe with the variables of the model.

        Args:
        @specialized_df (pandas.DataFrame): dataframe with the variables

            Index:
                reset index
            Columns:
                - Susceptible (int): the number of susceptible cases
                - Infected (int): the number of currently infected cases
                - Recovered (int): the number of recovered cases
                - Fatal (int): the number of fatal cases
                - any columns

        Returns:
            (pandas.DataFrame)
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
        df[cls.C] = df[cls.CI] + df[cls.R] + df[cls.F]
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.

        Returns:
            float
        """
        try:
            rt = self.rho / (self.sigma + self.kappa)
        except ZeroDivisionError:
            return None
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.

        Args:
            param tau (int): tau value [min]

        Returns:
            dict[str, int]
        """
        try:
            return {
                "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
                "1/beta [day]": int(tau / 24 / 60 / self.rho),
                "1/gamma [day]": int(tau / 24 / 60 / self.sigma)
            }
        except ZeroDivisionError:
            return {p: None for p in self.DAY_PARAMETERS}
