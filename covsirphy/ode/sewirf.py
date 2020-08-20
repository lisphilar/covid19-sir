#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SEWIRF(ModelBase):
    """
    SEWIR-F model.
    """
    # Model name
    NAME = "SEWIR-F"
    # names of parameters
    PARAMETERS = ["theta", "kappa", "rho1", "rho2", "rho3", "sigma"]
    DAY_PARAMETERS = [
        "alpha1 [-]", "1/alpha2 [day]",
        "1/beta1 [day]", "1/beta2 [day]", "1/beta3 [day]",
        "1/gamma [day]"
    ]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x1": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F,
        "x2": ModelBase.E,
        "x3": ModelBase.W,
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([0, 10, 10, 2, 0, 0])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.R, ModelBase.F]
    # Example set of parameters and initial values
    EXAMPLE = {
        "step_n": 180,
        "population": 1_000_000,
        "param_dict": {
            "theta": 0.002, "kappa": 0.005, "rho1": 0.2, "sigma": 0.075,
            "rho2": 0.167, "rho3": 0.167,
        },
        "y0_dict": {
            ModelBase.S: 994_000, ModelBase.E: 3000, ModelBase.W: 0.002,
            ModelBase.CI: 1000, ModelBase.R: 0, ModelBase.F: 0,
        },
    }

    def __init__(self, population, theta, kappa, rho1, rho2, rho3, sigma):
        """

        Args:
        @population (int): total population
        parameter values of non-dimensional ODE model
            - @theta (float)
            - @kappa (float)
            - @rho1 (float)
            - @rho2 (float)
            - @rho3 (float)
            - @sigma (float)
        """
        # Total population
        self.population = self.ensure_natural_int(
            population, name="population"
        )
        # Non-dim parameters
        self.theta = theta
        self.kappa = kappa
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.sigma = sigma
        self.non_param_dict = {
            "theta": theta, "kappa": kappa,
            "rho1": rho1, "rho2": rho2, "rho3": rho3,
            "sigma": sigma
        }

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
        s, i, *_, e, w = X
        beta_swi = self.rho1 * s * (w + i) / n
        dsdt = 0 - beta_swi
        dedt = beta_swi - self.rho2 * e
        dwdt = self.rho2 * e - self.rho3 * w
        drdt = self.sigma * i
        dfdt = self.kappa * i + self.theta * self.rho3 * w
        didt = 0 - dsdt - drdt - dfdt - dedt - dwdt
        return np.array([dsdt, didt, drdt, dfdt, dedt, dwdt])

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
        _, t, i, r = population, df[cls.TS], df[cls.CI], df[cls.R]
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # Calculate range
        _dict = {param: (0, 1) for param in cls.PARAMETERS}
        _dict["sigma"] = sigma_series.quantile(cls.QUANTILE_RANGE)
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
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - any columns @data_df has
                    - Susceptible (int): the number of susceptible cases
                    - Exposed (int): 0
                    - Waiting (int): 0
        """
        df = cls.ensure_dataframe(
            data_df, name="data_df", columns=cls.VALUE_COLUMNS)
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        df[cls.E] = 0
        df[cls.W] = 0
        return df

    @classmethod
    def restore(cls, specialized_df):
        """
        Restore Confirmed/Infected/Recovered/Fatal.
         using a dataframe with the variables of the model.

        Args:
            specialized_df (pandas.DataFrame): dataframe with the variables

                Index:
                    (object)
                Columns:
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
                    - Exposed (int): Exposed and in latent period (without infectivity)
                    - Waiting (int): Waiting cases for confirmation (with infectivity)
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
        df[cls.C] = df[cls.CI] + df[cls.R] + df[cls.F]
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.

        Returns:
            float
        """
        try:
            rho = self.rho1 / self.rho2 * self.rho3
            rt = rho * (1 - self.theta) / (self.sigma + self.kappa)
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
                "alpha1 [-]": round(self.theta, 3),
                "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
                "1/beta1 [day]": int(tau / 24 / 60 / self.rho1),
                "1/beta2 [day]": int(tau / 24 / 60 / self.rho2),
                "1/beta3 [day]": int(tau / 24 / 60 / self.rho3),
                "1/gamma [day]": int(tau / 24 / 60 / self.sigma)
            }
        except ZeroDivisionError:
            return {p: None for p in self.DAY_PARAMETERS}
