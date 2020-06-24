#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIRFV(ModelBase):
    """
    SIR-FV model.

    Args:
        population (int): total population
            theta (float)
            kappa (float)
            rho (float)
            sigma (float)
            omega (float) or v_per_day (int)
    """
    # Model name
    NAME = "SIR-FV"
    # names of parameters
    PARAMETERS = ["theta", "kappa", "rho", "sigma", "omega"]
    DAY_PARAMETERS = [
        "alpha1", "1/alpha2 [day]", "1/beta [day]", "1/gamma [day]",
        "Vaccinated [persons]"
    ]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F,
        "v": ModelBase.V
    }
    VARIABLES = list(VAR_DICT.values())
    # Priorities of the variables when optimization
    PRIORITIES = np.array([0, 10, 10, 2, 0])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.R, ModelBase.F]

    def __init__(self, population, theta, kappa, rho, sigma,
                 omega=None, v_per_day=None):
        # Total population
        if not isinstance(population, int):
            raise TypeError("@population must be an integer.")
        self.population = population
        # Non-dim parameters
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma
        if omega is None:
            if v_per_day is None:
                raise TypeError("@omega or @v_per_day must be applied.")
            omega = v_per_day / population
        else:
            if v_per_day is not None and omega != v_per_day / population:
                raise ValueError(
                    "@v_per_day / @population does not match @omega.")
        self.omega = omega

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
        beta_si = self.rho * s * i / n
        dvdt = self.omega * n
        dsdt = 0 - beta_si - dvdt
        drdt = self.sigma * i
        dfdt = self.kappa * i + (0 - beta_si) * self.theta
        didt = 0 - dsdt - drdt - dfdt - dvdt
        return np.array([dsdt, didt, drdt, dfdt, dvdt])

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
        n, t = population, df[cls.TS]
        s, i, r, f = df[cls.S], df[cls.CI], df[cls.R], df[cls.F]
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # omega = 0 - (dS/dt + dI/dt + dR/dt + dF/dt) / n
        omega_series = (n - s + i + r + f).diff() / t.diff() / n
        # Calculate range
        _dict = {param: (0, 1) for param in cls.PARAMETERS}
        _dict["sigma"] = sigma_series.quantile(cls.QUANTILE_RANGE)
        _dict["omega"] = omega_series.quantile(cls.QUANTILE_RANGE)
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
                    - Susceptible (int): 0
                    - Vactinated (int): 0
        """
        df = super().specialize(data_df, population)
        # Calculate dimensional variables
        df[cls.S] = 0
        df[cls.V] = 0
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
                - Vaccinated (int): the number of vactinated persons
                - any columns

        Returns:
            (pandas.DataFrame)
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
        """
        rt = self.rho * (1 - self.theta) / (self.sigma + self.kappa)
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.

        Args:
            param tau (int): tau value [min]
        """
        _dict = {
            "alpha1": round(self.theta, 3),
            "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
            "1/beta [day]": int(tau / 24 / 60 / self.rho),
            "1/gamma [day]": int(tau / 24 / 60 / self.sigma),
            "Vaccinated [persons]": int(self.omega * self.population)
        }
        return _dict
