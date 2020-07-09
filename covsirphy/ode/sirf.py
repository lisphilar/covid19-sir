#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numba import njit
import numpy as np
from covsirphy.ode.mbase import ModelBase


@njit
def ode_sirf(X, n, theta, kappa, rho, sigma):
    """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
    s, i, = X[0], X[1]
    dsdt = 0 - rho * s * i / n
    drdt = sigma * i
    dfdt = kappa * i + (0 - dsdt) * theta
    didt = 0 - dsdt - drdt - dfdt
    return np.array([dsdt, didt, drdt, dfdt])


class SIRF(ModelBase):
    """
    SIR-F model.

    Args:
        population (int): total population
        theta (float)
        kappa (float)
        rho (float)
        sigma (float)
    """
    # Model name
    NAME = "SIR-F"
    # names of parameters
    PARAMETERS = ["theta", "kappa", "rho", "sigma"]
    DAY_PARAMETERS = [
        "alpha1 [-]", "1/alpha2 [day]", "1/beta [day]", "1/gamma [day]"
    ]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F
    }
    VARIABLES = list(VAR_DICT.values())
    # Priorities of the variables when optimization
    PRIORITIES = np.array([1, 10, 10, 2])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.R, ModelBase.F]
    # Example set of parameters and initial values
    EXAMPLE = {
        "step_n": 180,
        "population": 1_000_000,
        "param_dict": {
            "theta": 0.002, "kappa": 0.005, "rho": 0.2, "sigma": 0.075,
        },
        "y0_dict": {
            "Susceptible": 999_000, "Infected": 1000, "Recovered": 0, "Fatal": 0,
        },
    }

    def __init__(self, population, theta, kappa, rho, sigma):
        # Total population
        if not isinstance(population, int):
            raise TypeError("@population must be an integer.")
        self.population = population
        # Non-dim parameters
        self.theta = theta
        self.kappa = kappa
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
        return ode_sirf(X, self.population, self.theta, self.kappa, self.rho, self.sigma)

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
        s, i, r = df[cls.S], df[cls.CI], df[cls.R]
        # rho = - n * (dS/dt) / S / I
        rho_series = 0 - n * s.diff() / t.diff() / s / i
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # Calculate quantile
        _dict = {
            k: v.quantile(cls.QUANTILE_RANGE)
            for (k, v) in zip(["rho", "sigma"], [rho_series, sigma_series])
        }
        _dict["theta"] = (0, 1)
        _dict["kappa"] = (0, 1)
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
        """
        df = super().specialize(data_df, population)
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
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
            "alpha1 [-]": round(self.theta, 3),
            "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
            "1/beta [day]": int(tau / 24 / 60 / self.rho),
            "1/gamma [day]": int(tau / 24 / 60 / self.sigma)
        }
        return _dict
