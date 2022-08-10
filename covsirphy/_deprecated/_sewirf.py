#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.util.error import deprecate
from covsirphy.util.validator import Validator
from covsirphy._deprecated._mbase import ModelBase


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
        "x2": ModelBase.E,
        "x3": ModelBase.W,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F,
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([0, 10, 10, 2, 0, 0])
    # Variables that increases monotonically
    VARS_INCREASE = [ModelBase.R, ModelBase.F]
    # Example set of parameters and initial values
    EXAMPLE = {
        ModelBase.STEP_N: 180,
        ModelBase.N.lower(): 1_000_000,
        ModelBase.PARAM_DICT: {
            "theta": 0.002, "kappa": 0.005, "rho1": 0.2, "sigma": 0.075,
            "rho2": 0.167, "rho3": 0.167,
        },
        ModelBase.Y0_DICT: {
            ModelBase.S: 994_000, ModelBase.E: 3000, ModelBase.W: 2000,
            ModelBase.CI: 1000, ModelBase.R: 0, ModelBase.F: 0,
        },
    }

    @deprecate(old="SEWIRF", new="SEWIRFModel", version="2.24.0-xi")
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
        self.population = Validator(population, "population").int(value_range=(1, None))
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
        except (ZeroDivisionError, ValueError):
            return {p: None for p in self.DAY_PARAMETERS}

    @classmethod
    def convert(cls, data, tau):
        """
        Divide dates by tau value [min] and convert variables to model-specialized variables.

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
                    - Susceptible (int): the number of susceptible cases
                    - Exposed (int): 0
                    - Waiting (int): 0
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
        """
        # Convert to tau-free if tau was specified
        df = cls._convert(data, tau)
        # Conversion of variables
        df[cls.E] = 0
        df[cls.W] = 0
        return df.loc[:, [cls.S, cls.E, cls.W, cls.CI, cls.R, cls.F]]

    @classmethod
    def convert_reverse(cls, converted_df, start, tau):
        """
        Calculate date with tau and start date, and restore Susceptible/Infected/Fatal/Recovered.

        Args:
            converted_df (pandas.DataFrame):
                Index
                    time steps: Dates divided by tau value
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Exposed (int): exposed and in latent period (without infectivity)
                    - Waiting (int): waiting for confirmation diagnosis (with infectivity)
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
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
        # Calculate date with tau and start date
        df = cls._convert_reverse(converted_df, start, tau)
        # Conversion of variables
        df[cls.S] = df[cls.S] + df[cls.E] + df[cls.W]
        return df.loc[:, [cls.DATE, cls.S, cls.CI, cls.F, cls.R]]

    @classmethod
    def guess(cls, data, tau, q=0.5):
        """
        With (X, dX/dt) for X=S, I, R and so on, guess parameter values.
        This is not implemented for SEWIR-F model.

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
        """
        raise NotImplementedError(
            "SEWIR-F cannot be used for parameter estimation because we do not have records "
            "of Exposed and Waiting. Please use SIR-F model with `covsirphy.SIRF` class."
        )
