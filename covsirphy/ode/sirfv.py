#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIRFV(ModelBase):
    """
    SIR-FV model.
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
        """
        @population <int>: total population
        parameter values of non-dimensional ODE model
            - @theta <float>
            - @kappa <float>
            - @rho <float>
            - @sigma <float>
            - @omega <float> or v_per_day <int>
        """
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
                raise ValueError("@v_per_day / @population does not match @omega.")
        self.omega = omega

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.
        @return <np.array>
        """
        n = self.population
        s, i, *_ = X
        beta_si = round(self.rho * s * i / n)
        dvdt = round(self.w * n)
        dsdt = 0 - beta_si - dvdt
        drdt = round(self.sigma * i)
        dfdt = round(self.kappa * i) + round((0 - beta_si) * self.theta)
        didt = 0 - dsdt - drdt - dfdt - dvdt
        return np.array([dsdt, didt, drdt, dfdt, dvdt])

    @classmethod
    def param_range(cls, taufree_df, population):
        """
        Define the range of parameters (not including tau value).
        @taufree_df <pd.DataFrame>:
            - index: reset index
            - t <int>: time steps (tau-free)
            - columns with dimensional variables
        @population <int>: total population
        @return <dict[name]=(min, max)>:
            - min <float>: min value
            - max <float>: max value
        """
        df = cls.validate_dataframe(
            taufree_df, name="taufree_df", columns=[cls.TS, *cls.VARIABLES]
        )
        n, t = population, df[cls.TS]
        s, i, r, f = df[cls.S], df[cls.CI], df[cls.R], df[cls.F]
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t / i
        # omega = 0 - (dS/dt + dI/dt + dR/dt + dF/dt) / n
        omega_series = 0 - (s + i + r + f).diff() / t / n
        # Calculate quantile
        _dict = {
            k: v.quantile(cls.QUANTILE_RANGE)
            for (k, v) in zip(["sigma", "omega"], [sigma_series, omega_series])
        }
        _dict["theta"] = (0, 1)
        _dict["kappa"] = (0, 1)
        _dict["rho"] = (0, 1)
        return _dict

    @classmethod
    def specialize(cls, data_df, population):
        """
        Specialize the dataset for this model.
        @data_df <pd.DataFrame>:
            - index: reset index
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
            - any columns
        @population <int>: total population in the place
        @return <pd.DataFrame>:
            - index: reset index
            - any columns @data_df has
            - Susceptible <int>: 0
            - Vactinated <int>: 0
        """
        df = super().specialize(data_df, population)
        # Calculate dimensional variables
        df[cls.S] = 0
        df[cls.V] = 0
        return df

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.
        """
        rt = self.rho * (1 - self.theta) / (self.sigma + self.kappa)
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        @param tau <int>: tau value [min]
        """
        _dict = {
            "alpha1": round(self.theta, 3),
            "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
            "1/beta [day]": int(tau / 24 / 60 / self.rho),
            "1/gamma [day]": int(tau / 24 / 60 / self.sigma),
            "Vaccinated [persons]": int(self.omega * self.population)
        }
        return _dict
