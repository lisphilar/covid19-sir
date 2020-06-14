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
    # Variable names in dimensional ODEs
    VARIABLES = [
        super().S, super().SI, super().R, super().F,
        super().E, super().W
    ]
    # Priorities of the variables when optimization
    PRIORITIES = np.array([0, 10, 10, 2, 0, 0])
    # Variables that increases monotonically
    VARS_INCLEASE = [super().R, super().F]

    def __init__(self, population, theta, kappa, rho1, rho2, rho3, sigma):
        """
        @population <int>: total population
        parameter values of non-dimensional ODE model
            - @theta <float>
            - @kappa <float>
            - @rho1 <float>
            - @rho2 <float>
            - @rho3 <float>
            - @sigma <float>
        """
        # Total population
        if not isinstance(population, int):
            raise TypeError("@population must be an integer.")
        self.population = population
        # Non-dim parameters
        self.theta = theta
        self.kappa = kappa
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.sigma = sigma

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.
        @return <np.array>
        """
        n, s, i, *_, e, w = self.population, X
        beta_swi = round(self.rho1 * s * (w + i) / n)
        dsdt = 0 - beta_swi
        dedt = beta_swi - round(self.rho2 * e)
        dwdt = round(self.rho2 * e) - round(self.rho3 * w)
        drdt = round(self.sigma * i)
        dfdt = round(self.kappa * i) + round(self.theta * self.rho3 * w)
        didt = 0 - dsdt - drdt - dfdt - dedt - dwdt
        return np.array([dsdt, didt, drdt, dfdt, dedt, dwdt])

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
        _, t, i, r = population, df[cls.TS], df[cls.CI], df[cls.R]
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t / i
        # Calculate quantile
        _dict = {param: (0, 1) for param in cls.PARAMETERS}
        _dict["sigma"] = sigma_series.quantile(cls.QUANTILE_RANGE)
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
            - Susceptible <int>: the number of susceptible cases
            - Exposed <int>: 0
            - Waiting <int>: 0
        """
        df = super().specialize(data_df, population)
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        df[cls.E] = 0
        df[cls.W] = 0
        return df

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.
        """
        rho = self.rho1 / self.rho2 * self.rho3
        rt = rho * (1 - self.theta) / (self.sigma + self.kappa)
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        @param tau <int>: tau value [min]
        """
        _dict = {
            "alpha1 [-]": round(self.theta, 3),
            "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
            "1/beta1 [day]": int(tau / 24 / 60 / self.rho1),
            "1/beta2 [day]": int(tau / 24 / 60 / self.rho2),
            "1/beta3 [day]": int(tau / 24 / 60 / self.rho3),
            "1/gamma [day]": int(tau / 24 / 60 / self.sigma)
        }
        return _dict
