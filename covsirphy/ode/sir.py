#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIR(ModelBase):
    """
    SIR model.
    """
    # Model name
    NAME = "SIR"
    # names of parameters
    PARAMETERS = ["rho", "sigma"]
    # Variable names in dimensional ODEs
    VARIABLES = [super().S, super().SI, super().FR]
    # Priorities of the variables when optimization
    PRIORITIES = np.array([1, 1, 1])
    # Variables that increases monotonically
    VARS_INCLEASE = [super().FR]

    def __init__(self, population, rho, sigma):
        """
        This method should be overwritten in subclass.
        @population <int>: total population
        parameter values of non-dimensional ODE model
            - @rho <float>
            - @sigma <float>
        """
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
        This method should be overwritten in subclass.
        @return <np.array>
        """
        s, i, _, n = X, self.population
        dsdt = 0 - round(self.beta * s * i / n)
        drdt = round(self.sigma * i)
        didt = 0 - dsdt - drdt
        return np.array([dsdt, didt, drdt])

    @classmethod
    def param_range(cls, tau_free_df=None):
        """
        Define the range of parameters (not including tau value).
        This function should be overwritten in subclass.
        @tau_free_df <pd.DataFrame>:
            - columns: t and dimensional variables
        @return <dict[name]=(min, max)>:
            - min <float>: min value
            - max <float>: max value
        """
        df = cls.validate_tau_free(tau_free_df)
        t, x, y, z = df[cls.TS], df[cls.S], df[cls.CI], df[cls.FR]
        # rho = - (dx/dt) / x / y
        rho_series = 0 - x.diff() / t.diff() / x / y
        # sigma = (dz/dt) / y
        sigma_series = z.diff() / t / y
        # Calculate quantile
        _dict = {
            k: v.quantile(cls.QUANTILE_RANGE)
            for (k, v) in zip(
                ["rho", "sigma"], [rho_series, sigma_series]
            )
        }
        return _dict

    @classmethod
    def calc_variables(cls, cleaned_df, population):
        """
        Calculate the variables of the model.
        This method should be overwritten in subclass.
        @cleaned_df <pd.DataFrame>: cleaned data
            - index (Date) <pd.TimeStamp>: Observation date
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        @population <int>: total population in the place
        @return <pd.DataFrame>
            - index (Date) <pd.TimeStamp>: Observation date
            - Elapsed <int>: Elapsed time from the start date [min]
            - columns with dimensional variables
        """
        df = cls.calc_elapsed(cleaned_df)
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        df[cls.FR] = df[cls.F] + df[cls.R]
        return df.loc[:, [cls.T, *cls.VARIABLES]]

    def calc_r0(self):
        """
        This method should be overwritten in subclass.
        """
        rt = self.rho / self.sigma
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        This method should be overwritten in subclass.
        @param tau <int>: tau value [min]
        """
        _dict = dict()
        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)
        _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)
        return dict()
