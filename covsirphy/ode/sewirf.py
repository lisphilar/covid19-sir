#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SEWIRF(ModelBase):
    NAME = "SEWIR-F"
    PARAMETERS = ["theta", "kappa", "rho1", "rho2", "rho3", "sigma"]
    DAY_PARAMETERS = [
        "alpha1 [-]", "1/alpha2 [day]",
        "1/beta1 [day]", "1/beta2 [day]", "1/beta3 [day]",
        "1/gamma [day]"
    ]
    VARIABLES = ["x1", "x2", "x3", "y", "z", "w"]
    PRIORITIES = np.array([0, 0, 0, 10, 10, 2])
    VARS_INCLEASE = ["z", "w"]

    def __init__(self, theta, kappa, rho1, rho2, rho3, sigma):
        super().__init__()
        self.theta = theta
        self.kappa = kappa
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.sigma = sigma

    def __call__(self, t, X):
        x1, x2, x3, y, z, w = X
        y = max(y, 0)
        dx1dt = - self.rho1 * x1 * (x3 + y)
        dx2dt = self.rho1 * x1 * (x3 + y) - self.rho2 * x2
        dx3dt = self.rho2 * x2 - self.rho3 * x3
        dydt = self.rho3 * (1 - self.theta) * x3 - (self.sigma + self.kappa) * y
        dzdt = self.sigma * y
        dwdt = self.rho3 * self.theta * x3 + self.kappa * y
        if y + dydt < 0:
            dydt = 0 - y
        return np.array([dx1dt, dx2dt, dx3dt, dydt, dzdt, dwdt])

    @classmethod
    def param(cls, tau_free_df=None, q_range=None):
        param_dict = super().param()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
        param_dict["theta"] = (0, 1)
        param_dict["kappa"] = (0, 1)
        param_dict["rho1"] = (0, 1)
        param_dict["rho2"] = (0, 1)
        param_dict["rho3"] = (0, 1)
        if tau_free_df is not None:
            df = tau_free_df.copy()
            # sigma = (dz/dt) / y
            sigma_series = df["z"].diff() / df["t"].diff() / df["y"]
            param_dict["sigma"] = sigma_series.quantile(q_range)
            return param_dict
        param_dict["sigma"] = (0, 1)
        return param_dict

    @classmethod
    def calc_variables(cls, cleaned_df, population):
        """
        Calculate the variables of SIR-F model.
        This function overwrites the parent class.
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
            - x1: Susceptible / Population
            - x2: 0 (will not be used for hyperparameter estimation)
            - x3: 0 (will not be used for hyperparameter estimation)
            - y: Infected / Population
            - z: Recovered / Population
            - w: Fatal / Population
        """
        df = super().calc_variables(cleaned_df, population)
        df["X1"] = df[cls.S]
        df["X2"] = 0
        df["X3"] = 0
        df["Y"] = df[cls.CI]
        df["Z"] = df[cls.R]
        df["W"] = df[cls.F]
        cols = ["X1", "X2", "X3", "Y", "Z", "W"]
        # Columns will be changed to lower cases
        return cls.nondim_cols(df, cols, population)

    @classmethod
    def calc_variables_reverse(cls, df, population):
        """
        Calculate measurable variables.
        @df <pd.DataFrame>:
            - index: reset index
            - x1: Susceptible / Population
            - x2: Exposed / Population
            - x3: Waiting / Population
            - y: Infected / Population
            - z: Recovered / Population
            - w: Fatal / Population
        @population <int>: population value in the place
        @return <pd.DataFrame>:
            - index: reset index
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
            - Exposed <int>: the number of exposed cases
            - Waiting <int>: the number of waiting cases
        """
        df[cls.S] = df["x1"]
        df[cls.E] = df["x2"]
        df[cls.W] = df["x3"]
        df[cls.C] = df[["y", "z", "w"]].sum(axis=1)
        df[cls.CI] = df["y"]
        df[cls.R] = df["z"]
        df[cls.F] = df["w"]
        df = df.loc[:, [cls.C, cls.CI, cls.F, cls.R, cls.E, cls.W]]
        df = (df * population).astype(np.int64)
        return df

    def calc_r0(self):
        try:
            r0 = self.rho1 * (1 - self.theta) / (self.sigma + self.kappa)
        except ZeroDivisionError:
            return np.nan
        return round(r0, 2)

    def calc_days_dict(self, tau):
        _dict = dict()
        _dict["alpha1 [-]"] = round(self.theta, 3)
        if self.kappa == 0:
            _dict["1/alpha2 [day]"] = 0
        else:
            _dict["1/alpha2 [day]"] = int(tau / 24 / 60 / self.kappa)
        _dict["1/beta1 [day]"] = int(tau / 24 / 60 / self.rho1)
        _dict["1/beta2 [day]"] = int(tau / 24 / 60 / self.rho2)
        _dict["1/beta3 [day]"] = int(tau / 24 / 60 / self.rho3)
        if self.sigma == 0:
            _dict["1/gamma [day]"] = 0
        else:
            _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)
        return _dict
