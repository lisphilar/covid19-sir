#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIRFV(ModelBase):
    NAME = "SIR-FV"
    PARAMETERS = ["theta", "kappa", "rho", "sigma", "omega"]
    DAY_PARAMETERS = [
        "alpha1 [-]", "1/alpha2 [day]", "1/beta [day]", "1/gamma [day]"
    ]
    VARIABLES = ["x", "y", "z", "w"]
    PRIORITIES = np.array([1, 10, 10, 2])
    VARS_INCLEASE = ["z", "w"]

    def __init__(self, theta, kappa, rho, sigma, omega=None, n=None, v_per_day=None):
        """
        (n and v_per_day) or omega must be applied.
        @n <float or int>: total population
        @v_par_day <float or int>: vacctinated persons per day
        """
        super().__init__()
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma
        if omega is None:
            try:
                self.omega = float(v_per_day) / float(n)
            except TypeError:
                s = "Neither (n and va_per_day) nor omega must be applied!"
                raise TypeError(s)
        else:
            self.omega = float(omega)

    def __call__(self, t, X):
        # x, y, z, w = [X[i] for i in range(len(self.VARIABLES))]
        # x with vacctination
        dxdt = - self.rho * X[0] * X[1] - self.omega
        dxdt = 0 - X[0] if X[0] + dxdt < 0 else dxdt
        # y, z, w
        dydt = self.rho * (1 - self.theta) * \
            X[0] * X[1] - (self.sigma + self.kappa) * X[1]
        dzdt = self.sigma * X[1]
        dwdt = self.rho * self.theta * X[0] * X[1] + self.kappa * X[1]
        return np.array([dxdt, dydt, dzdt, dwdt])

    @classmethod
    def param(cls, train_df_divided=None, q_range=None):
        param_dict = super().param()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
        param_dict["theta"] = (0, 1)
        param_dict["kappa"] = (0, 1)
        param_dict["omega"] = (0, 1)
        if train_df_divided is not None:
            df = train_df_divided.copy()
            # rho = - (dx/dt) / x / y
            rho_series = 0 - df["x"].diff() / df["t"].diff() / \
                df["x"] / df["y"]
            param_dict["rho"] = rho_series.quantile(q_range)
            # sigma = (dz/dt) / y
            sigma_series = df["z"].diff() / df["t"].diff() / df["y"]
            param_dict["sigma"] = sigma_series.quantile(q_range)
            return param_dict
        param_dict["rho"] = (0, 1)
        param_dict["sigma"] = (0, 1)
        return param_dict

    @classmethod
    def calc_variables(cls, cleaned_df, population):
        """
        Calculate the variables of SIR-FV model.
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
            - x: Susceptible / Population
            - y: Infected / Population
            - z: Recovered / Population
            - w: Fatal / Population
        """
        df = super().calc_variables(cleaned_df, population)
        df["X"] = population - df[cls.C]
        df["Y"] = df[cls.CI]
        df["Z"] = df[cls.R]
        df["W"] = df[cls.F]
        # Columns will be changed to lower cases
        return cls.nondim_cols(df, ["X", "Y", "Z", "W"], population)

    @classmethod
    def calc_variables_reverse(cls, df, population):
        """
        Calculate measurable variables.
        @df <pd.DataFrame>:
            - index: reseted index
            - x: Susceptible / Population
            - y: Infected / Population
            - z: Recovered / Population
            - w: Fatal / Population
        @population <int>: population value in the place
        @return <pd.DataFrame>:
            - index: reseted index
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
            - Vacctinated <int>: the number of vacctinated cases
        """
        df[cls.S] = df["x"]
        df[cls.C] = df[["y", "z", "w"]].sum(axis=1)
        df[cls.CI] = df["y"]
        df[cls.R] = df["z"]
        df[cls.F] = df["w"]
        df[cls.V] = 1 - df[["x", "y", "z", "w"]].sum(axis=1)
        df = df.loc[:, [cls.C, cls.CI, cls.F, cls.R, cls.V]]
        df = (df * population).astype(np.int64)
        return df

    def calc_r0(self):
        try:
            r0 = self.rho * (1 - self.theta) / (self.sigma + self.kappa)
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
        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)
        if self.sigma == 0:
            _dict["1/gamma [day]"] = 0
        else:
            _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)
        return _dict
