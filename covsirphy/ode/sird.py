#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIRD(ModelBase):
    NAME = "SIR-D"
    PARAMETERS = ["kappa", "rho", "sigma"]
    DAY_PARAMETERS = ["1/alpha2 [day]", "1/beta [day]", "1/gamma [day]"]
    VARIABLES = ["x", "y", "z", "w"]
    PRIORITIES = np.array([1, 10, 10, 2])
    VARS_INCLEASE = ["z", "w"]

    def __init__(self, kappa, rho, sigma):
        super().__init__()
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma

    def __call__(self, t, X):
        x, y, z, w = X
        y = max(y, 0)
        dxdt = - self.rho * x * y
        # dydt = self.rho * x * y - (self.sigma + self.kappa) * y
        dzdt = self.sigma * y
        dwdt = self.kappa * y
        dydt = 0 - min(dxdt + dzdt + dwdt, y)
        return np.array([dxdt, dydt, dzdt, dwdt])

    @classmethod
    def param(cls, train_df_divided=None, q_range=None):
        param_dict = super().param()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
        if train_df_divided is not None:
            df = train_df_divided.copy()
            # kappa = (dw/dt) / y
            kappa_series = df["w"].diff() / df["t"].diff() / df["y"]
            param_dict["kappa"] = kappa_series.quantile(q_range)
            # rho = - (dx/dt) / x / y
            rho_series = 0 - df["x"].diff() / df["t"].diff() / \
                df["x"] / df["y"]
            param_dict["rho"] = rho_series.quantile(q_range)
            # sigma = (dz/dt) / y
            sigma_series = df["z"].diff() / df["t"].diff() / df["y"]
            param_dict["sigma"] = sigma_series.quantile(q_range)
            return param_dict
        param_dict["kappa"] = (0, 1)
        param_dict["rho"] = (0, 1)
        param_dict["sigma"] = (0, 1)
        return param_dict

    @classmethod
    def calc_variables(cls, cleaned_df, population):
        """
        Calculate the variables of SIR-D model.
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
            - index: reset index
            - x: Susceptible / Population
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
        """
        df[cls.C] = 1 - df["x"]
        df[cls.CI] = df["y"]
        df[cls.R] = df["z"]
        df[cls.F] = df["w"]
        df = df.loc[:, [cls.C, cls.CI, cls.F, cls.R]]
        df = (df * population).astype(np.int64)
        return df

    def calc_r0(self):
        try:
            r0 = self.rho / (self.sigma + self.kappa)
        except ZeroDivisionError:
            return np.nan
        return round(r0, 2)

    def calc_days_dict(self, tau):
        _dict = dict()
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
