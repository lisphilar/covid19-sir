#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIR(ModelBase):
    NAME = "SIR"
    PARAMETERS = ["rho", "sigma"]
    VARIABLES = ["x", "y", "z"]
    PRIORITIES = np.array([1, 1, 1])
    VARS_INCLEASE = ["z"]

    def __init__(self, rho, sigma):
        super().__init__()
        self.rho = rho
        self.sigma = sigma

    def __call__(self, t, X):
        # x, y, z = [X[i] for i in range(len(self.VARIABLES))]
        # dxdt = - self.rho * x * y
        # dydt = self.rho * x * y - self.sigma * y
        # dzdt = self.sigma * y
        dxdt = - self.rho * X[0] * X[1]
        dydt = self.rho * X[0] * X[1] - self.sigma * X[1]
        dzdt = self.sigma * X[1]
        return np.array([dxdt, dydt, dzdt])

    @classmethod
    def param(cls, train_df_divided=None, q_range=None):
        param_dict = super().param()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
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
        Calculate the variables of SIR model.
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
            - z: (Recovered + Fatal) / Population
        """
        df = super().calc_variables(cleaned_df, population)
        df["X"] = population - df[cls.C]
        df["Y"] = df[cls.CI]
        df["Z"] = df[cls.R] + df[cls.F]
        # Columns will be changed to lower cases
        return cls.nondim_cols(df, ["X", "Y", "Z"], population)

    @classmethod
    def calc_variables_reverse(cls, df, population):
        """
        Calculate measurable variables.
        @df <pd.DataFrame>:
            - index: reseted index
            - x: Susceptible / Population
            - y: Infected / Population
            - z: (Recovered + Fatal) / Population
        @population <int>: population value in the place
        @return <pd.DataFrame>:
            - index: reseted index
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal + Recovered <int>:
                the number of fatal or recovered cases
        """
        df[cls.C] = 1 - df["x"]
        df[cls.CI] = df["y"]
        df[cls.FR] = df["z"]
        df = df.loc[:, [cls.C, cls.CI, cls.FR]]
        df = (df * population).astype(np.int64)
        return df

    def calc_r0(self):
        if self.sigma == 0:
            return np.nan
        r0 = self.rho / self.sigma
        return round(r0, 2)

    def calc_days_dict(self, tau):
        _dict = dict()
        _dict["1/beta [day]"] = int(tau / 24 / 60 / self.rho)
        _dict["1/gamma [day]"] = int(tau / 24 / 60 / self.sigma)
        return _dict
