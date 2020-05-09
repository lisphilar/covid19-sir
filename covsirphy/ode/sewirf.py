#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SEWIRF(ModelBase):
    NAME = "SEWIR-F"
    VARIABLES = ["x1", "x2", "x3", "y", "z", "w"]
    PRIORITIES = np.array([0, 0, 0, 10, 10, 2])
    MONOTONIC = ["z", "w"]

    def __init__(self, theta, kappa, rho1, rho2, rho3, sigma):
        super().__init__()
        self.theta = theta
        self.kappa = kappa
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.sigma = sigma

    def __call__(self, t, X):
        # x1, x2, x3, y, z, w = [X[i] for i in range(len(self.VARIABLES))]
        # dx1dt = - self.rho1 * x1 * (x3 + y)
        # dx2dt = self.rho1 * x1 * (x3 + y) - self.rho2 * x2
        # dx3dt = self.rho2 * x2 - self.rho3 * x3
        # dydt = self.rho3 * (1 - self.theta) * x3 - (self.sigma + self.kappa) * y
        # dzdt = self.sigma * y
        # dwdt = self.rho3 * self.theta * x3 + self.kappa * y
        dx1dt = - self.rho1 * X[0] * (X[2] + X[3])
        dx2dt = self.rho1 * X[0] * (X[2] + X[3]) - self.rho2 * X[1]
        dx3dt = self.rho2 * X[1] - self.rho3 * X[2]
        dydt = self.rho3 * (1 - self.theta) * \
            X[2] - (self.sigma + self.kappa) * X[3]
        dzdt = self.sigma * X[3]
        dwdt = self.rho3 * self.theta * X[2] + self.kappa * X[3]
        return np.array([dx1dt, dx2dt, dx3dt, dydt, dzdt, dwdt])

    @classmethod
    def param_dict(cls, train_df_divided=None, q_range=None):
        param_dict = super().param_dict()
        q_range = super().QUANTILE_RANGE[:] if q_range is None else q_range
        param_dict["theta"] = (0, 1)
        param_dict["kappa"] = (0, 1)
        param_dict["rho1"] = (0, 1)
        param_dict["rho2"] = (0, 1)
        param_dict["rho3"] = (0, 1)
        if train_df_divided is not None:
            df = train_df_divided.copy()
            # sigma = (dz/dt) / y
            sigma_series = df["z"].diff() / df["t"].diff() / df["y"]
            param_dict["sigma"] = sigma_series.quantile(q_range)
            return param_dict
        param_dict["sigma"] = (0, 1)
        return param_dict

    @staticmethod
    def calc_variables(df):
        df["X1"] = df["Susceptible"]
        df["X2"] = 0
        df["X3"] = 0
        df["Y"] = df["Infected"]
        df["Z"] = df["Recovered"]
        df["W"] = df["Fatal"]
        return df.loc[:, ["T", "X1", "X2", "X3", "Y", "Z", "W"]]

    @staticmethod
    def calc_variables_reverse(df, total_population):
        df["Susceptible"] = df["X1"]
        df["Infected"] = df["Y"]
        df["Recovered"] = df["Z"]
        df["Fatal"] = df["W"]
        df["Exposed"] = df["X2"]
        df["Waiting"] = df["X3"]
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
