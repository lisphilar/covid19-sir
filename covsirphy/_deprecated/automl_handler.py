#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from itertools import product
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated._mbase import ModelBase
from covsirphy.dynamics.ode import ODEModel
from covsirphy._deprecated.autots_predictor import _AutoTSPredictor


class AutoMLHandler(Term):
    """
    Predict ODE parameter values automatically with machine learning.

    Args:
        X (pandas.DataFrame):
            Index
                pandas.Timestamp: Observation date
            Columns
                observed variables (int or float)
        Y (pandas.DataFrame):
            Index
                pandas.Timestamp: Observation date
            Columns
                observed ODE parameter values (float)
        model (covsirphy.ModelBase or covsirphy.ODEModel): ODE model
        days (int): days to predict
        kwargs: keyword arguments of autots.AutoTS()

    Note:
        When X is a empty dataframe, only "univariate" can be used as @method with AutoHandler.predict().
    """
    _LIKELY = "Likely"
    _UPPER = "Upper"
    _LOWER = "Lower"

    @deprecate(old="AutoMLHandler()", new="ODEScenario().predict()", version="2.24.0-sigma")
    def __init__(self, X, Y, model, days, **kwargs):
        self._model = Validator(model, "model").subclass(parent=(ModelBase, ODEModel))
        if issubclass(self._model, ModelBase):
            self._parameters = self._model.PARAMETERS[:]
        else:
            self._parameters = self._model._PARAMETERS[:]
        self._X = Validator(X, "X").dataframe(time_index=True, empty_ok=True)
        self._Y = Validator(Y, "Y").dataframe(time_index=True, empty_ok=False, columns=self._parameters)
        self._days = Validator(days, name="days").int(value_range=(1, None))
        self._kwargs = kwargs.copy()
        self._pred_df = pd.DataFrame(columns=[self.SERIES, self.DATE, *Y.columns.tolist()])

    def predict(self, method):
        """
        Perform automated machine learning to predict values.

        Args:
            method (str): machine learning method name, "univariate" or "multivariate_regression"

        Returns:
            AutoMLHandler: self

        Note:
            Models used by "univariate" can be checked with from autots.models.model_list import model_lists; model_list["univariate"].

        Note:
            Model used by "multivariate_regression" is Multivariate Regression.
        """
        method_dict = {
            "univariate": self._univariate,
            "multivariate_regression": self._multivariate_regression,
        }
        if method not in method_dict:
            raise KeyError(
                f"Un-expected method: {method}. Supported methods are {', '.join(list(method_dict.keys()))}.")
        self._register_scenarios(method, *method_dict[method]())
        return self

    def summary(self):
        """
        Create and summarize the scenarios.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Scenario (str): scenario name, "Univariate_Likely", "Univariate_1" etc.
                    - Start (pandas.Timestamp): start date of the phase
                    - End (pandas.Timestamp): end date of the phase
                    - Rt (float): phase-dependent reproduction number
                    - columns of Y data

        Note:
            "Univariate_Likely" scenario is the most likely scenario when univariate forecasting is used.

        Note:
            "Univariate_01" scenario is the created with upper values of ODE parameter values.

        Note:
            "Univariate_16" scenario is the created with lower values of ODE parameter values. (if the model has four parameters)

        Note:
            Dates with the same Rt values at the 1st decimal place will be merged to one phase.
        """
        df = self._pred_df.copy()
        # Calculate reproduction number to create phases
        df["param"] = df[self._parameters].to_dict(orient="records")
        if issubclass(self._model, ModelBase):
            df[self.RT] = df.apply(lambda x: self._model(population=100, **x["param"]).calc_r0(), axis=1).round(1)
        else:
            model = self._model
            today = datetime.now()
            model_dict = {
                "date_range": (today, today + timedelta(days=10)),
                "tau": 1440,
                "initial_dict": model._SAMPLE_DICT["initial_dict"]
            }
            df[self.RT] = df[self._parameters].apply(lambda x: model(
                param_dict=x.to_dict(), **model_dict).r0(), axis=1).round(1)
        # Get start/end date
        criteria = [self.SERIES, self.RT]
        df = df.groupby(criteria).first().join(df[[*criteria, self.DATE]].groupby(criteria).last(), rsuffix="_last")
        df = df.rename(columns={self.DATE: self.START, f"{self.DATE}_last": self.END})
        df = df.reset_index().loc[:, [self.SERIES, self.START, self.END, self.RT, *self._parameters]]
        return df.sort_values([self.SERIES, self.START], ignore_index=True)

    def _register_scenarios(self, method, likely_df, upper_df, lower_df):
        """
        Create and register scenario with the most likely values, upper values and lower values.

        Args:
            method (str): machine learning method name
            likely_df (pandas.DataFrame): the most likely values with a forecasting method
                Index
                    Date (pandas.Timestamp): observation date
                Columns
                    predicted values (float)
            upper_df (pandas.DataFrame): the upper values with a forecasting method
                Index
                    Date (pandas.Timestamp): observation date
                Columns
                    predicted values (float)
            lower_df (pandas.DataFrame): the lower values with a forecasting method
                Index
                    Date (pandas.Timestamp): observation date
                Columns
                    predicted values (float)
        """
        # The most likely scenario
        df = likely_df.loc[:, self._Y.columns]
        df.index.name = self.DATE
        df = df.reset_index()
        df[self.SERIES] = f"{method.capitalize()}_{self._LIKELY}"
        dataframes = [df]
        # Upper/Lower
        ul_df = upper_df.loc[:, self._Y.columns].join(
            lower_df.loc[:, self._Y.columns], lsuffix=f"_{self._UPPER}", rsuffix=f"_{self._LOWER}")
        col_products = product(
            *([f"{param}_{suffix}" for suffix in (self._UPPER, self._LOWER)] for param in self._Y.columns))
        for (i, col_product) in enumerate(col_products):
            df = ul_df.loc[:, col_product]
            df.rename(
                columns=lambda x: x.replace(self._UPPER, "").replace(self._LOWER, "").replace("_", ""), inplace=True)
            df[self.SERIES] = f"{method.capitalize()}_{i:02}"
            df.index.name = self.DATE
            dataframes.append(df.reset_index())
        self._pred_df = pd.concat([self._pred_df, *dataframes], axis=0)

    def _univariate(self):
        """
        Perform univariate forecasting of Y without X.

        Returns:
            tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame): the most likely, upper, lower values
                Index
                    reset index
                Columns
                    predicted values (float)
        """
        predictor = _AutoTSPredictor(days=self._days, model_list="univariate")
        return predictor.predict(self._Y)

    def _multivariate_regression(self):
        """
        Perform multivariate forecasting (Regression with both of X(i-1) and Y(i-1)).

        Returns:
            tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame): the most likely, upper, lower values
                Index
                    reset index
                Columns
                    predicted values (float)
        """
        predictor = _AutoTSPredictor(days=self._days, model_list=["MultivariateRegression"])
        return predictor.predict(self._Y.join(self._X))
