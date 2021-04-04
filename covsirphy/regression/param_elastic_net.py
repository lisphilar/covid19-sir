#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log10, floor
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from covsirphy.regression.regbase import _RegressorBase


class _ParamElasticNetRegressor(_RegressorBase):
    """
    Predict parameter values of ODE models with Elastic Net regression.

    Args:
        X (pandas.DataFrame):
            Index
                Date (pandas.Timestamp): observation date
            Columns
                (int/float): indicators
        y (pandas.DataFrame):
            Index
                Date (pandas.Timestamp): observation date
            Columns
                (int/float) target values
        delay (int): delay period [days]
        kwargs: keyword arguments of sklearn.model_selection.train_test_split(test_size=0.2, random_state=0)

    Note:
        If @seed is included in kwargs, this will be converted to @random_state.
    """
    # Description of regressor
    DESC = "Indicators -> Parameters with Elastic Net"

    def __init__(self, X, y, delay, **kwargs):
        super().__init__(X, y, delay, **kwargs)

    def _fit(self):
        """
        Fit regression model with training dataset, update self._regressor and self._param.
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # Model for Elastic Net regression
        cv = MultiTaskElasticNetCV(
            alphas=[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            l1_ratio=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            cv=5, n_jobs=-1
        )
        # Fit with pipeline
        steps = [
            ("scaler", MinMaxScaler()),
            ("regressor", cv),
        ]
        pipeline = Pipeline(steps=steps)
        pipeline.fit(self._X_train, self._y_train)
        reg_output = pipeline.named_steps.regressor
        # Update regressor
        self._regressor = pipeline
        # Intercept/coef
        intercept_df = pd.DataFrame(
            reg_output.coef_, index=self._y_train.columns, columns=self._X_train.columns)
        intercept_df.insert(0, "Intercept", None)
        intercept_df["Intercept"] = reg_output.intercept_
        # Update param
        param_dict = {
            **{k: type(v) for (k, v) in steps},
            "alpha": reg_output.alpha_,
            "l1_ratio": reg_output.l1_ratio_,
            "intercept": intercept_df,
            "coef": intercept_df,
        }
        self._param.update(param_dict)

    def predict(self):
        """
        Predict parameter values (via y) with self._regressor and X_target.

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): future dates
                Columns
                    (float): parameter values (4 digits)
        """
        # Predict parameter values
        predicted = self._regressor.predict(self._X_target)
        df = pd.DataFrame(predicted, index=self._X_target.index, columns=self._y_train.columns)
        # parameter values: 4 digits
        return df.applymap(lambda x: np.around(x, 4 - int(floor(log10(abs(x)))) - 1))
