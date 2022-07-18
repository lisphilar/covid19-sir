#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from covsirphy._deprecated.regbase import _RegressorBase
from covsirphy._deprecated.reg_rate_converter import _RateConverter
from covsirphy._deprecated.reg_feature_selector import _FeatureSelector


class _ParamElasticNetRegressor(_RegressorBase):
    """
    Predict parameter values of ODE models with Elastic Net regression.

    Args:
        - X_train (pandas.DataFrame): X for training with time index
        - X_test (pandas.DataFrame): X for test with time index
        - Y_train (pandas.DataFrame): Y for training with time index
        - Y_test (pandas.DataFrame): Y for test with time index
        - X_target (pandas.DataFrame): X for prediction with time index
    """
    # Description of regressor
    DESC = "Indicators -> Parameters with Elastic Net"

    def _fit(self):
        """
        Fit regression model with training dataset, update self._pipeline and self._param.
        """
        # Model for Elastic Net regression
        tscv = TimeSeriesSplit(n_splits=5).split(self._X_train)
        cv = MultiTaskElasticNetCV(
            alphas=[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            l1_ratio=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            cv=tscv, n_jobs=-1
        )
        # Fit with pipeline
        steps = [
            ("converter", _RateConverter(to_convert=False)),
            ("scaler", MinMaxScaler()),
            ("selector", _FeatureSelector(corr_threshold=0.8)),
            ("regressor", cv),
        ]
        pipeline = Pipeline(steps=steps)
        pipeline.fit(self._X_train, self._Y_train)
        reg_output = pipeline.named_steps.regressor
        # Update regressor
        self._pipeline = pipeline
        # Intercept/coef
        intercept_df = pd.DataFrame(
            reg_output.coef_, index=self._Y_train.columns, columns=self._X_train.columns)
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
