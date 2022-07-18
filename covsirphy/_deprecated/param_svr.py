#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optuna.distributions import CategoricalDistribution, LogUniformDistribution
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from covsirphy._deprecated.regbase import _RegressorBase
from covsirphy._deprecated.reg_rate_converter import _RateConverter
from covsirphy._deprecated.reg_feature_selector import _FeatureSelector


class _ParamSVRegressor(_RegressorBase):
    """
    Predict parameter values of ODE models with epsilon-support vector regressor.

    Args:
        - X_train (pandas.DataFrame): X for training with time index
        - X_test (pandas.DataFrame): X for test with time index
        - Y_train (pandas.DataFrame): Y for training with time index
        - Y_test (pandas.DataFrame): Y for test with time index
        - X_target (pandas.DataFrame): X for prediction with time index
    """
    # Description of regressor
    DESC = "Indicators -> Parameters with Epsilon-Support Vector Regressor"

    def _fit(self):
        """
        Fit regression model with training dataset, update self._pipeline and self._param.
        """
        # Paramters of the steps
        param_grid = {
            "converter__to_convert": CategoricalDistribution([True, False]),
            "regressor__estimator__kernel": CategoricalDistribution(["linear", "rbf"]),
            "regressor__estimator__C": LogUniformDistribution(2 ** (-10), 2 ** 10),
            "regressor__estimator__gamma": LogUniformDistribution(2 ** (-10), 1),
            "regressor__estimator__epsilon": LogUniformDistribution(2 ** (-20), 1),
        }
        # Fit with pipeline
        steps = [
            ("converter", _RateConverter()),
            ("scaler", MinMaxScaler()),
            ("selector", _FeatureSelector(corr_threshold=0.8)),
            ("regressor", MultiOutputRegressor(SVR())),
        ]
        tscv = TimeSeriesSplit(n_splits=5).split(self._X_train)
        pipeline = OptunaSearchCV(
            Pipeline(steps=steps), param_grid, cv=tscv, random_state=0, n_trials=10)
        pipeline.fit(self._X_train, self._Y_train)
        # Update regressor
        self._pipeline = pipeline
        # Update param: pipeline.best_estimator_.named_steps.regressor.estimators_
        self._param.update(**{k: type(v) for (k, v) in steps})
