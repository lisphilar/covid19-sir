#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from covsirphy.regression.regbase import _RegressorBase
from covsirphy.regression.reg_rate_converter import _RateConverter


class _ParamSVRegressor(_RegressorBase):
    """
    Predict parameter values of ODE models with epsilon-support vector regressor.

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
        delay_values (list[int]): list of delay period [days]
        kwargs: keyword arguments of sklearn.model_selection.train_test_split()

    Note:
        If @seed is included in kwargs, this will be converted to @random_state.

    Note:
        default values regarding sklearn.model_selection.train_test_split() are
        test_size=0.2, random_state=0, shuffle=False.
    """
    # Description of regressor
    DESC = "Indicators -> Parameters with Epsilon-Support Vector Regressor"

    def __init__(self, X, y, delay_values, **kwargs):
        super().__init__(X, y, delay_values, **kwargs)

    def _fit(self):
        """
        Fit regression model with training dataset, update self._pipeline and self._param.
        """
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Paramters of the steps
        param_grid = {
            "converter__to_convert": [True, False],
            "regressor__estimator__kernel": ["linear", "rbf"],
            "regressor__estimator__C": [1, 10, 100],
            "regressor__estimator__gamma": [0.01, 0.1],
            "regressor__estimator__epsilon": [0.001, 0.01, 0.02, 0.05],
        }
        # Fit with pipeline
        steps = [
            ("converter", _RateConverter()),
            ("scaler", MinMaxScaler()),
            ("regressor", MultiOutputRegressor(SVR())),
        ]
        tscv = TimeSeriesSplit(n_splits=5).split(self._X_train)
        pipeline = RandomizedSearchCV(Pipeline(steps=steps), param_grid, n_jobs=-1, cv=tscv, n_iter=10)
        pipeline.fit(self._X_train, self._y_train)
        # Update regressor
        self._pipeline = pipeline
        # Update param: pipeline.best_estimator_.named_steps.regressor.estimators_
        self._param.update(**{k: type(v) for (k, v) in steps})
