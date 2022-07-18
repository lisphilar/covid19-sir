#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from covsirphy._deprecated.regbase import _RegressorBase
from covsirphy._deprecated.reg_rate_converter import _RateConverter


class _ParamDecisionTreeRegressor(_RegressorBase):
    """
    Predict parameter values of ODE models with decision tree regressor.

    Args:
        - X_train (pandas.DataFrame): X for training with time index
        - X_test (pandas.DataFrame): X for test with time index
        - Y_train (pandas.DataFrame): Y for training with time index
        - Y_test (pandas.DataFrame): Y for test with time index
        - X_target (pandas.DataFrame): X for prediction with time index
    """
    # Description of regressor
    DESC = "Indicators -> Parameters with Decision Tree Regressor"

    def _fit(self):
        """
        Fit regression model with training dataset, update self._pipeline and self._param.
        """
        # Paramters of the steps
        param_grid = {
            "converter__to_convert": [True, False],
            "pca__n_components": [0.3, 0.5, 0.7, 0.9],
            "regressor__max_depth": list(range(1, 10)),
        }
        # Fit with pipeline
        steps = [
            ("converter", _RateConverter()),
            ("scaler", MinMaxScaler()),
            ("pca", PCA(random_state=0)),
            ("regressor", DecisionTreeRegressor(random_state=0)),
        ]
        tscv = TimeSeriesSplit(n_splits=5).split(self._X_train)
        pipeline = GridSearchCV(Pipeline(steps=steps), param_grid, n_jobs=-1, cv=tscv)
        pipeline.fit(self._X_train, self._Y_train)
        # Update regressor
        self._pipeline = pipeline
        # Update param
        self._param.update(**{k: type(v) for (k, v) in steps})
