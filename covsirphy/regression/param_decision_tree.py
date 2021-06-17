#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from covsirphy.regression.regbase import _RegressorBase
from covsirphy.regression.reg_rate_converter import _RateConverter


class _ParamDecisionTreeRegressor(_RegressorBase):
    """
    Predict parameter values of ODE models with decision tree regressor.

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
        pipeline.fit(self._X_train, self._y_train)
        # Update regressor
        self._pipeline = pipeline
        # Update param
        self._param.update(**{k: type(v) for (k, v) in steps})
