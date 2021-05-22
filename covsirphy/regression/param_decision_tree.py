#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log10, floor
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from covsirphy.regression.regbase import _RegressorBase


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
        delay (int): delay period [days]
        kwargs: keyword arguments of sklearn.model_selection.train_test_split(test_size=0.2, random_state=0)

    Note:
        If @seed is included in kwargs, this will be converted to @random_state.
    """
    # Description of regressor
    DESC = "Indicators -> Parameters with Decision Tree Regressor"

    def __init__(self, X, y, delay, **kwargs):
        super().__init__(X, y, delay, **kwargs)

    def _fit(self):
        """
        Fit regression model with training dataset, update self._regressor and self._param.
        """
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Paramters of the steps
        param_grid = {
            "pca__n_components": [int(v * len(self._X_train.columns)) for v in [0.1, 0.5, 0.9, 1.0]],
            "regressor__max_depth": [3, 5, 7, 9],
        }
        # Fit with pipeline
        steps = [
            ("pca", PCA(random_state=0)),
            ("regressor", DecisionTreeRegressor(random_state=0)),
        ]
        pipeline = GridSearchCV(Pipeline(steps=steps), param_grid, n_jobs=-1, cv=5)
        pipeline.fit(self._X_train, self._y_train)
        # Update regressor
        self._regressor = pipeline
        # Update param
        param_dict = {
            **{k: type(v) for (k, v) in steps},
            "pca_n_components": pipeline.best_estimator_.named_steps.pca.n_components_,
            "intercept": pd.DataFrame(),
            "coef": pd.DataFrame(),
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
