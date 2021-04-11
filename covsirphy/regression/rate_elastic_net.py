#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log10, floor
import numpy as np
import pandas as pd
from covsirphy.regression.param_elastic_net import _ParamElasticNetRegressor


class _RateElasticNetRegressor(_ParamElasticNetRegressor):
    """
    Predict parameter values of ODE models with Elastic Net regression
    and Indicators(n)/Indicators(n-1) -> Parameters(n)/Parameters(n-1) approach.

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
    DESC = "Indicators(n)/Indicators(n-1) -> Parameters(n)/Parameters(n-1) with Elastic Net"

    def __init__(self, X, y, delay, **kwargs):
        # Remember the last value of y (= the previous value of target y)
        self._last_param_df = y.tail(1)
        # Calculate X(n) / X(n-1) and replace inf/NA with 0
        X_div = X.div(X.shift(1)).replace(np.inf, 0).fillna(0)
        # Calculate y(n) / y(n-1) and replace inf with NAs (NAs will be removed in ._split())
        y_div = y.div(y.shift(1)).replace(np.inf, np.nan)
        super().__init__(X_div, y_div, delay, **kwargs)

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
        # Calculate y(n) values with y(0) and y(n) / y(n-1)
        df = pd.concat([self._last_param_df, df], axis=0, sort=True)
        df = df.cumprod().iloc[1:]
        # parameter values: 4 digits
        return df.applymap(lambda x: np.around(x, 4 - int(floor(log10(abs(x)))) - 1))
