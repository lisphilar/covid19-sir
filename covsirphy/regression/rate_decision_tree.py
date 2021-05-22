#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.regression.param_decision_tree import _ParamDecisionTreeRegressor
from covsirphy.regression.rate_elastic_net import _RateElasticNetRegressor


class _RateDecisionTreeRegressor(_ParamDecisionTreeRegressor, _RateElasticNetRegressor):
    """
    Predict parameter values of ODE models with decision tree regressor,
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
    DESC = "Indicators(n)/Indicators(n-1) -> Parameters(n)/Parameters(n-1) with Decision Tree Regressor"

    def __init__(self, X, y, delay, **kwargs):
        _RateElasticNetRegressor.__init__(self, X, y, delay, **kwargs)

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
        return _RateElasticNetRegressor.predict(self)
