#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.metrics
from covsirphy.util.error import UnExpectedValueError


class Evaluator(object):
    """
    Evaluate residual errors.

    Args:
        y_true (pandas.DataFrame or pandas.Series): correct target values
        y_pred (pandas.DataFrame or pandas.Series): estimated target values
        on (str or list[str] or None): column names to join on or None (join on index)

    Raises:
        TypeError: un-expected types were used for the arguments

    Note:
        Evaluation with metrics will be done with sklearn.metrics package.
        https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    # Names
    _A = "_actual"
    _P = "_predicted"
    # Metrics
    _METRICS_DICT = {
        "ME": sklearn.metrics.max_error,
        "MAE": sklearn.metrics.mean_absolute_error,
        "MSE": sklearn.metrics.mean_squared_error,
        "MSLE": sklearn.metrics.mean_squared_log_error,
        "MAPE": sklearn.metrics.mean_absolute_percentage_error,
        "RMSE": lambda x1, x2: sklearn.metrics.mean_squared_error(x1, x2, squared=False),
        "RMSLE": lambda x1, x2: np.sqrt(sklearn.metrics.mean_squared_log_error(x1, x2)),
        "R2": sklearn.metrics.r2_score,
    }

    def __init__(self, y_true, y_pred, on=None):
        # Check types
        for (y, name) in zip([y_true, y_pred], ["correct", "estimated"]):
            if not isinstance(y, (pd.DataFrame, pd.Series)):
                raise TypeError(f"@{name} must be an instance of pandas.DataFrame or pandas.Series.")
        # Join dataframes
        true_df, pred_df = pd.DataFrame(y_true), pd.DataFrame(y_pred)
        if on is not None:
            true_df = true_df.set_index(on)
            pred_df = pred_df.set_index(on)
        all_df = true_df.join(pred_df, how="inner", lsuffix=self._A, rsuffix=self._P)
        # Register values
        self._true = all_df.loc[:, [f"{col}{self._A}" for col in true_df.columns]]
        self._pred = all_df.loc[:, [f"{col}{self._P}" for col in pred_df.columns]]

    def score(self, metric=None, metrics="RMSLE"):
        """
        Calculate score with specified metric.

        Args:
            metric (str): ME, MAE, MSE, MSLE, MAPE, RMSE, RMSLE, R2
            metrics (str): alias od @metric

        Raises:
            UnExpectedValueError: un-expected metric was applied
            ValueError: ME was selected as metric when the targets have multiple columns

        Returns:
            float: score with the metric

        Note:
            ME: maximum residual error
            MAE: mean absolute error
            MSE: mean square error
            MSLE: mean squared logarithmic error
            MAPE: mean absolute percentage error
            RMSE: root mean squared error
            RMSLE: root mean squared logarithmic error
            R2: the coefficient of determination

        Note:
            When @metric is None, @metrics will be used as @metric. Default value is "RMSLE".
        """
        metric = metric or metrics
        # Check metric name
        if metric not in self._METRICS_DICT:
            raise UnExpectedValueError("metric", metric, candidates=list(self._METRICS_DICT.keys()))
        # Calculate score
        try:
            return float(self._METRICS_DICT[metric](self._true, self._pred))
        except ValueError:
            # Multioutput not supported in max_error
            raise ValueError(
                "When the targets have multiple columns, we cannot select ME (max residual error).") from None
