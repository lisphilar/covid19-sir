#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.metrics
from covsirphy.util.error import UnExpectedValueError, NAFoundError
from covsirphy.util.validator import Validator


class Evaluator(object):
    """
    Evaluate residual errors.

    Args:
        y_true (pandas.DataFrame or pandas.Series): correct target values
        y_pred (pandas.DataFrame or pandas.Series): estimated target values
        how (str): "all" (use all records) or "inner" (intersection will be used)
        on (str or list[str] or None): column names to join on or None (join on index)

    Raises:
        NAFoundError: either @y_true or @pred has NA values

    Note:
        Evaluation with metrics will be done with sklearn.metrics package.
        https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    # Names
    _A = "_actual"
    _P = "_predicted"
    # Metrics: {name: (function(x1, x2), whether smaller is better or not)}
    _METRICS_DICT = {
        "ME": (sklearn.metrics.max_error, True),
        "MAE": (sklearn.metrics.mean_absolute_error, True),
        "MSE": (sklearn.metrics.mean_squared_error, True),
        "MSLE": (sklearn.metrics.mean_squared_log_error, True),
        "MAPE": (sklearn.metrics.mean_absolute_percentage_error, True),
        "RMSE": (lambda x1, x2: sklearn.metrics.mean_squared_error(x1, x2, squared=False), True),
        "RMSLE": (lambda x1, x2: np.sqrt(sklearn.metrics.mean_squared_log_error(x1, x2)), True),
        "R2": (sklearn.metrics.r2_score, False),
    }

    def __init__(self, y_true, y_pred, how="inner", on=None):
        # Check types
        for (y, name) in zip([y_true, y_pred], ["y_true", "y_pred"]):
            Validator(y, name, accept_none=False).instance(expected=(pd.DataFrame, pd.Series, list, tuple))
            if pd.DataFrame(y).isna().any().any():
                raise NAFoundError(name, y)
        # Join dataframes
        true_df, pred_df = pd.DataFrame(y_true), pd.DataFrame(y_pred)
        if how == "all":
            self._true, self._pred = true_df, pred_df
            return
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
            metric (str or None): ME, MAE, MSE, MSLE, MAPE, RMSE, RMSLE, R2 or None (use @metrics)
            metrics (str): alias of @metric

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
        metric = (metric or metrics).upper()
        # Check metric name
        if metric not in self._METRICS_DICT:
            raise UnExpectedValueError("metric", metric, candidates=list(self._METRICS_DICT.keys()))
        # Calculate score
        try:
            return float(self._METRICS_DICT[metric][0](self._true.values, self._pred.values))
        except ValueError:
            raise ValueError(
                f"When the targets have multiple columns or negative values, we cannot select {metric}.") from None

    @classmethod
    def metrics(cls):
        """
        Return the list of metric names.

        Returns:
            list[str]: list of metric names
        """
        return list(cls._METRICS_DICT.keys())

    @classmethod
    def smaller_is_better(cls, metric=None, metrics="RMSLE"):
        """
        Whether smaller value of the metric is better or not.

        Args:
            metric (str or None): ME, MAE, MSE, MSLE, MAPE, RMSE, RMSLE, R2 or None (use @metrics)
            metrics (str): alias of @metric

        Returns:
            bool: whether smaller value is better or not
        """
        metric = (metric or metrics).upper()
        # Check metric name
        if metric not in cls._METRICS_DICT:
            raise UnExpectedValueError("metric", metric, candidates=list(cls._METRICS_DICT.keys()))
        return cls._METRICS_DICT[metric][1]

    @classmethod
    def best_one(cls, candidate_dict, **kwargs):
        """
        Select the best one with scores.

        Args:
            candidate_dict (dict[object, float]): scores of candidates
            kwargs: keyword arguments of Evaluator.smaller_is_better()

        Returns:
            tuple(object, float): the best one and its score
        """
        comp_f = {True: min, False: max}[cls.smaller_is_better(**kwargs)]
        return comp_f(candidate_dict.items(), key=lambda x: x[1])
