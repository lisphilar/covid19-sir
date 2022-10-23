import warnings
import numpy as np
import pandas as pd
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
    """
    # Names
    _A = "_actual"
    _P = "_predicted"
    # Metrics: {name: (function(x1, x2), whether smaller is better or not)}
    _METRICS_DICT = {
        "ME": (lambda x1, x2: np.max(np.abs(x2 - x1)), True),
        "MAE": (lambda x1, x2: np.mean(np.abs(x2 - x1)), True),
        "MSE": (lambda x1, x2: np.mean(np.square(x2 - x1)), True),
        "MSLE": (lambda x1, x2: np.mean(np.square(np.log1p(x2) - np.log1p(x1))), True),
        "MAPE": (lambda x1, x2: np.mean(np.abs((x2 - x1) / x1)) * 100, True),
        "RMSE": (lambda x1, x2: np.sqrt(np.mean(np.square(x2 - x1))), True),
        "RMSLE": (lambda x1, x2: np.sqrt(np.mean(np.square(np.log1p(x2) - np.log1p(x1)))), True),
        "R2": (lambda x1, x2: np.corrcoef(x1, x2)[0, 1]**2, False),
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

    def score(self, metric="RMSLE"):
        """
        Calculate score with specified metric.

        Args:
            metric (str): ME, MAE, MSE, MSLE, MAPE, RMSE, RMSLE, R2

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
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Check metric name
        if metric.upper() not in self._METRICS_DICT:
            raise UnExpectedValueError("metric", metric, candidates=list(self._METRICS_DICT.keys()))
        # Calculate score
        func = self._METRICS_DICT[metric.upper()][0]
        try:
            return float(func(self._true.values, self._pred.values))
        except TypeError:
            outputs = float(func(self._true.values.astype(np.float64), self._pred.values.astype(np.float64)))
            return float(np.average(outputs))

    @classmethod
    def metrics(cls):
        """
        Return the list of metric names.

        Returns:
            list[str]: list of metric names
        """
        return list(cls._METRICS_DICT.keys())

    @classmethod
    def smaller_is_better(cls, metric="RMSLE"):
        """
        Whether smaller value of the metric is better or not.

        Args:
            metric (str): ME, MAE, MSE, MSLE, MAPE, RMSE, RMSLE, R2

        Returns:
            bool: whether smaller value is better or not
        """
        if metric.upper() not in cls._METRICS_DICT:
            raise UnExpectedValueError("metric", metric, candidates=list(cls._METRICS_DICT.keys()))
        return cls._METRICS_DICT[metric.upper()][1]

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
