#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log10, floor
import warnings
import numpy as np
from optuna.exceptions import ExperimentalWarning
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.term import Term
from covsirphy._deprecated.reg_pred_actual_plot import _PredActualPlot


class _RegressorBase(Term):
    """
    Basic class to predict parameter values of ODE models.

    Args:
        - X_train (pandas.DataFrame): X for training with time index
        - X_test (pandas.DataFrame): X for test with time index
        - Y_train (pandas.DataFrame): Y for training with time index
        - Y_test (pandas.DataFrame): Y for test with time index
        - X_target (pandas.DataFrame): X for prediction with time index
    """
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    warnings.simplefilter("ignore", category=ExperimentalWarning)
    # Description of regressor
    DESC = ""

    def __init__(self, X_train, X_test, Y_train, Y_test, X_target):
        # Datasets
        self._X_train, self._X_test = X_train.copy(), X_test.copy()
        self._Y_train, self._Y_test = Y_train.copy(), Y_test.copy()
        self._X_target = X_target.copy()
        # Regression model
        self._pipeline = None
        self._param = {}
        # Perform fitting
        self._fit()

    def _fit(self):
        """
        Fit regression model with training dataset, update self._pipeline and self._param.
        This method should be overwritten by child classes.
        """
        raise NotImplementedError

    def score_train(self, metric):
        """
        Calculate score with training dataset.

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()

        Returns:
            float: evaluation score
        """
        pred_train = pd.DataFrame(self._pipeline.predict(self._X_train), columns=self._Y_train.columns)
        return Evaluator(pred_train, self._Y_train, how="all").score(metric=metric)

    def score_test(self, metric):
        """
        Calculate score with test dataset.

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()

        Returns:
            float: evaluation score
        """
        pred_test = pd.DataFrame(self._pipeline.predict(self._X_test), columns=self._Y_test.columns)
        return Evaluator(pred_test, self._Y_test, how="all").score(metric=metric)

    def to_dict(self, metric):
        """
        Calculate score with training/test dataset.

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()

        Returns:
            dict(str, object)
        """
        try:
            param_dict = {
                **self._pipeline.best_params_,
                "intercept": pd.DataFrame(),
                "coef": pd.DataFrame(),
            }
        except AttributeError:
            param_dict = {}
        param_dict.update(self._param)
        return {
            **param_dict,
            "score_metric": metric,
            "score_train": self.score_train(metric=metric),
            "score_test": self.score_test(metric=metric),
            "dataset": {
                "X_train": self._X_train,
                "y_train": self._Y_train,
                "X_test": self._X_test,
                "y_test": self._Y_test,
                "X_target": self._X_target,
            }
        }

    def predict(self):
        """
        Predict parameter values (via Y) with self._pipeline and X_target.

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): future dates
                Columns
                    (float): parameter values (4 digits)
        """
        # Predict parameter values
        predicted = self._pipeline.predict(self._X_target)
        df = pd.DataFrame(predicted, index=self._X_target.index, columns=self._Y_train.columns)
        # parameter values: 4 digits
        return df.applymap(lambda x: self._round(x, digits=4))

    def _round(self, value, digits):
        """
        Round off the value to @digits significant digits.

        Args:
            value (float): target value
            digists (int): significant digits

        Return:
            float: rounded value
        """
        return 0 if not value else np.around(value, digits - int(floor(log10(abs(value)))) - 1)

    def _float2str(self, value, to_percentage):
        """
        Convert a float to a string.

        Args:
            value (float): target value
            to_percentage (bool): whether show the value as a percentage or not

        Returns:
            str: the string
        """
        return f"{value:.2%}" if to_percentage else f"{value:.3g}"

    def pred_actual_plot(self, metric, filename=None):
        """
        Create a scatter plot (predicted vs. actual parameter values).

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()
            fileaname (str): filename of the figure or None (display)
        """
        TITLE = f"Predicted vs. actual parameter values\n{self.DESC}"
        PRED, ACTUAL = "Predicted values", "Actual values"
        # Scores
        train_score = self.score_train(metric=metric)
        test_score = self.score_test(metric=metric)
        # Legend
        to_percentage = metric.upper() == "MAPE"
        train_score_str = self._float2str(train_score, to_percentage=to_percentage)
        test_score_str = self._float2str(test_score, to_percentage=to_percentage)
        train_title = f"Training data (n={len(self._X_train)}, {metric}={train_score_str})"
        test_title = f"Test data (n={len(self._X_test)}, {metric}={test_score_str})"
        # Predicted & training
        pred_train = pd.DataFrame(self._pipeline.predict(self._X_train), columns=self._Y_train.columns)
        pred_train["subset"] = train_title
        # Predicted & test
        pred_test = pd.DataFrame(self._pipeline.predict(self._X_test), columns=self._Y_test.columns)
        pred_test["subset"] = test_title
        # Actual & training
        act_train = self._Y_train.copy()
        act_train["subset"] = train_title
        # Actual & test
        act_test = self._Y_train.copy()
        act_test["subset"] = test_title
        # Combine data: index=reset, columns=parameter/subset/Predicted/Actual
        df = pd.concat([pred_train, pred_test], ignore_index=True)
        df = df.melt(id_vars=["subset"], var_name="parameter", value_name=PRED)
        act_df = pd.concat([act_train, act_test], ignore_index=True)
        act_df = act_df.melt(id_vars=["subset"], var_name="parameter", value_name=ACTUAL)
        df.loc[:, ACTUAL] = act_df.loc[:, ACTUAL]
        # Plotting
        with _PredActualPlot(filename=filename) as pa:
            pa.plot(df, x=ACTUAL, y=PRED)
            pa.title = TITLE
