#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log10, floor
import warnings
import numpy as np
from optuna.exceptions import ExperimentalWarning
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from covsirphy.util.argument import find_args
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.term import Term
from covsirphy.regression.reg_pred_actual_plot import _PredActualPlot


class _RegressorBase(Term):
    """
    Basic class to predict parameter values of ODE models.

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
        kwargs: keyword arguments of sklearn.model_selection.train_test_split(test_size=0.2, random_state=0)

    Note:
        If @seed is included in kwargs, this will be converted to @random_state.
    """
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    warnings.simplefilter("ignore", category=ExperimentalWarning)
    # Description of regressor
    DESC = ""

    def __init__(self, X, y, delay_values, **kwargs):
        # Validate values
        self._ensure_dataframe(X, name="X", time_index=True)
        self._ensure_dataframe(y, name="y", time_index=True)
        self._delay_values = [self._ensure_natural_int(v) for v in self._ensure_list(delay_values)]
        # Set training/test dataset
        splitted_all = self._split(X, y, self._delay_values, **kwargs)
        self._X_train, self._X_test, self._y_train, self._y_test, self._X_target = splitted_all
        # Regression model
        self._pipeline = None
        self._param = {}
        # Perform fitting
        self._fit()

    @staticmethod
    def _split(X, y, delay_values, **kwargs):
        """
        Add delayed indicator values to X and split X and y to train/test data.

        Args:
            X (pandas.DataFrame): indicators with time index
            y (pandas.DataFrame): target values with time index
            delay_values (list[int]): list of delay period [days]
            kwargs: keyword arguments of sklearn.model_selection.train_test_split()

        Returns:
            tuple(pandas.DataFrame): datasets with time index
                - X_train
                - X_test
                - y_train
                - y_test
                - X_target

        Note:
            If @seed is included in kwargs, this will be converted to @random_state.

        Note:
            default values regarding sklearn.model_selection.train_test_split() are
            test_size=0.2, random_state=0, shuffle=False.
        """
        split_kwargs = {"test_size": 0.2, "random_state": 0, "shuffle": False}
        split_kwargs.update(kwargs)
        split_kwargs["random_state"] = split_kwargs.get("seed", split_kwargs["random_state"])
        split_kwargs = find_args(train_test_split, **split_kwargs)
        # Add delayed indicator values to X
        X_delayed = X.copy()
        for delay in delay_values:
            X_delayed = X_delayed.join(X.shift(delay, freq="D"), how="outer", rsuffix=f"_{delay}")
        X_delayed = X_delayed.ffill()
        # Training/test data
        df = X_delayed.join(y, how="inner").dropna().drop_duplicates()
        X_arranged = df.loc[:, X_delayed.columns]
        y_arranged = df.loc[:, y.columns]
        splitted = train_test_split(X_arranged, y_arranged, **split_kwargs)
        # X_target
        X_target = X_delayed.loc[X_delayed.index > y.index.max()]
        return (*splitted, X_target)

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
        pred_train = pd.DataFrame(self._pipeline.predict(self._X_train), columns=self._y_train.columns)
        return Evaluator(pred_train, self._y_train, how="all").score(metric=metric)

    def score_test(self, metric):
        """
        Calculate score with test dataset.

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()

        Returns:
            float: evaluation score
        """
        pred_test = pd.DataFrame(self._pipeline.predict(self._X_test), columns=self._y_test.columns)
        return Evaluator(pred_test, self._y_test, how="all").score(metric=metric)

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
            "delay": self._delay_values,
            "dataset": {
                "X_train": self._X_train,
                "y_train": self._y_train,
                "X_test": self._X_test,
                "y_test": self._y_test,
                "X_target": self._X_target,
            }
        }

    def predict(self):
        """
        Predict parameter values (via y) with self._pipeline and X_target.

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): future dates
                Columns
                    (float): parameter values (4 digits)
        """
        # Predict parameter values
        predicted = self._pipeline.predict(self._X_target)
        df = pd.DataFrame(predicted, index=self._X_target.index, columns=self._y_train.columns)
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
        pred_train = pd.DataFrame(self._pipeline.predict(self._X_train), columns=self._y_train.columns)
        pred_train["subset"] = train_title
        # Predicted & test
        pred_test = pd.DataFrame(self._pipeline.predict(self._X_test), columns=self._y_test.columns)
        pred_test["subset"] = test_title
        # Actual & training
        act_train = self._y_train.copy()
        act_train["subset"] = train_title
        # Actual & test
        act_test = self._y_train.copy()
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
