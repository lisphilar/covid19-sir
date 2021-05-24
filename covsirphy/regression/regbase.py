#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from covsirphy.util.argument import find_args
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.term import Term


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
        delay (int): delay period [days]
        kwargs: keyword arguments of sklearn.model_selection.train_test_split(test_size=0.2, random_state=0)

    Note:
        If @seed is included in kwargs, this will be converted to @random_state.
    """
    # Description of regressor
    DESC = ""

    def __init__(self, X, y, delay, **kwargs):
        # Validate values
        self._ensure_dataframe(X, name="X", time_index=True)
        self._ensure_dataframe(y, name="y", time_index=True)
        self._delay = self._ensure_natural_int(delay, name="delay")
        # Set training/test dataset
        splitted_all = self._split(X, y, self._delay, **kwargs)
        self._X_train, self._X_test, self._y_train, self._y_test, self._X_target = splitted_all
        # Regression model
        self._regressor = None
        self._param = {}
        # Perform fitting
        self._fit()

    @staticmethod
    def _split(X, y, delay, **kwargs):
        """
        Apply delay period to the X dataset.

        Args:
            X (pandas.DataFrame): indicators with time index
            y (pandas.DataFrame): target values with time index
            delay (int): delay period [days]
            kwargs: keyword arguments of sklearn.model_selection.train_test_split(test_size=0.2, random_state=0)

        Returns:
            tuple(pandas.DataFrame): datasets with time index
                - X_train
                - X_test
                - y_train
                - y_test
                - X_target

        Note:
            If @seed is included in kwargs, this will be converted to @random_state.
        """
        split_kwargs = {"test_size": 0.2, "random_state": 0, }
        split_kwargs.update(kwargs)
        split_kwargs["random_state"] = split_kwargs.get("seed", split_kwargs["random_state"])
        split_kwargs = find_args(train_test_split, **split_kwargs)
        # Apply delay period to X
        X_delayed = X.copy()
        X_delayed.index += timedelta(days=delay)
        # Training/test data
        df = X_delayed.join(y, how="inner").dropna().drop_duplicates()
        X_arranged = df.loc[:, X.columns]
        y_arranged = df.loc[:, y.columns]
        splitted = train_test_split(X_arranged, y_arranged, **split_kwargs)
        # X_target
        X_target = X_delayed.loc[X_delayed.index > y.index.max()]
        return (*splitted, X_target)

    def _fit(self):
        """
        Fit regression model with training dataset, update self._regressor and self._param.
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
        pred_train = pd.DataFrame(self._regressor.predict(self._X_train), columns=self._y_train.columns)
        return Evaluator(pred_train, self._y_train, how="all").score(metric=metric)

    def score_test(self, metric):
        """
        Calculate score with test dataset.

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()

        Returns:
            float: evaluation score
        """
        pred_test = pd.DataFrame(self._regressor.predict(self._X_test), columns=self._y_test.columns)
        return Evaluator(pred_test, self._y_test, how="all").score(metric=metric)

    def to_dict(self, metric):
        """
        Calculate score with training/test dataset.

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()
        """
        return {
            **self._param,
            "score_metric": metric,
            "score_train": self.score_train(metric=metric),
            "score_test": self.score_test(metric=metric),
            "delay": self._delay,
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
        Predict parameter values (via y) with self._regressor and X_target.
        This method should be overwritten by child classes.

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): future dates
                Columns
                    (float): parameter values
        """
        raise NotImplementedError
