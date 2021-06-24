#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from covsirphy.util.argument import find_args
from covsirphy.util.term import Term


class _FeatureEngineer(Term):
    """
    Perform feature engineering and split X, Y to train/test/target datasets.

    Args:
        X (pandas.DataFrame):
            Index
                Date (pandas.Timestamp): observation date
            Columns
                (int/float): indicators
        Y (pandas.DataFrame):
            Index
                Date (pandas.Timestamp): observation date
            Columns
                (int/float) target values
    """

    def __init__(self, X, Y):
        # Validate values
        self._X_raw = self._ensure_dataframe(X, name="X", time_index=True)
        self._X = X.copy()
        self._Y = self._ensure_dataframe(Y, name="Y", time_index=True)

    def split(self, **kwargs):
        """
        Split the current X and Y to train/test/target datasets.

        Args:
            kwargs: keyword arguments of sklearn.model_selection.train_test_split()

        Returns:
            dict[str, pandas.DataFrame]: datasets with time index
                - X_train
                - X_test
                - Y_train
                - Y_test
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
        # Train/test
        X, Y = self._X.copy(), self._Y.copy()
        df = X.join(Y, how="inner").dropna().drop_duplicates()
        X_arranged = df.loc[:, X.columns]
        Y_arranged = df.loc[:, Y.columns]
        splitted = train_test_split(X_arranged, Y_arranged, **split_kwargs)
        names = ["X_train", "X_test", "Y_train", "Y_test"]
        data_dict = {name: data for (name, data) in zip(names, splitted)}
        # X_target
        data_dict["X_target"] = X.loc[X.index > Y.index.max()]
        return data_dict

    def add_elapsed(self):
        """
        Calculate elapsed days from the last change point of indicators and add them as new features.
        """
        X, raw = self._X.copy(), self._X_raw.copy()
        for (name, col) in raw.iteritems():
            X[f"{name}_elapsed"] = col.groupby((col != col.shift()).cumsum()).cumcount() + 1
        self._X = X.copy()

    def apply_delay(self, delay_values):
        """
        Add delayed indicator values to X.

        Args:
            delay_values (list[int]): list of delay period [days]
        """
        X, raw = self._X.copy(), self._X_raw.copy()
        for delay in delay_values:
            X = X.join(raw.shift(delay, freq="D"), how="outer", rsuffix=f"_{delay}")
        self._X = X.ffill()
