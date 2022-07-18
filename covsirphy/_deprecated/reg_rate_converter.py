#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _RateConverter(Term, BaseEstimator, TransformerMixin):
    """
    Convert Indicators(n) to Indicators(n)/Indicators(n-1).

    Args:
        to_convert (bool): whether convert the values or not

    Note:
        We can use this convert in scikit-learn pipeline because this is a child class of
        sklearn.base.BaseEstimator and sklearn.base.TransformerMixin.
    """

    def __init__(self, to_convert=True):
        self.to_convert = bool(to_convert)

    @property
    def to_convert_(self):
        """
        bool: whether convert the values or not
        """
        return self.to_convert

    def fit(self, X, y):
        """
        Return self because fitting is unnecessary.

        Args:
            X (pandas.DataFrame): input X samples
            y (pandas.DataFrame): input y samples

        Returns:
            _RateConverter: self
        """
        return self

    def transform(self, X):
        """
        Transform the data.

        Args:
            X (pandas.DataFrame): input samples

        Returns:
            pandas.DataFrame: transformed samples
        """
        Validator(X, "X").dataframe()
        if not self.to_convert:
            return X
        return X.div(X.shift(1)).replace(np.inf, 0).fillna(0)
