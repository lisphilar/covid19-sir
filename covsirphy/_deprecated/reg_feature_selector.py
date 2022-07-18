#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _FeatureSelector(Term, BaseEstimator, TransformerMixin):
    """
    Select useful features.
    - Highly correlated features will be replaced with zeros.

    Args:
        corr_threshold (float): lower limit of correlation

    Note:
        We can use this convert in scikit-learn pipeline because this is a child class of
        sklearn.base.BaseEstimator and sklearn.base.TransformerMixin.
    """

    def __init__(self, corr_threshold=0.9):
        self.corr_threshold = Validator(corr_threshold, "corr_threshold").float(value_range=(0, 1))

    @property
    def corr_threshold_(self):
        """
        bool: whether convert the values or not
        """
        return self.corr_threshold

    def fit(self, X, y):
        """
        Return self because fitting is unnecessary.

        Args:
            X (pandas.DataFrame): input X samples
            y (pandas.DataFrame): input y samples

        Returns:
            _FeatureSelector: self
        """
        return self

    def transform(self, X):
        """
        Transform the data.

        Args:
            X (pandas.DataFrame or numpy.ndarray): input samples

        Returns:
            pandas.DataFrame: transformed samples
        """
        df = pd.DataFrame(X)
        # Remove features with high correlation
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        dropped_features = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        df[dropped_features] = 0
        return df
