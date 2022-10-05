#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pca import pca
from covsirphy.util.config import config
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.science._autots import _AutoTSHandler


class MLEngineer(Term):
    """
    Class for machine learning and preprocessing.

    Args:
        seed (int or None): random seed
    """

    def __init__(self, seed=0, **kwargs):
        self._seed = Validator(seed, name="seed").int()
        if "verbose" in kwargs:
            verbose = kwargs.get("verbose", 2)
            config.logger(level=verbose)
            config.warning(
                f"Argument verbose was deprecated, please use covsirphy.config.logger(level={verbose}) instead.")

    def pca(self, X, n_components=0.95):
        """Perform PCA (principal component analysis) after standardization (Z-score normalization) with pca package.

        Args:
            X (pandas.DataFrame or None):
                Index
                    pandas.Timestamp: Observation date
                Columns
                    (int or float): observed values of the training vectors
            n_components (float or int): _the number of principal components or percentage of variance to cover at least

        Returns:
            dict of {str: object}: as the same as pca.pca().fit_transform()
                {"loadings": pandas.DataFrame}: structured dataframe containing loadings for PCs
                {"PC": pandas.DataFrame}: reduced dimensionality space, the Principal Components (PCs)
                    Index
                        pandas.Timestamp
                    COlumns
                        PC1, PC2,...
                {"explained_var": array-like}: explained variance for each fo the PCs (same ordering as the PCs)
                {"variance_ratio": array-like};: variance ratio
                {"model": object}: fitted model to be used for further usage of the model
                {"scaler": object}: scaler model
                {"pcp": int}: pcp
                {"topfeat": pandas.DataFrame}: top features
                    Index
                        reset index
                    Columns
                        PC (str): PC1, PC2,...
                        feature (str): feature name of X
                        loading (float): loading values
                        type (str): "best" or "weak
                {"outliers": pandas.DataFrame}: outliers
                    Index
                        pandas.Timestamp
                    Columns
                        y_proba (float)
                        y_score (float)
                        y_bool (bool)
                        y_bool_spe (bool)
                        y_score_spe (float)
                {"outlier_params": object}: parameter values of the model of finding outliers

        Note:
            Regarding pca package, please refer to https://github.com/erdogant/pca
        """
        Validator(X, name="X", accept_none=False).dataframe(time_index=True, empty_ok=False)
        model = pca(n_components=n_components, normalize=True, random_state=self._seed, verbose=config.logger_level)
        return {**model.fit_transform(X), "model": model}

    def forecast(self, Y, days, X=None, **kwargs):
        """Forecast Y for given days with/without indicators (X).

        Args:
            Y (pandas.DataFrame):
                Index
                    pandas.Timestamp: Observation date
                Columns
                    observed and the target variables (int or float)
            X (pandas.DataFrame or None): indicators for regression or None (no indicators)
                Index
                    pandas.Timestamp: Observation date
                Columns
                    observed and the target variables (int or float)
            days (int): days to predict
            **kwargs: keyword arguments of autots.AutoTS() except for verbose, forecast_length (always the same as @days)

        Return:
           pandas.DataFrame:
                Index
                    pandas.Timestamp: Observation date, from the next date of Y.index to the ast predicted date
                Columns
                    observed and the target variables (int or float)

        Note:
            AutoTS package is developed at https://github.com/winedarksea/AutoTS
        """
        model = _AutoTSHandler(Y=Y, days=days, seed=self._seed, **kwargs)
        return model.fit(X=X).predict()
