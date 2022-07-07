#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from autots import AutoTS
from sklearn.exceptions import ConvergenceWarning
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _AutoTSPredictor(Term):
    """
    Predict ODE parameter values with AutoTS library.

    Args:
        model_list (list[str]): list of machine learning models
        days (int): days to predict
        kwargs: keyword arguments of autots.AutoTS.

    Note:
        Candidates of @model_list can be checked with `from autots.models.model_list import model_lists; print(model_lists)`.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def __init__(self, days, model_list, **kwargs):
        autots_kwargs = {
            "forecast_length": days,
            "model_list": model_list,
            "frequency": "D",
            "ensemble": "horizontal",
            "max_generations": 1,
            "num_validations": 2,
            "validation_method": "backwards",
            "no_negatives": True,
            "constraint": 2.0,
            "random_seed": 0,
            "verbose": 1,
            "n_jobs": -1,
        }
        self._autots_kwargs = Validator(kwargs, "keyword arguments").kwargs(functions=AutoTS, default=autots_kwargs)

    def predict(self, past_df):
        """
        Run AutoTS prediction.

        Args:
            past_df (pandas.DataFrame): dataset for training/validation
                Index
                    pandas.Timestamp: Observation date
                Columns
                    observed variables (int or float)

        Returns:
            tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame): the most likely, upper, lower values
                Index
                    reset index
                Columns
                    predicted values (float)
        """
        model = AutoTS(**self._autots_kwargs)
        model = model.fit(past_df)
        prediction = model.predict()
        return (prediction.forecast, prediction.upper_forecast, prediction.lower_forecast)
