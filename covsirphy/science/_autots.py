#!/usr/bin/env python
# -*- coding: utf-8 -*-

from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor
from covsirphy.util.config import config
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _AutoTSHandler(Term):
    """Class for forecasting with AutoTS package.

    Args:
        Y (pandas.DataFrame):
            Index
                pandas.Timestamp: Observation date
            Columns
                observed and the target variables (int or float)
        days (int): days to predict
        seed (int or None): random seed
        **kwargs: keyword arguments of autots.AutoTS() except for forecast_length (always the same as @days)

    Note:
        AutoTS package is developed at https://github.com/winedarksea/AutoTS
    """

    def __init__(self, Y, days, seed, **kwargs):
        self._Y = Validator(Y, "Y").dataframe(time_index=True, empty_ok=False)
        self._days = Validator(days, name="days").int(value_range=(1, None))
        autots_kwargs = {
            "forecast_length": self._days,
            "frequency": "D",
            "no_negatives": True,
            "model_list": "superfast",
            "model_interrupt": True,
            "ensemble": "horizontal",
            "max_generations": 1,
            "num_validations": 2,
            "validation_method": "backwards",
            "random_seed": Validator(seed, name="seed").int(),
            "n_jobs": "auto",
            "verbose": config.logger_level,
        }
        autots_kwargs.update(kwargs)
        self._autots = AutoTS(**autots_kwargs)
        self._regressor_forecast = None

    def fit(self, X=None):
        """Fit the model with/without other information.

        Args:
            X (pandas.DataFrame or None): information for regression or None (no information)
                Index
                    pandas.Timestamp: Observation date
                Columns
                    observed and the target variables (int or float)

        Returns:
            _AutoTSHandler: self
        """
        if X is not None:
            regressor_train, self._regressor_forecast = fake_regressor(
                self._Y,
                dimensions=4,
                forecast_length=self._days,
                drop_most_recent=self._autots.drop_most_recent,
                aggfunc=self._autots.aggfunc,
                verbose=self._autots.verbose,
            )
        self._autots.fit(self._Y, future_regressor=None if X is None else regressor_train)
        return self

    def predict(self):
        """Return the predicted values with the observed values.

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): Observation date
                Columns
                    as the same as Y of covsirphy._AutoTSHandler()
        """
        return self._autots.predict(future_regressor=self._regressor_forecast).forecast.rename_axis(self.DATE)
