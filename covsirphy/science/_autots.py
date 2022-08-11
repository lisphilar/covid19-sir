#!/usr/bin/env python
# -*- coding: utf-8 -*-

from autots import AutoTS
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
        **kwargs: keyword arguments of autots.AutoTS() except for forecast_length (always the same as @days)

    Note:
        AutoTS package is developed at https://github.com/winedarksea/AutoTS
    """

    def __init__(self, Y, days, **kwargs):
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
            "random_seed": 0,
            "n_jobs": "auto",
            "verbose": 1,
        }
        autots_kwargs.update(kwargs)
        self._autots = AutoTS(**autots_kwargs)
        self._autots.fit(self._Y)

    def predict(self):
        """Return the predicted values with the observed values.

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): Observation date
                Columns
                    as the same as Y of covsirphy._AutoTSHandler()
        """
        return self._autots.predict().forecast.rename_axis(self.DATE)
