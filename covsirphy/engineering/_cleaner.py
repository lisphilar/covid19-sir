#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import pandas as pd
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _DataCleaner(Term):
    """Class for data cleaning.

    Args:
        data (pandas.DataFrame): raw data
            Index
                reset index
            Column
                columns defined by @layers
                column defined by @date
                the other columns
        layers (list[str]): location layers of the data
        date (str): column name of observation dates of the data
    """

    def __init__(self, data, layers, date):
        self._df = data.copy()
        self._layers = Validator(layers, "layers").sequence()
        self._date = str(date)
        self._id_cols = [*self._layers, self._date]

    def all(self):
        """Return all available data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Column
                    columns defined by @layers of _DataCleaner()
                    (pandas.Timestamp): observation dates defined by @date of _DataCleaner()
                    the other columns
        """
        return self._df

    def convert_date(self, **kwargs):
        """Convert dtype of date column to pandas.Timestamp.

        Args:
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"
        """
        df = self._df.copy()
        df[self._date] = pd.to_datetime(df[self._date], **kwargs).dt.round("D")
        with contextlib.suppress(TypeError):
            df[self._date] = df[self._date].dt.tz_convert(None)
        self._df = df.copy()

    def resample(self, date_range=None):
        """Resample records with dates.
        """
        self.convert_date()
        df = self._df.copy()
        if date_range is not None:
            start_date, end_date = Validator(date_range, "date_range").sequence(length=2)
            start = Validator(start_date, name="the first value of @date_range").date(default=df[self._date].min())
            end = Validator(
                end_date, name="the second date of @date_range").date(default=df[self._date].max(), value_range=(start, None))
            df = df[df[self._date].between(start, end, inclusive="both")]
        grouped = df.set_index(self._date).groupby(self._layers, as_index=False)
        df = grouped.resample("D").ffill()
        self._df = df.reset_index().drop("level_0", errors="ignore", axis=1)

    def fillna(self):
        """Fill NA values with '-' (layers) and the previous values and 0.
        """
        df = self._df.copy()
        df[self._layers] = df.loc[:, self._layers].astype(str).fillna(self.NA)
        for col in set(df.columns) - set(self._id_cols):
            df[col] = df.groupby(self._layers)[col].ffill().fillna(0)
        self._df = df.copy()
