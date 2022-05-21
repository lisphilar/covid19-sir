#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _DataTransformer(Term):
    """Class for data transformation.

    Args:
        data (pandas.DataFrame): raw data
            Index
                reset index
            Column
                - columns defined by @layers
                - column defined by @date
                - the other columns
        layers (list[str]): location layers of the data
        date (str): column name of observation dates of the data
    """

    def __init__(self, data, layers, date):
        self._df = data.copy()
        self._layers = Validator(layers, "layers").sequence()
        self._date = str(date)

    def all(self):
        """Return all available data.

        Returns:
            pandas.DataFrame: transformed data
        """
        return self._df

    def susceptible(self, new, **kwargs):
        """Calculate the number of susceptible cases with "Susceptible = Population - Confirmed" formula.

        Args:
            new (str): new column name
            kwargs (dict[str, str]): dictionary of columns
                - population (str): total population
                - confirmed (str): the number of confirmed cases
        """
        c_dict = Validator(kwargs, "kwargs").dict(required_keys=["population", "confirmed"], errors="raise")
        df = Validator(self._df, "raw data").dataframe(columns=list(c_dict.values()))
        df[new] = df[c_dict["population"]] - df[c_dict["confirmed"]]
        self._df = df.copy()

    def infected(self, new, **kwargs):
        """Calculate the number of infected cases with "Infected = Confirmed - Fatal - Recovered" formula.

        Args:
            new (str): new column name
            kwargs (dict[str, str]): dictionary of columns
                - confirmed (str): the number of confirmed cases
                - fatal (str): the number of fatal cases
                - recovered (str): the number of recovered cases
        """
        c_dict = Validator(kwargs, "kwargs").dict(required_keys=["confirmed", "fatal", "recovered"], errors="raise")
        df = Validator(self._df, "raw data").dataframe(columns=list(c_dict.values()))
        df[new] = df[c_dict["confirmed"]] - df[c_dict["fatal"]] - df[c_dict["recovered"]]
        self._df = df.copy()

    def diff(self, column, suffix, freq):
        """Calculate first discrete difference of element with "F(x>0) = F(x) - F(x-1), F(0) = 0".

        Args:
            column (str): column name of the cumulative numbers
            suffix (str): suffix if the column (new column name will be '{column}{suffix}')
            freq (str): offset aliases of shifting dates

        Note:
            Regarding @freq, refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """
        df = Validator(self._df, "raw data").dataframe(columns=[column])
        new_column = f"{column}{suffix}"
        series = df.set_index(self._date).groupby(self._layers).shift(freq=freq)[column]
        series.name = new_column
        df = df.merge(series, how="left", left_on=[*self._layers, self._date], right_index=True)
        df[new_column] = (df[column] - df[new_column]).fillna(0)
        self._df = df.copy()
