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

    def mul(self, columns, new, fill_value):
        """Calculate element-wise multiplication, X1 * X2 * X3 *...

        Args:
            columns (str): columns to multiply
            new (str): column name of floating division
            fill_value (float): value to fill in NAs
        """
        Validator(columns, "columns").sequence(candidates=list(self._df.columns))
        fill_value_validated = Validator(fill_value, "fill_value").float()
        df = self._df.copy()
        df[new] = df[columns[0]].copy()
        for col in columns[1:]:
            df[new] = df[new].mul(df[col], fill_value=fill_value_validated)
        self._df = df.copy()

    def div(self, numerator, denominator, new, fill_value):
        """Calculate element-wise floating division, numerator / denominator * 100.

        Args:
            numerator (str): numerator column
            denominator (str): denominator column
            new (str): column name of floating division
            fill_value (float): value to fill in NAs

        Note:
            Positive rate could be calculated with Confirmed / Tested * 100 (%), `.div(numerator="Confirmed", denominator="Tested", new="Positive_rate")`
        """
        v = Validator([numerator, denominator], "columns of numerator and denominator")
        v.sequence(candidates=list(self._df.columns))
        df = self._df.copy()
        df[new] = numerator.div(denominator, fill_value=Validator(fill_value, "fill_value").float())
        self._df = df.copy()
