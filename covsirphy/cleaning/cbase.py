#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dask import dataframe as dd
from covsirphy.cleaning.word import Word


class CleaningBase(Word):
    """
    Basic class for data cleaning.
    """

    def __init__(self, filename):
        """
        @filename <str>: CSV filename of the dataset
        """
        self._raw = dd.read_csv(filename).compute()
        self._cleaned_df = self.cleaning()

    @property
    def raw(self):
        """
        Return the raw data.
        @return <pd.DataFrame>
        """
        return self._raw

    def cleaned(self):
        """
        Return the cleaned dataset.
        Cleaning method is defined by self.cleaning() method.
        @return <pd.DataFrame>
        """
        return self._cleaned_df

    def cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method will be defined in child classes.
        @return <pd.DataFrame>
        """
        df = self._raw.copy()
        return df

    def total(self):
        """
        Return a dataframe to show chronological change of number and rates.
        @return <pd.DataFrame>:
            - index (Date) <pd.TimeStamp>: Observation date
            - with group-by Date, sum of the values
                - Confirmed <int>: the number of confirmed cases
                - Infected <int>: the number of currently infected cases
                - Fatal <int>: the number of fatal cases
                - Recovered <int>: the number of recovered cases
            - Fatal per Confirmed <int>
            - Recovered per Confirmed <int>
            - Fatal per (Fatal or Recovered) <int>
        """
        df = self._cleaned_df.groupby("Date").sum()
        cols = ["Infected", "Fatal", "Recovered"]
        r_cols = self.RATE_COLUMNS[:]
        df[r_cols[0]] = df["Fatal"] / df[cols].sum(axis=1)
        df[r_cols[1]] = df["Recovered"] / df[cols].sum(axis=1)
        df[r_cols[2]] = df["Fatal"] / (df["Fatal"] + df["Recovered"])
        return df
