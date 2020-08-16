#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dask import dataframe as dd
import numpy as np
import pandas as pd
from covsirphy.cleaning.cbase import CleaningBase


class CountryData(CleaningBase):
    """
    Data cleaning of country level data.

    Args:
        filename (str): filename to read the data
        country (str): country name
    """

    def __init__(self, filename, country):
        self._raw = dd.read_csv(filename).compute()
        self._country = country
        self.province_col = None
        self.var_dict = {}
        self._cleaned_df = pd.DataFrame()
        self._citation = str()

    @property
    def country(self):
        """
        Return the country name.

        Returns:
            (str): country name
        """
        return self._country

    def raw_columns(self):
        """
        Return the column names of the raw data.

        Returns:
            (list[str]): the list of column names of the raw data
        """
        return self._raw.columns.tolist()

    def set_variables(self, date, confirmed, fatal, recovered, province=None):
        """
        Set the correspondence of the variables and columns of the raw data.

        Args:
            date (str): column name for Date
            confirmed (str): column name for Confirmed
            fatal (str): column name for Fatal
            recovered (str): column name for Confirmed
            province (str): (optional) column name for Province
        """
        self.province_col = province
        self.var_dict = {
            date: self.DATE,
            confirmed: self.C,
            fatal: self.F,
            recovered: self.R
        }

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrite super()._cleaning() method.

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        if not self.var_dict:
            raise ValueError(
                "Please execute CountryData.set_variables() in advance.")
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(self.var_dict, axis=1)
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.DATE, self.C, self.F, self.R
        ]
        self.ensure_dataframe(df, name="the raw data", columns=expected_cols)
        # Remove empty rows
        df = df.dropna(subset=[self.DATE])
        # Add province column
        if self.province_col:
            df = df.rename({self.province_col: self.PROVINCE}, axis=1)
        else:
            df[self.PROVINCE] = self.UNKNOWN
        # Values
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        v_cols = self.VALUE_COLUMNS[:]
        df[v_cols] = df[v_cols].fillna(0).astype(np.int64)
        # Groupby date and province
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        df = df.groupby([self.DATE, self.PROVINCE]).sum().reset_index()
        # Add country column
        df[self.COUNTRY] = self._country
        df = df.loc[:, self.COLUMNS]
        return df

    def cleaned(self):
        """
        Return the cleaned dataset.
        Cleaning method is defined by CountryData._cleaning() method.

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/sstate name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        self._cleaned_df = self._cleaning()
        return self._cleaned_df

    def total(self):
        """
        Return a dataframe to show chronological change of number and rates.

        Returns:
            (pandas.DataFrame): group-by Date, sum of the values

                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Fatal per Confirmed (int)
                    - Recovered per Confirmed (int)
                    - Fatal per (Fatal or Recovered) (int)
        """
        df = self.cleaned()
        # Calculate total values at country level if not registered
        c_level_df = df.groupby(self.DATE).sum().reset_index()
        c_level_df[self.PROVINCE] = self.UNKNOWN
        df = pd.concat([df, c_level_df], axis=0, ignore_index=True)
        df = df.drop_duplicates(subset=[self.DATE, self.PROVINCE])
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN, :]
        df = df.drop([self.COUNTRY, self.PROVINCE], axis=1)
        df = df.set_index(self.DATE)
        # Calculate rates
        total_series = df.sum(axis=1)
        r_cols = self.RATE_COLUMNS[:]
        df[r_cols[0]] = df[self.F] / total_series
        df[r_cols[1]] = df[self.R] / total_series
        df[r_cols[2]] = df[self.F] / (df[self.F] + df[self.R])
        return df.loc[:, [*self.VALUE_COLUMNS, *r_cols]]

    def countries(self):
        """
        Return names of countries where records are registered.

        Returns:
            (list[str]): list of country names
        """
        return [self._country]
