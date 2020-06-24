#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        self._raw = pd.read_csv(filename)
        self._country = country
        self.province_col = None
        self.var_dict = dict()
        self._cleaned_df = pd.DataFrame()

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

    def cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrite super().cleaning() method.

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
            s = "Please execute CountryData.set_variables() in advance."
            raise Exception(s)
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(self.var_dict, axis=1)
        # Add province column
        if self.province_col:
            df = df.rename({self.province_col: self.PROVINCE}, axis=1)
        else:
            df[self.PROVINCE] = "-"
        # Values
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df[self.VALUE_COLUMNS] = df[self.VALUE_COLUMNS].astype(np.int64)
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
        Cleaning method is defined by self.cleaning() method.

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
        self._cleaned_df = self.cleaning()
        return self._cleaned_df
