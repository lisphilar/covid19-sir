#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate, UnExecutedError
from covsirphy._deprecated.cbase import CleaningBase


class CountryData(CleaningBase):
    """
    Deprecated.
    Data cleaning of country level data.

    Args:
        filename (str or None): filename to read the data
        country (str): country name
        province (str or None): province name

    Note:
        If province name will be set in CountryData.set_variables(), @province will be ignored.
    """

    @deprecate("CountryData()", new="DataLoader.read_dataframe()", version="sigma",
               ref="https://lisphilar.github.io/covid19-sir/markdown/LOADING.html")
    def __init__(self, filename, country, province=None):
        self._raw = pd.DataFrame() if filename is None else pd.read_csv(filename)
        self._country = country
        self._province = province
        self.province_col = None
        self.var_dict = {}
        self._cleaned_df = pd.DataFrame()
        self._citation = str()
        # Directory that save the file
        if filename is None:
            self._dirpath = Path("input")
        else:
            self._dirpath = Path(filename).resolve().parent

    @property
    def country(self):
        """
        str: country name
        """
        return self._country

    def raw_columns(self):
        """
        Return the column names of the raw data.

        Returns:
            list[str]: the list of column names of the raw data
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

    def _cleaning(self, date_format):
        """
        Perform data cleaning of the raw data.
        This method overwrite super()._cleaning() method.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            date_format (str or None): the strftime to parse time (e.g. "%d/%m/%Y") or None (automated)
        """
        if not self.var_dict:
            raise UnExecutedError("CountryData.set_variables()")
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(self.var_dict, axis=1)
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.DATE, self.C, self.F, self.R
        ]
        self._ensure_dataframe(df, name="the raw data", columns=expected_cols)
        # Remove empty rows
        df = df.dropna(subset=[self.DATE])
        # Add province column
        if self.province_col is not None:
            df = df.rename({self.province_col: self.PROVINCE}, axis=1)
        else:
            df[self.PROVINCE] = self._province or self.NA
        # Values
        v_cols = [self.C, self.F, self.R]
        for col in v_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[v_cols] = df[v_cols].fillna(0).astype(np.int64)
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        # Groupby date and province
        df[self.DATE] = pd.to_datetime(df[self.DATE], format=date_format).dt.round("D")
        try:
            df[self.DATE] = df[self.DATE].dt.tz_convert(None)
        except TypeError:
            pass
        df = df.groupby([self.DATE, self.PROVINCE]).sum().reset_index()
        # Add country column
        df[self.COUNTRY] = self._country
        df = df.loc[:, self.COLUMNS]
        # Update data types to reduce memory
        df[self.AREA_COLUMNS] = df[self.AREA_COLUMNS].astype("category")
        return df

    def cleaned(self, date_format=None):
        """
        Return the cleaned dataset.
        Cleaning method is defined by CountryData._cleaning() method.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/sstate name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            date_format (str or None): the strftime to parse time (e.g. "%d/%m/%Y") or None (automated)

        Note:
            Please specify @date_format if dates are parsed incorrectly.

        Note:
            Date format codes: refer to https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        """
        if self._cleaned_df.empty:
            self._cleaned_df = self._cleaning(date_format=date_format)
        return self._cleaned_df

    def total(self):
        """
        Return a dataframe to show chronological change of number and rates.

        Returns:
            pandas.DataFrame: group-by Date, sum of the values

                Index
                    Date (pd.Timestamp): Observation date
                Columns
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
        c_level_df[self.PROVINCE] = self.NA
        df = pd.concat([df, c_level_df], axis=0, ignore_index=True)
        df = df.drop_duplicates(subset=[self.DATE, self.PROVINCE])
        df = df.loc[df[self.PROVINCE] == self.NA, :]
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
            list[str]: list of country names
        """
        return [self._country]

    def register_total(self):
        """
        Register total value of all provinces as country level data.

        Returns:
            covsirphy.CountryData: self

        Note:
            If country level data was registered, this will be overwritten.
        """
        # Calculate total values at province level
        clean_df = self.cleaned()
        clean_df = clean_df.loc[clean_df[self.PROVINCE] != self.NA]
        total_df = clean_df.groupby(self.DATE).sum().reset_index()
        total_df[self.COUNTRY] = self._country
        total_df[self.PROVINCE] = self.NA
        # Add/overwrite country level data
        df = clean_df.loc[clean_df[self.PROVINCE] != self.NA]
        df = pd.concat([df, total_df], ignore_index=True, sort=True)
        df[self.STR_COLUMNS] = df[self.STR_COLUMNS].astype("category")
        self._cleaned_df = df.loc[:, self.COLUMNS]
        return self

    def map(self, country=None, variable="Confirmed", date=None, **kwargs):
        """
        Create colored map to show the values.

        Args:
            country (None): None
            variable (str): variable name to show
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()

        Raises:
            NotImplementedError: @country was specified
        """
        if country is not None:
            raise NotImplementedError("@country cannot be specified, always None.")
        # Date
        date_str = date or self.cleaned()[self.DATE].max().strftime(self.DATE_FORMAT)
        title = f"{self._country}: the number of {variable.lower()} cases on {date_str}"
        # Country-specific map
        return self._colored_map_country(
            country=self._country, variable=variable, title=title, date=date, **kwargs)
