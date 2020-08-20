#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dask import dataframe as dd
import pandas as pd
from covsirphy.cleaning.term import Term


class CleaningBase(Term):
    """
    Basic class for data cleaning.

    Args:
        filename (str): CSV filename of the dataset
    """

    def __init__(self, filename):
        if filename is None:
            self._raw = pd.DataFrame()
            self._cleaned_df = pd.DataFrame()
        else:
            self._raw = dd.read_csv(
                filename, dtype={"Province/State": "object"}
            ).compute()
            self._cleaned_df = self._cleaning()
        self._citation = str()

    @property
    def raw(self):
        """
        Return the raw data.

        Returns:
            (pandas.DataFrame): raw data
        """
        return self._raw

    def cleaned(self):
        """
        Return the cleaned dataset.

        Notes:
            Cleaning method is defined by self._cleaning() method.

        Returns:
            (pandas.DataFrame): cleaned data
        """
        return self._cleaned_df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Notes:
            Cleaning method is defined by self._cleaning() method.

        Returns:
            (pandas.DataFrame): cleaned data
        """
        return self._raw.copy()

    @property
    def citation(self):
        """
        str: citation/description of the dataset
        """
        return self._citation

    @citation.setter
    def citation(self, description):
        self._citation = str(description)

    def iso3_to_country(self, iso3_code):
        """
        Convert ISO3 code to country name if records are available.

        Args:
            iso3_code (str): ISO3 code or country name

        Returns:
            (str): country name

        Notes:
            If ISO3 codes are not registered, return the string as-si @iso3_code.
        """
        df = self._cleaned_df.copy()
        if self.ISO3 not in df.columns or iso3_code not in df[self.ISO3].unique():
            return iso3_code
        country_dict = df.set_index(self.ISO3)[self.COUNTRY].to_dict()
        return country_dict[iso3_code]

    def country_to_iso3(self, country):
        """
        Convert country name to ISO3 code if records are available.

        Args:
            country (str): country name

        Raises:
            KeyError: ISO3 code of the country is not registered

        Returns:
            (str): ISO3 code
        """
        if self.ISO3 not in self._cleaned_df.columns:
            return self.UNKNOWN
        df = self._cleaned_df.copy()
        iso3_dict = df.set_index(self.COUNTRY)[self.ISO3].to_dict()
        if country not in iso3_dict.keys():
            raise KeyError(f"@country {country} has not been registered.")
        return iso3_dict[country]

    @classmethod
    def area_name(cls, country, province=None):
        """
        Return area name of the country/province.

        Args:
            country (str): country name or ISO3 code
            province (str): province name

        Returns:
            (str): area name

        Notes:
            If province is None or '-', return country name.
            If not, return the area name, like 'Japan/Tokyo'
        """
        if province in [None, cls.UNKNOWN]:
            return country
        return f"{country}{cls.SEP}{province}"

    def _subset_by_country(self, country):
        """
        Return the subset of the country.

        Args:
            country (str): country name or ISO3 code

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    without ISO3 column and Country column
        """
        country = self.iso3_to_country(country)
        df = self._cleaned_df.copy()
        if self.COUNTRY not in df.columns:
            return df
        df = df.loc[df[self.COUNTRY] == country, :]
        df = df.drop(self.COUNTRY, axis=1)
        df = df.drop(self.ISO3, axis=1, errors="ignore")
        return df.reset_index(drop=True)

    def _subset_by_province(self, record_df, province):
        """
        Return subset of the province.

        Args:
            record_df (pandas.DataFrame): dataframe of the records
                Index:
                    reset index
                Columns:
                    without ISO3 column and Country column
            province (str or None): province name

        Raises:
            KeyError: records of the area (country/province) are not registered

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    without ISO3, Country and Province column
        """
        province = province or self.UNKNOWN
        df = record_df.copy()
        p_series = df[self.PROVINCE]
        # Province was selected
        if province != self.UNKNOWN:
            df = df.loc[p_series == province, :]
            if df.empty:
                raise KeyError(
                    f"@province {province} has not been registered.")
            if self.DATE in df.columns:
                df = df.groupby(self.DATE).last().reset_index()
            return df.drop(self.PROVINCE, axis=1)
        # Calculate total values at country level if not registered
        total_df = df.loc[p_series != self.UNKNOWN, :]
        if self.DATE in df.columns:
            total_df = total_df.groupby(self.DATE).sum().reset_index()
            if not total_df.empty:
                total_df.loc[:, self.PROVINCE] = self.UNKNOWN
        else:
            sum_dict = total_df.sum(axis=0, numeric_only=True).to_dict()
            sum_dict[self.PROVINCE] = self.UNKNOWN
            total_df = total_df.append(pd.Series(sum_dict), ignore_index=True)
        df = pd.concat([df, total_df], axis=0, ignore_index=True, sort=False)
        if self.DATE in df.columns:
            df = df.groupby([self.PROVINCE, self.DATE]).max().reset_index()
        else:
            df = df.groupby(self.PROVINCE).max().reset_index()
        # Return country-level records
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN, :]
        df = df.reset_index(drop=True)
        return df.drop(self.PROVINCE, axis=1)

    def _subset_by_area(self, country, province=None):
        """
        Return subset of the country/province.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name

        Raises:
            ValueError: @province is not None, but the dataset does not have Province column
            KeyError: records of the area (country/province) are not registered

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    without ISO3, Country, Province column
        """
        # Country level
        df = self._subset_by_country(country=country)
        if df.empty:
            raise KeyError(
                f"Please register population value for country={country}"
            )
        # Province level
        if self.PROVINCE not in df.columns:
            raise ValueError(
                "@province was specified, but the dataset does not have Province column.")
        df = self._subset_by_province(df, province=province)
        if df.empty:
            raise KeyError(
                f"Please register population value for (country={country}, province={province})"
            )
        return df

    def subset(self, country, province=None, start_date=None, end_date=None):
        """
        Return subset of the country/province and start/end date.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Raises:
            ValueError:
                - @country was None
                - @province is not None, but the dataset does not have Province column
            KeyError: selected records are not registered

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    without ISO3, Country, Province column
        """
        if country is None:
            ValueError("@country must be specified.")
        df = self._subset_by_area(country=country, province=province)
        # Subset with Start/end date
        if start_date is None and end_date is None:
            return df.reset_index(drop=True)
        if self.DATE not in df.columns:
            raise KeyError(
                "@start_date or @end_date was specified, but the dataset does not have Date column.")
        series = df[self.DATE].copy()
        start_obj = self.date_obj(date_str=start_date, default=series.min())
        end_obj = self.date_obj(date_str=end_date, default=series.max())
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        df = df.reset_index(drop=True)
        if df.empty:
            s1 = f"Records from {start_date} to {end_date} were not registered."
            raise KeyError(
                f"{s1} (country={country}, province={province})")
        return df

    def countries(self):
        """
        Return names of countries where records are registered.

        Raises:
            KeyError: Country names are not registered in this dataset

        Returns:
            (list[str]): list of country names
        """
        df = self._cleaned_df.copy()
        if self.COUNTRY not in df.columns:
            raise KeyError("Country names are not registered in this dataset.")
        return list(df[self.COUNTRY].unique())

    def total(self):
        """
        Calculate total values of the cleaned dataset.

        Returns:
            (pandas.DataFrame or pandas.Series or float):
                Index: 'Date' (pandas.TimeStamp) or not exist (pandas.Series)
                Columns: column names of the cleaned dataset (dtype=int or float)
                Values: total values
        """
        df = self._cleaned_df.copy()
        cols = list(set([self.COUNTRY, self.DATE]) & set(df.columns))
        if not cols:
            return df.sum(axis=0)
        # Calculate total values at country level if not registered
        c_level_df = df.groupby(cols).sum().reset_index()
        if self.PROVINCE in df.columns:
            c_level_df[self.PROVINCE] = self.UNKNOWN
            cols = [*cols, self.PROVINCE]
        df = pd.concat([df, c_level_df], axis=0, ignore_index=True, sort=False)
        df = df.drop_duplicates(subset=cols)
        if self.PROVINCE in df.columns:
            df = df.loc[df[self.PROVINCE] == self.UNKNOWN, :]
        # Calculate total value of each column
        if self.DATE in df.columns:
            return df.groupby(self.DATE).sum()
        return df.sum()
