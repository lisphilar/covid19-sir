#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.country_data import CountryData


class JHUData(CleaningBase):
    """
    Data cleaning of JHU-style dataset.
    """

    def __init__(self, filename):
        super().__init__(filename)

    def cleaned(self, **kwargs):
        """
        Return the cleaned dataset.

        Note:
            Cleaning method is defined by self.cleaning() method.

        Args:
            kwargs: keword arguments will be ignored.

        Returns:
            (pandas.DataFrame)
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
        if "population" in kwargs.keys():
            raise KeyError(
                "@population was removed in JHUData.cleaned(). Please use JHUData.subset()")
        return self._cleaned_df

    def cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrite super().cleaning() method.

        Returns:
            (pandas.DataFrame)
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
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(
            {
                "ObservationDate": self.DATE,
                "Country/Region": self.COUNTRY,
                "Province/State": self.PROVINCE,
                "Deaths": self.F
            },
            axis=1
        )
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.DATE, self.COUNTRY, self.PROVINCE, self.C, self.F, self.R
        ]
        self.validate_dataframe(df, name="the raw data", columns=expected_cols)
        # Datetime columns
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Country
        df[self.COUNTRY] = df[self.COUNTRY].replace(
            {
                "Mainland China": "China",
                "Hong Kong SAR": "Hong Kong",
                "Taipei and environs": "Taiwan",
                "Iran (Islamic Republic of)": "Iran",
                "Republic of Korea": "South Korea",
                "Republic of Ireland": "Ireland",
                "Macao SAR": "Macau",
                "Russian Federation": "Russia",
                "Republic of Moldova": "Moldova",
                "Taiwan*": "Taiwan",
                "Cruise Ship": "Others",
                "United Kingdom": "UK",
                "Viet Nam": "Vietnam",
                "Czechia": "Czech Republic",
                "St. Martin": "Saint Martin",
                "Cote d'Ivoire": "Ivory Coast",
                "('St. Martin',)": "Saint Martin",
                "Congo (Kinshasa)": "Congo",
                "Congo (Brazzaville)": "Congo",
                "The, Bahamas": "Bahamas",
            }
        )
        # Province
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.UNKNOWN).replace(
            {
                "Cruise Ship": "Diamond Princess",
                "Diamond Princess cruise ship": "Diamond Princess"
            }
        )
        df.loc[df[self.COUNTRY] == "Diamond Princess", [
            self.COUNTRY, self.PROVINCE]] = ["Others", "Diamond Princess"]
        # Values
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df[self.VALUE_COLUMNS] = df[self.VALUE_COLUMNS].astype(np.int64)
        df = df.loc[:, self.COLUMNS].reset_index(drop=True)
        return df

    def replace(self, country_data):
        """
        Replace a part of cleaned dataset with a dataframe.

        Args:
            country_data (covsirphy.CountryData): dataset object of the country

        Returns:
            self
        """
        if not isinstance(country_data, CountryData):
            raise TypeError(
                "Type of @country_data must be <covsirphy.CountryData>.")
        # Read new dataset
        country = country_data.country
        new = country_data.cleaned()
        # Remove the data in the country from JHU dataset
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] != country, :]
        # Combine JHU data and the new data
        df = pd.concat([df, new], axis=0)
        self._cleaned_df = df.copy()
        return self

    @classmethod
    def area_name(cls, country, province=None):
        """
        Return area name of the country/province.

        Args:
            country (str): country name
            province (str): province name

        Returns:
            (str): area name
        """
        if province is None:
            return country
        return f"{country}{cls.SEP}{province}"

    def _subset_area(self, country, province=None, population=None):
        """
        Return the subset in the area.

        Args:
            country (str): country name
            province (str): province name
            population (int or None): population value

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Notes:
            If @population is not None, the number of susceptible cases will be calculated.
            Records with Recovered > 0 will be selected.
        """
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] == country, :]
        # Calculate Susceptible if population value was applied
        if population is not None:
            population = self.validate_natural_int(
                population, name="population")
            df[self.S] = population - df[self.C]
        # Check the country was registered
        if df.empty:
            raise KeyError(
                f"Records of {country} were not registered."
            )
        # Province was selected
        if province is not None:
            if province in df[self.PROVINCE].unique():
                df = df.loc[df[self.PROVINCE] == province, :]
                df = df.groupby(self.DATE).last().reset_index()
                df = df.drop([self.COUNTRY, self.PROVINCE], axis=1)
                return df.loc[df[self.R] > 0, :]
            raise KeyError(
                f"Records of {province} in {country} were not registered.")
        # Province was not selected and COVID-19 Data Hub dataset
        c_level_set = set(
            df.loc[df[self.PROVINCE] == self.UNKNOWN, self.DATE].unique()
        )
        all_date_set = set(df[self.DATE].unique())
        if c_level_set == all_date_set:
            df = df.loc[df[self.PROVINCE] == self.UNKNOWN, :]
            df = df.groupby(self.DATE).last().reset_index()
            df = df.drop([self.COUNTRY, self.PROVINCE], axis=1)
            return df.loc[df[self.R] > 0, :]
        # Province was not selected and Kaggle dataset
        df = df.groupby(self.DATE).sum().reset_index()
        return df.loc[df[self.R] > 0, :]

    def subset(self, country, province=None,
               start_date=None, end_date=None, population=None):
        """
        Return the subset of dataset.

        Args:
            country (str): country name
            province (str): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int or None): population value

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Notes:
            If @population is not None, the number of susceptible cases will be calculated.
            Records with Recovered > 0 will be selected.
        """
        # Subset with area
        df = self._subset_area(
            country, province=province, population=population
        )
        # Subset with Start/end date
        series = df[self.DATE].copy()
        start_obj = self.to_date_obj(date_str=start_date, default=series.min())
        end_obj = self.to_date_obj(date_str=end_date, default=series.max())
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        return df
