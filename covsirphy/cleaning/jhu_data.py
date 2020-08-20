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
            Cleaning method is defined by self._cleaning() method.

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
            raise ValueError(
                "@population was removed in JHUData.cleaned(). Please use JHUData.subset()")
        df = self._cleaned_df.copy()
        df = df.loc[:, self.COLUMNS]
        return df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrite super()._cleaning() method.

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - ISO3 (str): ISO3 code
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = super()._cleaning()
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
        # ISO3 code
        if self.ISO3 not in df.columns:
            df[self.ISO3] = self.UNKNOWN
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.DATE, self.ISO3, self.COUNTRY, self.PROVINCE, self.C, self.F, self.R
        ]
        self.ensure_dataframe(df, name="the raw data", columns=expected_cols)
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
                "Congo, the Democratic Republic of the": "Congo",
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
        df = df.loc[:, [self.ISO3, *self.COLUMNS]].reset_index(drop=True)
        return df

    def replace(self, country_data):
        """
        Replace a part of cleaned dataset with a dataframe.

        Args:
            country_data (covsirphy.CountryData): dataset object of the country

        Returns:
            self
        """
        self.ensure_instance(country_data, CountryData, name="country_data")
        # Read new dataset
        country = country_data.country
        new = country_data.cleaned()
        new[self.ISO3] = self.country_to_iso3(country)
        # Remove the data in the country from JHU dataset
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] != country, :]
        # Combine JHU data and the new data
        df = pd.concat([df, new], axis=0, sort=False)
        self._cleaned_df = df.copy()
        return self

    def subset(self, country, province=None,
               start_date=None, end_date=None, population=None):
        """
        Return the subset of dataset.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
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
        # Subset with area and start/end date
        subset_df = super().subset(
            country=country, province=province, start_date=start_date, end_date=end_date)
        # Select records where Recovered > 0
        df = subset_df.loc[subset_df[self.R] > 0, :]
        df = df.reset_index(drop=True)
        if df.empty:
            series = subset_df[self.DATE]
            start_date = start_date or series.min().strftime(self.DATE_FORMAT)
            end_date = end_date or series.max().strftime(self.DATE_FORMAT)
            s1 = "Records with Recovered > 0 are not registered."
            s2 = f"(country={country}, province={province}, period={start_date}-{end_date})"
            raise ValueError(f"{s1} {s2}")
        # Calculate Susceptible if population value was applied
        if population is None:
            return df
        population = self.ensure_natural_int(population, name="population")
        df.loc[:, self.S] = population - df.loc[:, self.C]
        return df

    def to_sr(self, country, province=None,
              start_date=None, end_date=None, population=None):
        """

        Args:
            country (str): country name
            province (str): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int): population value

        Returns:
            (pandas.DataFrame)
                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases

        Notes:
            @population must be specified.
            Records with Recovered > 0 will be used.
        """
        population = self.ensure_natural_int(population, "population")
        subset_df = self.subset(
            country=country, province=province,
            start_date=start_date, end_date=end_date, population=population
        )
        return subset_df.set_index(self.DATE).loc[:, [self.R, self.S]]

    @classmethod
    def from_dataframe(cls, dataframe):
        """
        Create JHUData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): Cleaned dataset

        Returns:
            (covsirphy.JHUData): JHU-style dataset
        """
        instance = cls(filename=None)
        instance._cleaned_df = cls.ensure_dataframe(
            dataframe, name="dataframe", columns=cls.COLUMNS)
        return instance

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
        df = super().total()
        total_series = df.sum(axis=1)
        r_cols = self.RATE_COLUMNS[:]
        df[r_cols[0]] = df[self.F] / total_series
        df[r_cols[1]] = df[self.R] / total_series
        df[r_cols[2]] = df[self.F] / (df[self.F] + df[self.R])
        return df.loc[:, [*self.VALUE_COLUMNS, *r_cols]]

    def countries(self):
        """
        Return names of countries where records with Recovered > 0 are registered.

        Returns:
            (list[str]): list of country names
        """
        df = self._cleaned_df.copy()
        df = df.loc[df[self.R] > 0, :]
        return list(df[self.COUNTRY].unique())
