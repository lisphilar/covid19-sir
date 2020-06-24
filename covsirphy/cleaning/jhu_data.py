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

    def cleaned(self, population=None):
        """
        Return the cleaned dataset.

        Note:
            Cleaning method is defined by self.cleaning() method.

        Args:
            population (int):
                - if this is not None, Susceptible will be calculated.

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
                    - if @population is not None:
                        - Susceptible (int): the number of susceptible cases
        """
        df = self._cleaned_df.copy()
        if population is None:
            return df
        population = self.validate_natural_int(population, name="population")
        df[self.S] = population - df[self.C]
        return df

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
        df[self.PROVINCE] = df[self.PROVINCE].fillna("-").replace(
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
            raise TypeError("Type of @country_data must be <covsirphy.CountryData>.")
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

    def subset(self, country, province=None):
        """
        Return the subset in the area.

        Args:
            country (str): country name
            province (str): province name

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

        Notes:
            Records with Recovered > 0 will be selected.
        """
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] == country, :]
        if province:
            df = df.loc[df[self.PROVINCE] == province, :]
        df = df.groupby(self.DATE).sum().reset_index()
        df = df.loc[df[self.R] > 0, :]
        if df.empty:
            if province is None:
                raise KeyError(
                    f"@country {country} is not included in the dataset."
                )
            raise KeyError(
                f"({country}, {province}) is not included in the dataset."
            )
        return df
