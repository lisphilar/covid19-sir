#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.util.error import deprecate


class PopulationData(CleaningBase):
    """
    Data cleaning of total population dataset.
    """
    POPULATION_COLS = [
        CleaningBase.ISO3,
        CleaningBase.COUNTRY,
        CleaningBase.PROVINCE,
        CleaningBase.N
    ]

    def __init__(self, filename):
        super().__init__(filename)

    def cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - ISO3 (str): ISO3 code or "-"
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Population (int): total population
        """
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(
            {
                "Country.Region": self.COUNTRY,
                "Country/Region": self.COUNTRY,
                "Province.State": self.PROVINCE,
                "Province/State": self.PROVINCE,
                "Population": self.N
            },
            axis=1
        )
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.COUNTRY, self.PROVINCE, self.N
        ]
        self.validate_dataframe(df, name="the raw data", columns=expected_cols)
        # ISO3
        if self.ISO3 not in df.columns:
            df[self.ISO3] = self.UNKNOWN
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
        df[self.N] = df[self.N].astype(np.int64)
        # Columns to use
        df = df.loc[:, [self.ISO3, self.COUNTRY, self.PROVINCE, self.N]]
        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    def total(self):
        """
        Return the total value of population in the datset.

        Returns:
            (int)
        """
        values = self._cleaned_df[self.N]
        return sum(values)

    def to_dict(self, country_level=True):
        """
        Return dictionary of population values.

        Args:
        country_level (str): whether key is country name or not

        Returns:
            (dict)
                - if @country_level is True, {"country", population}
                - if False, {"country/province", population}
        """
        df = self._cleaned_df.copy()
        if country_level:
            df = df.loc[df[self.PROVINCE] == self.UNKNOWN, :]
            df["key"] = df[self.COUNTRY]
        else:
            df = df.loc[df[self.PROVINCE] != self.UNKNOWN, :]
            df["key"] = df[self.COUNTRY].str.cat(
                df[self.PROVINCE], sep=self.SEP
            )
        pop_dict = df.set_index("key").to_dict()[self.N]
        return pop_dict

    def value(self, country, province=None):
        """
        Return the value of population in the place.

        Args:
            country (str): country name or ISO3 code
            province (str): province name

        Returns:
            (int): population in the place
        """
        cleaned_df = self._cleaned_df.copy()
        iso_df = cleaned_df.loc[cleaned_df[self.ISO3] == country, :]
        c_df = cleaned_df.loc[cleaned_df[self.COUNTRY] == country, :]
        df = pd.concat([iso_df, c_df], axis=0)
        if df.empty:
            if (cleaned_df[self.ISO3].unique()) == [self.UNKNOWN]:
                raise KeyError(
                    f"{country} is not registered. Please use registered country name.")
            raise KeyError(
                f"{country} is not registered. Please use ISO3 code as @country, like JPN.")
        if province is None:
            values = df.loc[df[self.PROVINCE] == self.UNKNOWN, self.N].values
            return int(values[0])
        if province not in df[self.PROVINCE].unique():
            raise KeyError(
                f"{province} is not registered as a province of {country}.")
        values = df.loc[df[self.PROVINCE] == province, self.N].values
        return int(values[0])

    def update(self, value, country, province="-"):
        """
        Update the value of a new place.

        Args:
        value (int): population in the place
        country (str): country name
        province (str): province name
        """
        value = self.validate_natural_int(value, "value")
        df = self._cleaned_df.copy()
        c_series = df[self.COUNTRY]
        p_series = df[self.PROVINCE]
        if country in c_series.unique() and province in p_series.unique():
            sel = (c_series == country) & (p_series == province)
            df.loc[sel, self.N] = value
            self._cleaned_df = df.copy()
            return self
        series = pd.Series(
            [self.UNKNOWN, country, province, value],
            index=[self.ISO3, self.COUNTRY, self.PROVINCE, self.N]
        )
        self._cleaned_df = df.append(series, ignore_index=True)
        return self


class Population(PopulationData):
    @deprecate(old="Population()", new="PopulationData()")
    def __init__(self, filename):
        super().__init__(filename)
