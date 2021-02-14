#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from pathlib import Path
from dask import dataframe as dd
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy.cleaning.cbase import CleaningBase


class PopulationData(CleaningBase):
    """
    Data cleaning of total population dataset.

    Args:
        filename (str): CSV filename of the dataset
        citation (str): citation
    """
    POPULATION_COLS = [
        CleaningBase.ISO3,
        CleaningBase.COUNTRY,
        CleaningBase.PROVINCE,
        CleaningBase.DATE,
        CleaningBase.N
    ]

    def __init__(self, filename=None, citation=None):
        self.created_time = datetime.now()
        if filename is None:
            self._raw = pd.DataFrame()
            self._cleaned_df = pd.DataFrame(columns=self.POPULATION_COLS)
        else:
            self._raw = dd.read_csv(
                filename, dtype={"Province/State": "object"}
            ).compute()
            self._cleaned_df = self._cleaning()
        self._citation = citation or ""
        # Directory that save the file
        if filename is None:
            self._dirpath = Path("input")
        else:
            self._dirpath = Path(filename).resolve().parent

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - ISO3 (str): ISO3 code or "-"
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Date (pd.TimeStamp): date of the records (if available) or today
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
                "ObservationDate": self.DATE
            },
            axis=1
        )
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.COUNTRY, self.PROVINCE, self.N
        ]
        self._ensure_dataframe(df, name="the raw data", columns=expected_cols)
        # ISO3
        df[self.ISO3] = df[self.ISO3] if self.ISO3 in df.columns else self.UNKNOWN
        # Date
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Country
        df[self.COUNTRY] = df[self.COUNTRY].replace(
            {
                # COD
                "Congo, the Democratic Republic of the": "Democratic Republic of the Congo",
                # COG
                "Congo": "Republic of the Congo",
                # South Korea
                "Korea, South": "South Korea",
            }
        )
        # Province
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.UNKNOWN)
        # Set 'Others' as the country name of cruise ships
        ships = ["Diamond Princess", "Costa Atlantica", "Grand Princess", "MS Zaandam"]
        for ship in ships:
            df.loc[df[self.COUNTRY] == ship, [self.COUNTRY, self.PROVINCE]] = [self.OTHERS, ship]
        # Values
        df = df.dropna(subset=[self.N]).reset_index(drop=True)
        df[self.N] = df[self.N].astype(np.int64)
        # Columns to use
        df = df.loc[
            :, [self.ISO3, self.COUNTRY, self.PROVINCE, self.DATE, self.N]]
        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        return df

    def _created_date(self):
        """
        Return 0:00 AM of the created date.

        Returns:
            datetime.datetime
        """
        return self.created_time.replace(hour=0, minute=0, second=0, microsecond=0)

    def total(self):
        """
        Return the total value of population in the dataset.

        Returns:
            int
        """
        values = self._cleaned_df[self.N]
        return int(sum(values))

    def to_dict(self, country_level=True):
        """
        Return dictionary of population values.

        Args:
        country_level (str): whether key is country name or not

        Returns:
            dict
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
        return df.set_index("key").to_dict()[self.N]

    def value(self, country, province=None, date=None):
        """
        Return the value of population in the place.

        Args:
            country (str): country name or ISO3 code
            province (str): province name
            date (str or None): observation date, like 01Jun2020

        Returns:
            int: population in the place

        Note:
            If @date is None, the created date of the instancewill be used
        """
        country_alias = self.ensure_country_name(country)
        try:
            df = self.subset(
                country=country, province=province, start_date=date, end_date=date)
        except KeyError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province, date=date)
        df = df.sort_values(self.DATE)
        return int(df.loc[df.index[-1], [self.N]].values[0])

    def update(self, value, country, province=None, date=None):
        """
        Update the value of a new place.

        Args:
            value (int): population in the place
            country (str): country name
            province (str): province name
            date (str or None): observation date, like 01Jun2020

        Returns:
            covsirphy.PopulationData: self

        Note:
            If @date is None, the created date of the instance will be used.
            If @province is None, "-" will be used.
        """
        population = self._ensure_natural_int(value, "value")
        province = province or self.UNKNOWN
        date_stamp = pd.to_datetime(date or self._created_date())
        df = self._cleaned_df.copy()
        c_series = df[self.COUNTRY]
        p_series = df[self.PROVINCE]
        d_series = df[self.DATE]
        sel = (c_series == country) & (
            p_series == province) & (d_series == date_stamp)
        if not df.loc[sel, :].empty:
            df.loc[sel, self.N] = value
            self._cleaned_df = df.copy()
            return self
        series = pd.Series(
            [self.UNKNOWN, country, province, date_stamp, population],
            index=[self.ISO3, self.COUNTRY, self.PROVINCE, self.DATE, self.N]
        )
        self._cleaned_df = df.append(series, ignore_index=True)
        return self

    def countries(self):
        """
        Return names of countries where records are registered.

        Raises:
            KeyError: Country names are not registered in this dataset

        Returns:
            list[str]: list of country names

        Note:
            Country 'Others' will be removed.
        """
        country_list = super().countries()
        removed_countries = ["Others"]
        country_list = list(set(country_list) - set(removed_countries))
        return country_list

    def map(self, country=None, variable="Population", date=None, **kwargs):
        """
        Create colored map with the number of tests.

        Args:
            country (str or None): country name or None (global map)
            variable (str): always 'Population'
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()

        Raises:
            NotImplementedError: @variable was specified

        Note:
            When @country is None, country level data will be shown on global map.
            When @country is a country name, province level data will be shown on country map.
        """
        if variable != self.N:
            raise NotImplementedError(f"@variable cannot be changed, always {self.N}.")
        # Date
        date_str = date or self.cleaned()[self.DATE].max().strftime(self.DATE_FORMAT)
        country_str = country or "Global"
        title = f"{country_str}: {variable.lower()} on {date_str}"
        # Global map
        if country is None:
            return self._colored_map_global(
                variable=variable, title=title, date=date, **kwargs)
        # Country-specific map
        return self._colored_map_country(
            country=country, variable=variable, title=title, date=date, **kwargs)


class Population(PopulationData):
    @deprecate(old="Population()", new="PopulationData()")
    def __init__(self, filename):
        super().__init__(filename)
