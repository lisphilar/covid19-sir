#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy._deprecated.cbase import CleaningBase


class PopulationData(CleaningBase):
    """
    Deprecated.
    Data cleaning of total population dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        data (pandas.DataFrame or None):
            Index
                reset index
            Columns
                - Date: Observation date
                - ISO3: ISO3 code
                - Country: country/region name
                - Province: province/prefecture/state name
                - Population: total population
        citation (str or None): citation or None (empty)

    Note:
        Either @filename (high priority) or @data must be specified.
    """

    @deprecate("PopulationData", new="JHUData", version="2.21.0-xi-fu1")
    def __init__(self, filename=None, data=None, citation=None):
        super().__init__(filename=filename, data=data, citation=citation, variables=[self.N])

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
                    - Date (pd.Timestamp): date of the records (if available) or today
                    - Population (int): total population
        """
        df = self._raw.copy()
        # Date
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Province
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.NA)
        df[self.N] = df[self.N].fillna(0).astype(np.int64)
        # Columns to use
        df = df.loc[:, self._raw_cols]
        # Remove duplicates
        df = df.dropna(subset=[self.N]).drop_duplicates().reset_index(drop=True)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        return df

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
            df = df.loc[df[self.PROVINCE] == self.NA, :]
            df["key"] = df[self.COUNTRY]
        else:
            df = df.loc[df[self.PROVINCE] != self.NA, :]
            df["key"] = df[self.COUNTRY].str.cat(df[self.PROVINCE], sep=self.SEP)
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
            df = self.subset(country=country, province=province, start_date=date, end_date=date)
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
        province = province or self.NA
        date_stamp = self._ensure_date(date, name="date", default=self._cleaned_df[self.DATE].max())
        df = self._cleaned_df.copy()
        c_series = df[self.COUNTRY]
        p_series = df[self.PROVINCE]
        d_series = df[self.DATE]
        sel = (c_series == country) & (p_series == province) & (d_series == date_stamp)
        if not df.loc[sel, :].empty:
            df.loc[sel, self.N] = value
            self._cleaned_df = df.copy()
            return self
        series = pd.Series(
            [date_stamp, self.NA, country, province, population], index=self._raw_cols)
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
            return self._colored_map_global(variable=variable, title=title, date=date, **kwargs)
        # Country-specific map
        return self._colored_map_country(
            country=country, variable=variable, title=title, date=date, **kwargs)


class Population(PopulationData):
    @deprecate(old="Population()", new="PopulationData()")
    def __init__(self, filename):
        super().__init__(filename)
