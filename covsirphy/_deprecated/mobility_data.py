#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.util.error import SubsetNotFoundError, deprecate
from covsirphy._deprecated.cbase import CleaningBase


class MobilityData(CleaningBase):
    """
    Data cleaning of mobility dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        data (pandas.DataFrame or None):
            Index
                reset index
            Columns
                - Date: Observation date
                - Country: country/region name
                - ISO3: ISO 3166-1 alpha-3, like JPN
                - Province: province/prefecture/state name
                - variables defined by @variables
        citation (str or None): citation or None (empty)
        variables (list[str] or None): variables to parse or None (use default variables listed as follows)
            - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
            - Mobility_parks: % to baseline in visits (parks etc.)
            - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
            - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
            - Mobility_residential: % to baseline in visits (places of residence)
            - Mobility_workplaces: % to baseline in visits (places of work)

    Note:
        Either @filename (high priority) or @data must be specified.

    Note:
        The default categories of places are listed in covid-19-open-data.
        https://github.com/GoogleCloudPlatform/covid-19-open-data/blob/main/docs/table-mobility.md
    """
    _MOBILITY_VARS = [
        "Mobility_grocery_and_pharmacy",
        "Mobility_parks",
        "Mobility_transit_stations",
        "Mobility_retail_and_recreation",
        "Mobility_residential",
        "Mobility_workplaces",
    ]

    @deprecate(old="MobilityData", new="DataEngineer", version="2.24.0-xi")
    def __init__(self, filename=None, data=None, citation=None, variables=None):
        self._variables = variables or self._MOBILITY_VARS[:]
        super().__init__(filename=filename, data=data, citation=citation, variables=self._variables)

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - ISO3 (str): ISO 3166-1 alpha-3, like JPN
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - variables defined by MobilityData(variables)
        """
        df = self._raw.copy()
        # Confirm the expected columns are in raw data
        self._ensure_dataframe(df, name="the raw data", columns=self._raw_cols)
        # Read date records
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Confirm int type
        for col in self._variables:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(100).astype(np.int64)
        # Update data types to reduce memory
        cat_cols = [self.ISO3, self.COUNTRY, self.PROVINCE]
        df[cat_cols] = df[cat_cols].astype("category")
        return df.loc[:, self._raw_cols]

    def subset(self, country, province=None):
        """
        Create a subset for a country.

        Args:
            country (str): country name or ISO 3166-1 alpha-3, like JPN
            province (str): province name

        Raises:
            covsirphy.SubsetNotFoundError: no records were found

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - variables defined by MobilityData(variables)
        """
        country_arg = country
        country = self.ensure_country_name(country)
        try:
            df = super().subset(country=country, province=province)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(country=country_arg, country_alias=country, province=province) from None
        df = df.groupby(self.DATE).last().reset_index()
        return df.loc[:, self._subset_cols]

    def total(self):
        """
        This method is not defined for MobilityData class.
        """
        raise NotImplementedError

    def map(self, country=None, variable="Mobility_grocery_and_pharmacy", date=None, **kwargs):
        """
        Create global colored map to show the values.

        Args:
            country (str or None): country name or None (global map)
            variable (str): variable name to show
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()

        Note:
            When @country is None, country level data will be shown on global map.
            When @country is a country name, province level data will be shown on country map.
        """
        # Date
        date_str = date or self.cleaned()[self.DATE].max().strftime(self.DATE_FORMAT)
        country_str = country or "Global"
        category_name = variable.replace("Mobility_", "").replace("_", " ")
        title = f"{country_str}: Mobility data ({category_name}) on {date_str}"
        # Global map
        if country is None:
            return self._colored_map_global(variable=variable, title=title, date=date, **kwargs)
        # Country-specific map
        return self._colored_map_country(
            country=country, variable=variable, title=title, date=date, **kwargs)
