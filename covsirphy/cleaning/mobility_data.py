#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.cleaning.cbase import CleaningBase


class MobilityData(CleaningBase):
    """
    Data cleaning of mobility dataset.

    Args:
        arguments defined for CleaningBase class except for @variables
        variables (list[str] or None): variables to parse or None (use default variables listed as follows)
            - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
            - Mobility_parks: % to baseline in visits (parks etc.)
            - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
            - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
            - Mobility_residential: % to baseline in visits (places of residence)
            - Mobility_workplaces: % to baseline in visits (places of work)

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

    def __init__(self, variables=None, **kwargs):
        self._variables = variables or self._MOBILITY_VARS[:]
        super().__init__(variables=self._variables, **kwargs)

    def _cleaning(self, raw):
        """
        Perform data cleaning of the values of the raw data (without location information).

        Args:
            pandas.DataFrame: raw data

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Location_ID (str): location identifiers
                    - Date (pandas.Timestamp): Observation date
                    - variables defined by MobilityData(variables)
        """
        df = raw.copy()
        # Confirm the expected columns are in raw data
        self._ensure_dataframe(df, name="the raw data", columns=self._subset_cols)
        # Read date records
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Confirm int type
        for col in self._variables:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(100).astype(np.int64)
        return df

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
