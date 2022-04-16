#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.cleaning.cbase import CleaningBase


class OxCGRTData(CleaningBase):
    """
    Data cleaning of OxCGRT dataset.

    Args:
        arguments defined for CleaningBase class except for @variables
        variables (list[str] or None): variables to parse or None (use default variables listed as follows)
            - School_closing
            - Workplace_closing
            - Cancel_events
            - Gatherings_restrictions
            - Transport_closing
            - Stay_home_restrictions
            - Internal_movement_restrictions
            - International_movement_restrictions
            - Information_campaigns
            - Testing_policy
            - Contact_tracing
            - Stringency_index

    Note:
        The default policy indices (Overall etc.) are from README.md and documentation/index_methodology.md in
        https://github.com/OxCGRT/covid-policy-tracker/
    """
    OXCGRT_VARS = [
        "School_closing",
        "Workplace_closing",
        "Cancel_events",
        "Gatherings_restrictions",
        "Transport_closing",
        "Stay_home_restrictions",
        "Internal_movement_restrictions",
        "International_movement_restrictions",
        "Information_campaigns",
        "Testing_policy",
        "Contact_tracing",
        "Stringency_index"
    ]
    # Indicators except for Stringency index
    OXCGRT_VARS_INDICATORS = [v for v in OXCGRT_VARS if v != "Stringency_index"]

    def __init__(self, variables=None, **kwargs):
        self._variables = variables or self.OXCGRT_VARS[:]
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
                    - variables defined by OxCGRTData(variables)
        """
        df = raw.copy()
        # Prepare data for Greenland
        denmark_series = self._loc_df.loc[self._loc_df[self.COUNTRY] == "Denmark", self._LOC]
        if not denmark_series.empty:
            denmark_id = denmark_series.unique()[0]
            greenland_id = "greenland"
            grl_df = df.loc[df[self._LOC] == denmark_id].copy()
            grl_df.loc[:, self._LOC] = greenland_id
            df = pd.concat([df, grl_df], sort=True, ignore_index=True)
            self._loc_df = self._loc_df.append(
                {
                    col: "GRL" if col == self.ISO3 else "Greenland" if col == self.COUNTRY else self.NA for col in self._layers
                }, ignore_index=True)
        # Confirm the expected columns are in raw data
        self._ensure_dataframe(df, name="the raw data", columns=self._subset_cols)
        # Read date records
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Confirm float type
        for col in self._variables:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
        return df

    def total(self):
        """
        This method is not defined for OxCGRTData class.
        """
        raise NotImplementedError

    def map(self, country=None, variable="Stringency_index", date=None, **kwargs):
        """
        Create global colored map to show the values.

        Args:
            country (None): always None
            variable (str): variable name to show
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()
        """
        # Date
        date_str = date or self.cleaned()[self.DATE].max().strftime(self.DATE_FORMAT)
        country_str = country or "Global"
        title = f"{country_str}: {variable.lower().replace('_', ' ')} on {date_str}"
        # Global map
        if country is None:
            return self._colored_map_global(
                variable=variable, title=title, date=date, logscale=False, **kwargs)
        # Country-specific map
        return self._colored_map_country(
            country=country, variable=variable, title=title, date=date, logscale=False, **kwargs)
