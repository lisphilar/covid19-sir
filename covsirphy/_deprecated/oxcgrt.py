#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.error import SubsetNotFoundError, deprecate
from covsirphy._deprecated.cbase import CleaningBase


class OxCGRTData(CleaningBase):
    """
    Data cleaning of OxCGRT dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        data (pandas.DataFrame or None):
            Index
                reset index
            Columns
                - Date: Observation date
                - ISO3: ISO 3166-1 alpha-3, like JPN
                - Country: country/region name
                - variables defined by @variables
        citation (str or None): citation or None (empty)
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
        Either @filename (high priority) or @data must be specified.

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

    @deprecate(old="OxCGRTData", new="DataEngineer", version="2.24.0-xi")
    def __init__(self, filename=None, data=None, citation=None, variables=None):
        self._variables = variables or self.OXCGRT_VARS[:]
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
                    - variables defined by OxCGRTData(variables)
        """
        df = self._raw.copy()
        # Prepare data for Greenland
        grl_df = df.loc[df[self.COUNTRY] == "Denmark"].copy()
        grl_df.loc[:, [self.ISO3, self.COUNTRY]] = ["GRL", "Greenland"]
        df = pd.concat([df, grl_df], sort=True, ignore_index=True)
        # Confirm the expected columns are in raw data
        self._ensure_dataframe(df, name="the raw data", columns=self._raw_cols)
        # Read date records
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Confirm float type
        for col in self._variables:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
        # Update data types to reduce memory
        cat_cols = [self.ISO3, self.COUNTRY, self.PROVINCE]
        df[cat_cols] = df[cat_cols].astype("category")
        return df.loc[:, self._raw_cols]

    def subset(self, country, province=None, **kwargs):
        """
        Create a subset for a country.

        Args:
            country (str): country name or ISO 3166-1 alpha-3, like JPN
            province (str): province name
            kwargs: the other arguments will be ignored at the latest version

        Raises:
            covsirphy.SubsetNotFoundError: no records were found

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - variables defined by OxCGRTData(variables)
        """
        country_arg = country
        country = self.ensure_country_name(country)
        try:
            df = super().subset(country=country, province=province)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country_arg, country_alias=country, province=province) from None
        df = df.groupby(self.DATE).last().reset_index()
        return df.loc[:, self._subset_cols]

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
