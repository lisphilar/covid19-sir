#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from covsirphy.util.error import SubsetNotFoundError
from covsirphy.cleaning.cbase import CleaningBase


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
        citation (str or None): citation or None (empty)

    Note:
        Either @filename (high priority) or @data must be specified.

    Note:
        Policy indices (Overall etc.) are from README.md and documentation/index_methodology.md in
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
    # Columns of self._raw and self._clean_df
    RAW_COLS = [CleaningBase.DATE, CleaningBase.ISO3, CleaningBase.COUNTRY, *OXCGRT_VARS]
    # Columns of self.cleaned()
    CLEANED_COLS = RAW_COLS[:]
    # Columns of self.subset()
    SUBSET_COLS = [CleaningBase.DATE, *OXCGRT_VARS]
    # Indicators except for Stringency index
    OXCGRT_VARS_INDICATORS = [v for v in OXCGRT_VARS if v != "Stringency_index"]
    # Deprecated
    OXCGRT_VARIABLES_RAW = [v.lower() for v in OXCGRT_VARS]
    OXCGRT_COLS = RAW_COLS[:]
    OXCGRT_COLS_WITHOUT_COUNTRY = SUBSET_COLS[:]

    def __init__(self, filename=None, data=None, citation=None):
        # Raw data
        self._raw = self._parse_raw(filename, data, self.RAW_COLS)
        # Data cleaning
        self._cleaned_df = pd.DataFrame(columns=self.RAW_COLS) if self._raw.empty else self._cleaning()
        # Citation
        self._citation = citation or ""
        # Directory that save the file
        if filename is None:
            self._dirpath = Path("input")
        else:
            Path(filename).parent.mkdir(exist_ok=True, parents=True)
            self._dirpath = Path(filename).resolve().parent

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
        """
        df = self._raw.copy()
        # Prepare data for Greenland
        grl_df = df.loc[df[self.COUNTRY] == "Denmark"].copy()
        grl_df.loc[:, [self.ISO3, self.COUNTRY]] = ["GRL", "Greenland"]
        df = pd.concat([df, grl_df], sort=True, ignore_index=True)
        # Confirm the expected columns are in raw data
        self._ensure_dataframe(df, name="the raw data", columns=self.CLEANED_COLS)
        # Read date records
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Confirm float type
        for col in self.OXCGRT_VARS:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
        # Update data types to reduce memory
        cat_cols = [self.ISO3, self.COUNTRY]
        df[cat_cols] = df[cat_cols].astype("category")
        return df.loc[:, self.CLEANED_COLS]

    def subset(self, country, **kwargs):
        """
        Create a subset for a country.

        Args:
            country (str): country name or ISO 3166-1 alpha-3, like JPN
            kwargs: the other arguments will be ignored in the latest version.

        Raises:
            covsirphy.SubsetNotFoundError: no records were found

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
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
        """
        country_arg = country
        country = self.ensure_country_name(country)
        try:
            df = super().subset(country=country)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(country=country_arg, country_alias=country) from None
        df = df.groupby(self.DATE).last().reset_index()
        return df.loc[:, self.SUBSET_COLS]

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

        Raises:
            NotImplementedError: @country was specified
        """
        if country is not None:
            raise NotImplementedError("@country cannot be specified, always None.")
        # Date
        date_str = date or self.cleaned()[self.DATE].max().strftime(self.DATE_FORMAT)
        country_str = country or "Global"
        title = f"{country_str}: {variable.lower().replace('_', ' ')} on {date_str}"
        # Global map
        return self._colored_map_global(variable=variable, title=title, date=date, logscale=False, **kwargs)
