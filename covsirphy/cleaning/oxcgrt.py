#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.error import SubsetNotFoundError
from covsirphy.cleaning.cbase import CleaningBase


class OxCGRTData(CleaningBase):
    """
    Data cleaning of OxCGRT dataset.

    Args:
        filename (str): CSV filename of the dataset
        citation (str): citation
    """
    OXCGRT_VARIABLES_RAW = [
        "school_closing",
        "workplace_closing",
        "cancel_events",
        "gatherings_restrictions",
        "transport_closing",
        "stay_home_restrictions",
        "internal_movement_restrictions",
        "international_movement_restrictions",
        "information_campaigns",
        "testing_policy",
        "contact_tracing",
        "stringency_index"
    ]
    OXCGRT_COL_DICT = {v: v.capitalize() for v in OXCGRT_VARIABLES_RAW}
    OXCGRT_VARS = list(OXCGRT_COL_DICT.values())
    OXCGRT_COLS = [
        CleaningBase.DATE, CleaningBase.COUNTRY, CleaningBase.ISO3,
        *list(OXCGRT_COL_DICT.values())
    ]
    OXCGRT_COLS_WITHOUT_COUNTRY = [
        CleaningBase.DATE, *list(OXCGRT_COL_DICT.values())
    ]
    OXCGRT_VARS_INDICATORS = [
        v for v in OXCGRT_VARS if v != "Stringency_index"]

    def __init__(self, filename, citation=None):
        super().__init__(filename, citation)

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrite super()._cleaning() method.
        Policy indices (Overall etc.) are from
        README.md and documentation/index_methodology.md in
        https://github.com/OxCGRT/covid-policy-tracker/

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Country (pandas.Category): country/region name
                    - ISO3 (str): ISO 3166-1 alpha-3, like JPN
                    - other column names are defined by OxCGRTData.COL_DICT
        """
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(self.OXCGRT_COL_DICT, axis=1)
        df = df.rename(
            {
                "CountryName": self.COUNTRY, "CountryCode": self.ISO3,
                "Country/Region": self.COUNTRY, "ObservationDate": self.DATE,
            },
            axis=1
        )
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
        # Set 'Others' as the country name of cruise ships
        ships = ["Diamond Princess", "Costa Atlantica", "Grand Princess", "MS Zaandam"]
        for ship in ships:
            df.loc[df[self.COUNTRY] == ship, self.COUNTRY] = self.OTHERS
        # Confirm the expected columns are in raw data
        self._ensure_dataframe(
            df, name="the raw data", columns=self.OXCGRT_COLS
        )
        # Read date records
        try:
            df[self.DATE] = pd.to_datetime(df[self.DATE], format="%Y%m%d")
        except ValueError:
            df[self.DATE] = pd.to_datetime(df[self.DATE], format="%Y-%m-%d")
        # Confirm float type
        float_cols = list(self.OXCGRT_COL_DICT.values())
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(method="ffill")
        # Select the columns to use
        df = df.loc[:, [self.DATE, self.COUNTRY, self.ISO3, *float_cols]]
        # Update data types to reduce memory
        cat_cols = [self.ISO3, self.COUNTRY]
        df[cat_cols] = df[cat_cols].astype("category")
        return df

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
                    - Date (pd.TimeStamp): Observation date
                    - other column names are defined by OxCGRTData.COL_DICT
        """
        country_arg = country
        country = self.ensure_country_name(country)
        try:
            df = super().subset(country=country)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country_arg, country_alias=country) from None
        df = df.groupby(self.DATE).last().reset_index()
        return df.loc[:, self.OXCGRT_COLS_WITHOUT_COUNTRY]

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
