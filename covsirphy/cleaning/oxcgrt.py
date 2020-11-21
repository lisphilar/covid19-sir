#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
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
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - ISO3 (str): ISO 3166-1 alpha-3, like JPN
                    - other column names are defined by OxCGRTData.COL_DICT
        """
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(self.OXCGRT_COL_DICT, axis=1)
        df = df.rename(
            {
                "CountryName": self.COUNTRY, "Date": self.DATE, "CountryCode": self.ISO3,
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
                # Diamond princess
                "Diamond Princess": "Others",
            }
        )
        # Confirm the expected columns are in raw data
        self.ensure_dataframe(
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
        return df

    def subset(self, country, **kwargs):
        """
        Create a subset for a country.

        Args:
            country (str): country name or ISO 3166-1 alpha-3, like JPN
            kwargs: the other arguments will be ignored in the latest version.

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - other column names are defined by OxCGRTData.COL_DICT
        """
        country = self.ensure_country_name(country)
        df = super().subset(country=country)
        df = df.groupby(self.DATE).last().reset_index()
        return df.loc[:, self.OXCGRT_COLS_WITHOUT_COUNTRY]

    def total(self):
        """
        This method is not defined for OxCGRTData class.
        """
        raise NotImplementedError
