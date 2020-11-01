#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.cleaning.cbase import CleaningBase


class LinelistData(CleaningBase):
    """
    Data cleaning of a linelist of case reports.
    """
    LINELIST_COLS = [
        "Age",
        "Sex",
        CleaningBase.COUNTRY,
        CleaningBase.PROVINCE,
        "Onset_date",
        "Hospitalized_date",
        "Confirmation_date",
        "Outcome",
        "Outcome_date",
        "Symptom",
        "Chronic_disease",
    ]

    def __init__(self, filename):
        super().__init__(filename)

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns: defined by LinelistData.LINELIST_COLS.
        """
        df = self._raw.copy()
        # Age/sex/country/province
        df.rename(
            {
                "age": "Age", "sex": "Sex",
                "country": self.COUNTRY, "province": self.PROVINCE
            },
            axis=1,
            inplace=True
        )
        # TODO: add cleaning code
        # TODO: add test in test/test_data_loader
        # TODO: documentation
        print(df)
        import sys
        sys.exit()
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

    def subset(self, country, iso3=None):
        """
        Create a subset for a country.

        Notes:
            One of @country and @iso3 must be specified.

        Args:
            country (str): country name or ISO 3166-1 alpha-3, like JPN

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - other column names are defined by OxCGRTData.COL_DICT
        """
        country = self.iso3_to_country(country or iso3)
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] == country, :]
        if df.empty:
            raise KeyError(
                f"Records of {country} are not included in the dataset.")
        df = df.groupby(self.DATE).last().reset_index()
        df = df.loc[:, self.OXCGRT_COLS_WITHOUT_COUNTRY]
        return df

    def total(self):
        """
        This is not defined for this child class.
        """
        raise NotImplementedError
