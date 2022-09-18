#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy.downloading._db import _DataBase


class _WPP(_DataBase):
    """
    Access "World Population Prospects by United nations" server.
    https://population.un.org/wpp/
    """
    TOP_URL = "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/"
    # File title without extensions and suffix
    TITLE = "world-population-prospects"
    # Dictionary of column names
    COL_DICT = {
        "Time": "Year",
        "ISO3_code": Term.ISO3,
        "PopTotal": Term.N,
    }
    ALL_COLS = [Term.DATE, Term.ISO3, Term.PROVINCE, Term.CITY, Term.N]
    # Stdout when downloading (shown at most one time)
    STDOUT = "Retrieving datasets from World Population Prospects https://population.un.org/wpp/"
    # Citation
    CITATION = 'United Nations, Department of Economic and Social Affairs,' \
        ' Population Division (2022). World Population Prospects 2022, Online Edition.'

    def _country(self):
        """Returns country-level data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (object): NAs
                    - City (object): NAs
                    - Population (numpy.float64): population values
        """
        url = f"{self.TOP_URL}WPP2022_TotalPopulationBySex.zip"
        df = self._provide(url=url, suffix="_level1", columns=list(self.COL_DICT.keys()))
        df[self.DATE] = pd.to_datetime(df["Year"], format="%Y") + pd.offsets.DateOffset(months=6)
        df[self.PROVINCE] = self.NA
        df[self.CITY] = self.NA
        df[self.N] = df[self.N] * 1_000
        return df.dropna(how="any").loc[:, self.ALL_COLS]

    def _province(self, country):
        """Returns province-level data.

        Args:
            country (str): country name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (object): NAs
                    - City (object): NAs
                    - Population (numpy.float64): population values
        """
        return pd.DataFrame(columns=self.ALL_COLS)

    def _city(self, country, province):
        """Returns city-level data.

        Args:
            country (str): country name
            province (str): province/state/prefecture name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (object): NAs
                    - City (object): NAs
                    - Population (numpy.float64): population values
        """
        return pd.DataFrame(columns=self.ALL_COLS)
