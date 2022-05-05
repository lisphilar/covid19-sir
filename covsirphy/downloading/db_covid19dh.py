#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy.downloading.db import _DataBase


class _COVID19dh(_DataBase):
    """
    Access "COVID-19 Data Hub" server.
    https://covid19datahub.io/
    """
    # File title without extensions and suffix
    TITLE = "covid19dh"
    # Dictionary of column names
    _OXCGRT_COLS_RAW = [
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
        "stringency_index",
    ]
    OXCGRT_VARS = [v.capitalize() for v in _OXCGRT_COLS_RAW]
    COL_DICT = {
        "date": Term.DATE,
        "iso_alpha_3": Term.ISO3,
        "administrative_area_level_1": Term.COUNTRY,
        "administrative_area_level_2": Term.PROVINCE,
        "administrative_area_level_3": Term.CITY,
        "tests": Term.TESTS,
        "confirmed": Term.C,
        "deaths": Term.F,
        "recovered": Term.R,
        "population": Term.N,
        **dict(zip(_OXCGRT_COLS_RAW, OXCGRT_VARS)),
    }
    # Stdout when downloading (shown at most one time)
    STDOUT = "Retrieving datasets from COVID-19 Data Hub https://covid19datahub.io/"
    # Citation
    CITATION = 'Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub",' \
        ' Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.'

    def _country(self):
        """Returns country-level data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (pandas.NA): NAs
                    - City (pandas.NA): NAs
                    - Confirmed (numpy.float64): the number of confirmed cases
                    - Fatal (numpy.float64): the number of fatal cases
                    - Recovered (numpy.float64): the number of recovered cases
                    - Population (numpy.int64): population values
                    - Tests (numpy.float64): the number of tests
        """
        url = "https://storage.covid19datahub.io/level/1.csv.zip"
        df = self._provide(
            url=url, suffix="_level1", columns=list(self.COL_DICT.keys()) + ["id"], date="date", date_format="%Y-%m-%d")
        # ships will be regarded as provinces of "Others" country
        ships = df.loc[df[self.ISO3].isna(), self.COUNTRY].unique()
        for ship in ships:
            df.loc[df[self.COUNTRY] == ship, [self.ISO3, self.PROVINCE]] = [self.OTHERS, ship]
        return df.groupby("id").ffill().reset_index(drop=True)

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
                    - Province (str): province/state/prefecture names
                    - City (pandas.NA): NAs
                    - Confirmed (numpy.float64): the number of confirmed cases
                    - Fatal (numpy.float64): the number of fatal cases
                    - Recovered (numpy.float64): the number of recovered cases
                    - Population (numpy.int64): population values
                    - Tests (numpy.float64): the number of tests
        """
        url = "https://storage.covid19datahub.io/level/2.csv.zip"
        df = self._provide(
            url=url, suffix="_level2", columns=list(self.COL_DICT.keys()) + ["id"], date="date", date_format="%Y-%m-%d")
        if country == self.OTHERS:
            c_df = self._country()
            others_df = c_df.loc[df[self.ISO3] == self.OTHERS]
            df = pd.concat([df, others_df], axis=0, ignore_index=True)
        return df.groupby("id").ffill().reset_index(drop=True)

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
                    - Province (str): province/state/prefecture names
                    - City (str): city names
                    - Confirmed (numpy.float64): the number of confirmed cases
                    - Fatal (numpy.float64): the number of fatal cases
                    - Recovered (numpy.float64): the number of recovered cases
                    - Population (numpy.int64): population values
                    - Tests (numpy.float64): the number of tests
        """
        url = "https://storage.covid19datahub.io/level/3.csv.zip"
        df = self._provide(
            url=url, suffix="_level3", columns=list(self.COL_DICT.keys()) + ["id"], date="date", date_format="%Y-%m-%d")
        return df.groupby("id").ffill().reset_index(drop=True)
