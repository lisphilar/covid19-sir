#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.term import Term
from covsirphy.downloading._db import _DataBase


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
            url=url, suffix="_level1", columns=list(self.COL_DICT.keys()), date="date", date_format="%Y-%m-%d")
        # ships will be regarded as provinces of "Others" country
        ships = df.loc[df[self.ISO3].isna(), self.COUNTRY].unique()
        for ship in ships:
            df.loc[df[self.COUNTRY] == ship, [self.ISO3, self.PROVINCE]] = [self.OTHERS, ship]
        return df

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
        iso3 = self._to_iso3(country)[0]
        if iso3 == self.OTHERS:
            df = self._country()
            return df.loc[df[self.ISO3] == iso3]
        url = f"https://storage.covid19datahub.io/country/{iso3}.csv.zip"
        df = self._provide(
            url=url, suffix=f"_{iso3.lower()}", columns=list(self.COL_DICT.keys()), date="date", date_format="%Y-%m-%d")
        df = df.loc[(~df[self.PROVINCE].isna()) & (df[self.CITY].isna())]
        df.loc[:, self.CITY] = df.loc[:, self.CITY].fillna(self.NA)
        return df

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
        iso3 = self._to_iso3(country)[0]
        url = f"https://storage.covid19datahub.io/country/{iso3}.csv.zip"
        df = self._provide(
            url=url, suffix=f"_{iso3.lower()}", columns=list(self.COL_DICT.keys()), date="date", date_format="%Y-%m-%d")
        return df.loc[(df[self.PROVINCE] == province) & (~df[self.CITY].isna())]
