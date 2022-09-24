#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy.downloading._db import _DataBase


class _CSJapan(_DataBase):
    """
    Access "COVID-19 Dataset in Japan - CovsirPhy project" server.
    https://github.com/lisphilar/covid19-sir/tree/master/data
    """
    GITHUB_URL = "https://raw.githubusercontent.com"
    URL_C = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_total.csv"
    URL_P = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_prefecture.csv"
    # File title without extensions and suffix
    TITLE = "covid_japan"
    # Dictionary of column names
    COL_DICT = {
        "Date": Term.DATE,
        "Prefecture": Term.PROVINCE,
        "Positive": Term.C,
        "Fatal": Term.F,
        "Discharged": Term.R,
        "Tested": Term.TESTS,
    }
    # Stdout when downloading (shown at most one time)
    STDOUT = "Retrieving COVID-19 dataset from https://github.com/lisphilar/covid19-sir/data/"
    # Citation
    CITATION = "Hirokazu Takaya (2020-2022), COVID-19 dataset in Japan, GitHub repository, " \
        "https://github.com/lisphilar/covid19-sir/data/japan"
    # All columns
    _all_columns = [
        Term.DATE, Term.ISO3, Term.PROVINCE, Term.CITY,
        Term.C, Term.F, Term.R, Term.TESTS, Term.VAC, Term.VAC_BOOSTERS, Term.V_ONCE, Term.V_FULL]

    def _country(self):
        """Returns country-level data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): "Japan"
                    - Province (str): NAs
                    - City (str): NAs
                    - Confirmed (numpy.float64): the number of confirmed cases
                    - Fatal (numpy.float64): the number of fatal cases
                    - Recovered (numpy.float64): the number of recovered cases
                    - Tests (numpy.float64): the number of tests
                    - Vaccinations (numpy.float64): cumulative number of vaccinations
                    - Vaccinations_boosters (numpy.float64): cumulative number of booster vaccinations
                    - Vaccinated_once (numpy.float64): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (numpy.float64): cumulative number of people who received all doses prescribed by the protocol
        """
        cols = [
            "Date", "Location", "Positive", "Tested", "Discharged", "Fatal", "Vaccinated_1st", "Vaccinated_2nd", "Vaccinated_3rd"]
        df = self._provide(url=self.URL_C, suffix="", columns=cols, date="Date", date_format="%Y-%m-%d")
        df = df.groupby("Date").sum(numeric_only=True).reset_index()
        df[self.ISO3] = "JPN"
        df[self.PROVINCE] = self.NA
        df[self.CITY] = self.NA
        df[self.V_ONCE] = df["Vaccinated_1st"].cumsum()
        df[self.V_FULL] = df["Vaccinated_2nd"].cumsum()
        df[self.VAC_BOOSTERS] = df["Vaccinated_3rd"].cumsum()
        df[self.VAC] = df[[self.V_ONCE, self.V_FULL, self.VAC_BOOSTERS]].sum(axis=1)
        return df.loc[:, self._all_columns]

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
                    - City (str): NAs
                    - Confirmed (numpy.float64): the number of confirmed cases
                    - Fatal (numpy.float64): the number of fatal cases
                    - Recovered (numpy.float64): the number of recovered cases
                    - Population (numpy.int64): population values
                    - Tests (numpy.float64): the number of tests
        """
        if self._to_iso3(country)[0] != "JPN":
            return pd.DataFrame(columns=[
                self.DATE, self.ISO3, self.PROVINCE, self.CITY,
                self.C, self.F, self.R, self.TESTS, self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL])
        cols = ["Date", "Prefecture", "Positive", "Tested", "Discharged", "Fatal"]
        df = self._provide(url=self.URL_P, suffix="_prefecture", columns=cols, date="Date", date_format="%Y-%m-%d")
        df[self.ISO3] = "JPN"
        df[self.CITY] = self.NA
        df[self.V_ONCE] = pd.NA
        df[self.V_FULL] = pd.NA
        df[self.VAC_BOOSTERS] = pd.NA
        df[self.VAC] = pd.NA
        return df.loc[:, self._all_columns]

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
                    - ISO3 (str): "Japan"
                    - Province (str): province/state/prefecture names
                    - City (str): city names
                    - Confirmed (numpy.float64): the number of confirmed cases
                    - Fatal (numpy.float64): the number of fatal cases
                    - Recovered (numpy.float64): the number of recovered cases
                    - Tests (numpy.float64): the number of tests
                    - Vaccinations (numpy.float64): cumulative number of vaccinations
                    - Vaccinations_boosters (numpy.float64): cumulative number of booster vaccinations
                    - Vaccinated_once (numpy.float64): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (numpy.float64): cumulative number of people who received all doses prescribed by the protocol
        """
        return pd.DataFrame(columns=self._all_columns)
