#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy.downloading._db import _DataBase


class _OWID(_DataBase):
    """
    Access "Our World In Data" server.
    https://github.com/owid/covid-19-data/tree/master/public/data
    https://ourworldindata.org/coronavirus
    """
    # File title without extensions and suffix
    TITLE = "ourworldindata"
    # Dictionary of column names
    COL_DICT = {
        # Vaccine
        "iso_code": Term.ISO3,
        "date": Term.DATE,
        "vaccines": Term.PRODUCT,
        "total_vaccinations": Term.VAC,
        "total_boosters": Term.VAC_BOOSTERS,
        "people_vaccinated": Term.V_ONCE,
        "people_fully_vaccinated": Term.V_FULL,
        # Tests
        "ISO code": Term.ISO3,
        "Date": Term.DATE,
        "Cumulative total": Term.TESTS,
    }
    # Stdout when downloading (shown at most one time)
    STDOUT = "Retrieving datasets from Our World In Data https://github.com/owid/covid-19-data/"
    # Citation
    CITATION = "Hasell, J., Mathieu, E., Beltekian, D. et al." \
        " A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020)." \
        " https: //doi.org/10.1038/s41597-020-00688-8"

    def _country(self):
        """Returns country-level data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (str): NAs
                    - City (str): NAs
                    - Tests (numpy.float64): the number of tests
                    - Product (numpy.int64): vaccine product names
                    - Vaccinations (numpy.int64): cumulative number of vaccinations
                    - Vaccinations_boosters (numpy.int64): cumulative number of booster vaccinations
                    - Vaccinated_once (numpy.int64): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (numpy.int64): cumulative number of people who received all doses prescribed by the protocol
        """
        # URLs
        URL_V = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/"
        URL_V_REC = f"{URL_V}vaccinations.csv"
        URL_V_LOC = f"{URL_V}locations.csv"
        # URL for PCR data
        URL_P = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/"
        URL_P_REC = f"{URL_P}covid-testing-all-observations.csv"
        # Vaccine
        v_rec_cols = [
            "date", "iso_code", "total_vaccinations", "total_boosters", "people_vaccinated", "people_fully_vaccinated"]
        v_rec_df = self._provide(
            url=URL_V_REC, suffix="_vaccine", columns=v_rec_cols, date="date", date_format="%Y-%m-%d")
        v_loc_df = self._provide(
            url=URL_V_LOC, suffix="_vaccine_locations", columns=["iso_code", "vaccines"], date=None, date_format="%Y-%m-%d")
        v_df = v_rec_df.merge(v_loc_df, how="left", on=self.ISO3)
        # Tests
        pcr_rec_cols = ["ISO code", "Date", "Daily change in cumulative total", "Cumulative total"]
        pcr_df = self._provide(url=URL_P_REC, suffix="", columns=pcr_rec_cols, date="Date", date_format="%Y-%m-%d")
        pcr_df["cumsum"] = pcr_df.groupby(self.ISO3)["Daily change in cumulative total"].cumsum()
        pcr_df = pcr_df.assign(tests=lambda x: x[self.TESTS].fillna(x["cumsum"]))
        pcr_df.rename(columns={"tests": self.TESTS})
        pcr_df = pcr_df.loc[:, [self.ISO3, self.DATE, self.TESTS]]
        # Combine data (vaccinations/tests)
        df = v_df.merge(pcr_df, how="outer", on=[self.ISO3, self.DATE])
        # Location (iso3/province)
        df = df.loc[~df[self.ISO3].str.contains("OWID_")]
        df = df.loc[~df[self.ISO3].isna()]
        df[self.PROVINCE] = self.NA
        df[self.CITY] = self.NA
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
                    - City (str): NAs
                    - Tests (numpy.float64): the number of tests
                    - Product (numpy.int64): vaccine product names
                    - Vaccinations (numpy.int64): cumulative number of vaccinations
                    - Vaccinations_boosters (numpy.int64): cumulative number of booster vaccinations
                    - Vaccinated_once (numpy.int64): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (numpy.int64): cumulative number of people who received all doses prescribed by the protocol
        """
        return self._empty()

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
                    - Tests (numpy.float64): the number of tests
                    - Product (numpy.int64): vaccine product names
                    - Vaccinations (numpy.int64): cumulative number of vaccinations
                    - Vaccinations_boosters (numpy.int64): cumulative number of booster vaccinations
                    - Vaccinated_once (numpy.int64): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (numpy.int64): cumulative number of people who received all doses prescribed by the protocol
        """
        return self._empty()

    def _empty(self):
        """Return empty dataframe.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (str): province/state/prefecture names
                    - City (str): city names
                    - Tests (numpy.float64): the number of tests
                    - Product (numpy.int64): vaccine product names
                    - Vaccinations (numpy.int64): cumulative number of vaccinations
                    - Vaccinations_boosters (numpy.int64): cumulative number of booster vaccinations
                    - Vaccinated_once (numpy.int64): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (numpy.int64): cumulative number of people who received all doses prescribed by the protocol
        """
        columns = [
            self.DATE, self.ISO3, self.PROVINCE, self.CITY, self.TESTS, self.PRODUCT, self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL]
        return pd.DataFrame(columns=columns)
