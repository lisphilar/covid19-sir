#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy._deprecated.db_base import _RemoteDatabase


class _OWID(_RemoteDatabase):
    """
    Access "Our World In Data".
    https://github.com/owid/covid-19-data/tree/master/public/data
    https://ourworldindata.org/coronavirus

    Args:
        filename (str): CSV filename to save records
        iso3 (str or None): ignored
    """
    # URL for vaccine data
    URL_V = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/"
    URL_V_REC = f"{URL_V}vaccinations.csv"
    URL_V_LOC = f"{URL_V}locations.csv"
    # URL for PCR data
    URL_P = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/"
    URL_P_REC = f"{URL_P}covid-testing-all-observations.csv"
    # Citation
    CITATION = "Hasell, J., Mathieu, E., Beltekian, D. et al." \
        " A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020)." \
        " https://doi.org/10.1038/s41597-020-00688-8"
    # Column names and data types
    # {"name in database": "name defined in Term class"}
    COL_DICT = {
        "date": Term.DATE,
        "iso_code": Term.ISO3,
        Term.PROVINCE: Term.PROVINCE,
        "vaccines": Term.PRODUCT,
        "total_vaccinations": Term.VAC,
        "total_boosters": Term.VAC_BOOSTERS,
        "people_vaccinated": Term.V_ONCE,
        "people_fully_vaccinated": Term.V_FULL,
        "Cumulative total": Term.TESTS,
    }

    def download(self, verbose):
        """
        Download the dataset from the server and set the list of primary sources.

        Args:
            verbose (int): level of verbosity

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by the first values of self.COL_DICT.values()

        Note:
            If @verbose is equal to or over 1, how to show the list will be explained.
        """
        # Download datasets
        if verbose:
            print("Retrieving datasets from Our World In Data https://github.com/owid/covid-19-data/")
        # Vaccinations
        v_rec_cols = [
            "date", "iso_code", "total_vaccinations", "total_boosters", "people_vaccinated", "people_fully_vaccinated"]
        v_rec_df = pd.read_csv(self.URL_V_REC, usecols=v_rec_cols)
        v_loc_df = pd.read_csv(self.URL_V_LOC, usecols=["iso_code", "vaccines"])
        v_df = v_rec_df.merge(v_loc_df, how="left", on="iso_code")
        # Tests
        pcr_rec_cols = ["ISO code", "Date", "Daily change in cumulative total", "Cumulative total"]
        pcr_df = pd.read_csv(self.URL_P_REC, usecols=pcr_rec_cols)
        pcr_df = pcr_df.rename(columns={"ISO code": "iso_code", "Date": "date"})
        pcr_df["cumsum"] = pcr_df.groupby("iso_code")["Daily change in cumulative total"].cumsum()
        pcr_df = pcr_df.assign(tests=lambda x: x["Cumulative total"].fillna(x["cumsum"]))
        # Combine data (vaccinations/tests)
        df = v_df.merge(pcr_df, how="outer", on=["iso_code", "date"])
        # Location (iso3/province)
        df = df.loc[~df["iso_code"].str.contains("OWID_")]
        df = df.loc[~df["iso_code"].isna()]
        df[self.PROVINCE] = self.NA
        return df
