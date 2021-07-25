#!/usr/bin/env python
# -*- coding: utf-8 -*-

import country_converter as coco
import pandas as pd
from covsirphy.util.term import Term
from covsirphy.loading.db_base import _RemoteDatabase


class _OWID(_RemoteDatabase):
    """
    Access "Our World In Data".
    https://github.com/owid/covid-19-data/tree/master/public/data
    https://ourworldindata.org/coronavirus

    Args:
        filename (str): CSV filename to save records
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
        "location": Term.COUNTRY,
        Term.PROVINCE: Term.PROVINCE,
        "iso_code": Term.ISO3,
        "vaccines": Term.PRODUCT,
        "total_vaccinations": Term.VAC,
        "people_vaccinated": Term.V_ONCE,
        "people_fully_vaccinated": Term.V_FULL,
        "tests": Term.TESTS,
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
            "date", "location", "iso_code", "total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]
        v_rec_df = pd.read_csv(self.URL_V_REC, usecols=v_rec_cols)
        v_loc_df = pd.read_csv(self.URL_V_LOC, usecols=["location", "vaccines"])
        v_df = v_rec_df.merge(v_loc_df, how="left", on="location")
        # Tests
        pcr_rec_cols = ["ISO code", "Date", "Daily change in cumulative total", "Cumulative total"]
        pcr_df = pd.read_csv(self.URL_P_REC, usecols=pcr_rec_cols)
        pcr_df = pcr_df.rename(columns={"ISO code": "iso_code", "Date": "date"})
        pcr_df["cumsum"] = pcr_df.groupby("iso_code")["Daily change in cumulative total"].cumsum()
        pcr_df = pcr_df.assign(tests=lambda x: x["Cumulative total"].fillna(x["cumsum"]))
        # Combine data (vaccinations/tests)
        df = v_df.set_index(["iso_code", "date"])
        df = df.combine_first(pcr_df.set_index(["iso_code", "date"]).loc[:, ["tests"]])
        df = df.reset_index()
        # Location (country/province)
        df["location"] = df["location"].replace(
            {
                # COG
                "Congo": "Republic of the Congo",
            }
        )
        df = df.loc[~df["iso_code"].str.contains("OWID_")]
        df["location"] = df.groupby("iso_code")["location"].bfill()
        df.loc[df["location"] == df["iso_code"], "location"] = None
        df.loc[df["location"].isna(), "location"] = df.loc[df["location"].isna(), "iso_code"].apply(
            lambda x: coco.convert(x, to="name_short", not_found=None))
        df[self.PROVINCE] = self.UNKNOWN
        return df
