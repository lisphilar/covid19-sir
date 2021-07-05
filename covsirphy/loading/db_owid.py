#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    # Citation
    CITATION = "Hasell, J., Mathieu, E., Beltekian, D. et al." \
        " A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020)." \
        " https://doi.org/10.1038/s41597-020-00688-8"
    # Column names and data types
    # {"name in database": ("name defined in Term class", "data type")}
    COL_DICT = {
        "date": (Term.DATE, "object"),
        "location": (Term.COUNTRY, "object"),
        Term.PROVINCE: (Term.PROVINCE, "object"),
        "iso_code": (Term.ISO3, "object"),
        "vaccines": (Term.PRODUCT, "object"),
        "total_vaccinations": (Term.VAC, "int"),
        "people_vaccinated": (Term.V_ONCE, "int"),
        "people_fully_vaccinated": (Term.V_FULL, "int"),
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
            print("Retrieving datasets from Our World In Data https://covid19datahub.io/")
        # Vaccination
        v_rec_cols = [
            "date", "location", "iso_code", "total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]
        v_rec_df = pd.read_csv(self.URL_V_REC, usecols=v_rec_cols)
        v_loc_df = pd.read_csv(self.URL_V_LOC, usecols=["location", "vaccines"])
        v_df = v_rec_df.merge(v_loc_df, how="left", on="location")
        v_df[self.PROVINCE] = self.UNKNOWN
        return v_df
