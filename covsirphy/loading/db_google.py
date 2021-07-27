#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from unidecode import unidecode
from covsirphy.util.term import Term
from covsirphy.loading.db_base import _RemoteDatabase


class _GoogleOpenData(_RemoteDatabase):
    """
    Access "COVID-19 Open Data by Google Cloud Platform".
    https://github.com/GoogleCloudPlatform/covid-19-open-data
    https://goo.gle/covid-19-open-data

    Args:
        filename (str): CSV filename to save records
    """
    # URL for mobility data
    URL_M = "https://storage.googleapis.com/covid19-open-data/v3/mobility.csv"
    # URL for index data (country names etc.)
    URL_I = "https://storage.googleapis.com/covid19-open-data/v3/index.csv"
    # Citation
    CITATION = "O. Wahltinez and others (2020)," \
        " COVID-19 Open-Data: curating a fine-grained, global-scale data repository for SARS-CoV-2, " \
        " Work in progress, https://goo.gle/covid-19-open-data"
    # Column names and data types
    # {"name in database": "name defined in Term class"}
    _MOBILITY_COLS_RAW_INT = [
        "mobility_grocery_and_pharmacy",
        "mobility_parks",
        "mobility_transit_stations",
        "mobility_retail_and_recreation",
        "mobility_residential",
        "mobility_workplaces",
    ]
    MOBILITY_VARS = [v.capitalize() for v in _MOBILITY_COLS_RAW_INT]
    COL_DICT = {
        "date": Term.DATE,
        "country_name": Term.COUNTRY,
        Term.PROVINCE: Term.PROVINCE,
        "iso_3166_1_alpha_3": Term.ISO3,
        **{v: v.capitalize() for v in _MOBILITY_COLS_RAW_INT},
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
            print("Retrieving datasets from COVID-19 Open Data by Google Cloud Platform https://github.com/GoogleCloudPlatform/covid-19-open-data")
        # Index
        i_cols = ["location_key", "country_name", "subregion1_name", "subregion2_name", "iso_3166_1_alpha_3"]
        i_df = pd.read_csv(self.URL_I, usecols=i_cols)
        # Mobility
        m_df = pd.read_csv(self.URL_M)
        m_df = (m_df.set_index(["date", "location_key"]) + 100).reset_index()
        # Combine data
        df = m_df.merge(i_df, how="left", on="location_key")
        # Location (country/province)
        df = df.loc[df["subregion2_name"].isna()]
        df[self.PROVINCE] = df["subregion1_name"].fillna(self.UNKNOWN).apply(unidecode)
        df["country_name"] = df["country_name"].replace(
            {
                # CIV
                "Ivory Coast": "Cote d'Ivoire",
            }
        )
        return df
