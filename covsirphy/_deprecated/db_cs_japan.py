#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy._deprecated.db_base import _RemoteDatabase


class _CSJapan(_RemoteDatabase):
    """
    Access "COVID-19 Dataset in Japan.
    https://github.com/lisphilar/covid19-sir/tree/master/data

    Args:
        filename (str): CSV filename to save records
        iso3 (str or None): ignored
    """
    # URL
    GITHUB_URL = "https://raw.githubusercontent.com"
    URL_C = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_total.csv"
    URL_P = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_prefecture.csv"
    # Citation
    CITATION = "Hirokazu Takaya (2020-2022), COVID-19 dataset in Japan, GitHub repository, " \
        "https://github.com/lisphilar/covid19-sir/data/japan"
    # Column names and data types
    # {"name in database": "name defined in Term class"}
    COL_DICT = {
        "Date": Term.DATE,
        Term.COUNTRY: Term.COUNTRY,
        "Area": Term.PROVINCE,
        Term.ISO3: Term.ISO3,
        "Positive": Term.C,
        "Fatal": Term.F,
        "Discharged": Term.R,
        "Hosp_require": "Hosp_require",
        Term.MODERATE: Term.MODERATE,
        "Hosp_severe": Term.SEVERE,
        "Tested": Term.TESTS,
        Term.VAC: Term.VAC,
        Term.VAC_BOOSTERS: Term.VAC_BOOSTERS,
        Term.V_ONCE: Term.V_ONCE,
        Term.V_FULL: Term.V_FULL,
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
            print("Retrieving COVID-19 dataset from https://github.com/lisphilar/covid19-sir/data/")
        # Domestic/Airport/Returnee
        dar_value_cols = ["Positive", "Tested", "Discharged", "Fatal", "Hosp_require", "Hosp_severe"]
        dar_cols = [*dar_value_cols, "Date", "Location", "Vaccinated_1st", "Vaccinated_2nd", "Vaccinated_3rd"]
        dar_df = pd.read_csv(self.URL_C, usecols=dar_cols)
        dar_df = dar_df.rename(columns={"Location": "Area"}).set_index("Date")
        # Country level data
        c_df = dar_df.groupby("Date").sum().reset_index()
        c_df["Area"] = self.NA
        # Entering (= Airport + Returnee)
        e_df = dar_df.loc[dar_df["Area"].isin(["Airport", "Returnee"])].groupby("Date").sum().reset_index()
        e_df["Area"] = "Entering"
        # Province level data
        p_cols = [*dar_value_cols, "Date", "Prefecture"]
        p_df = pd.read_csv(self.URL_P, usecols=p_cols)
        p_df = p_df.rename(columns={"Prefecture": "Area"})
        # Combine
        df = pd.concat([c_df, e_df, p_df], axis=0, ignore_index=True, sort=True)
        # Set additional columns
        df[self.COUNTRY] = "Japan"
        df[self.ISO3] = "JPN"
        df[self.MODERATE] = df["Hosp_require"] - df["Hosp_severe"]
        df[self.V_ONCE] = df["Vaccinated_1st"].cumsum()
        df[self.V_FULL] = df["Vaccinated_2nd"].cumsum()
        df[self.VAC_BOOSTERS] = df["Vaccinated_3rd"].cumsum()
        df[self.VAC] = df[[self.V_ONCE, self.V_FULL, self.VAC_BOOSTERS]].sum(axis=1)
        return df
