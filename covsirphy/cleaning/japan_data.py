#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from covsirphy.cleaning.country_data import CountryData


class JapanData(CountryData):
    """
    Linelist of case reports.

    Args:
        filename (str or pathlib.path): CSV filename to save the raw dataset
        force (bool): if True, always download the dataset from the server
        verbose (int): level of verbosity

    Notes:
        Columns of JapanData.cleaned():
            - Date (pandas.TimeStamp): date
            - Country (str): 'Japan'
            - Province (str): '-' (country level), 'Entering' or province names
            - Confirmed (int): the number of confirmed cases
            - Infected (int): the number of currently infected cases
            - Fatal (int): the number of fatal cases
            - Recovered (int): the number of recovered cases
            - Tested (int): the number of tested persons
            - Moderate (int): the number of cases who requires hospitalization but not severe
            - Severe (int): the number of severe cases
    """
    GITHUB_URL = "https://raw.githubusercontent.com"
    URL_C = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_total.csv"
    URL_P = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_prefecture.csv"
    # Moderate: cases who requires hospitalization but not severe
    MODERATE = "Moderate"
    # Severe
    SEVERE = "Severe"
    # Column names
    JAPAN_VALUE_COLS = [
        CountryData.C, CountryData.CI, CountryData.F, CountryData.R,
        CountryData.TESTS, MODERATE, SEVERE,
    ]
    JAPAN_COLS = [
        CountryData.DATE, CountryData.COUNTRY, CountryData.PROVINCE,
        *JAPAN_VALUE_COLS,
    ]

    def __init__(self, filename, force=False, verbose=1):
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        if Path(filename).exists() and not force:
            self._raw = self.load(filename)
        else:
            self._raw = self._retrieve(filename=filename, verbose=verbose)
        self._cleaned_df = self._cleaning()
        self._country = "Japan"
        self._citation = "Lisphilar (2020), COVID-19 dataset in Japan, GitHub repository, " \
            "https://github.com/lisphilar/covid19-sir/data/japan"

    def _retrieve(self, filename, verbose=1):
        """
        Retrieve the dataset from server.

        Args:
            filename (str or pathlib.path): CSV filename to save the raw dataset
            verbose (int): level of verbosity

        Returns:
            pd.DataFrame: raw dataset
        """
        # Show URL
        if verbose:
            print(
                "Retrieving COVID-19 dataset in Japan from https://github.com/lisphilar/covid19-sir/data/japan")
        # Download the dataset at country level
        cols = [
            "Area", "Date", "Positive",
            "Tested", "Discharged", "Fatal", "Hosp_require", "Hosp_severe",
        ]
        c_df = self.load(self.URL_C, header=0).rename(
            {"Location": "Area"}, axis=1)[cols]
        # Download the datset at province level
        p_df = self.load(self.URL_P, header=0).rename(
            {"Prefecture": "Area"}, axis=1)[cols]
        # Combine the datsets
        df = pd.concat([c_df, p_df], axis=0, ignore_index=True, sort=True)
        # Save the raw data
        df.to_csv(filename, index=False)
        return df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            pandas.DataFrame: cleaned data
        """
        df = self._raw.copy()
        # Rename columns
        df = df.rename(
            {
                "Area": self.PROVINCE,
                "Date": self.DATE,
                "Positive": self.C,
                "Fatal": self.F,
                "Discharged": self.R,
                "Hosp_severe": self.SEVERE,
                "Tested": self.TESTS
            },
            axis=1
        )
        # Date
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Fill NA values
        for col in [self.C, self.F, self.R, self.SEVERE, "Hosp_require", self.TESTS]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.groupby(self.PROVINCE).apply(
            lambda x: x.set_index(self.DATE).resample("D").interpolate("linear", limit_direction="both"))
        df = df.fillna(0).drop(self.PROVINCE, axis=1).reset_index()
        df = df.sort_values(self.DATE).reset_index(drop=True)
        # Records at country level (Domestic/Airport/Returnee) and entering Japan(Airport/Returnee)
        e_cols = ["Airport", "Returnee"]
        e_df = df.loc[df[self.PROVINCE].isin(e_cols)].groupby(self.DATE).sum()
        e_df[self.PROVINCE] = "Entering"
        c_cols = ["Domestic", "Airport", "Returnee"]
        c_df = df.loc[df[self.PROVINCE].isin(c_cols)].groupby(self.DATE).sum()
        c_df[self.PROVINCE] = self.UNKNOWN
        df = pd.concat(
            [
                df.loc[~df[self.PROVINCE].isin(c_cols)],
                e_df.reset_index(),
                c_df.reset_index(),
            ],
            ignore_index=True, sort=True)
        # Moderate
        df[self.MODERATE] = df["Hosp_require"] - df[self.SEVERE]
        # Value columns
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df[self.JAPAN_VALUE_COLS] = df[self.JAPAN_VALUE_COLS].astype(np.int64)
        # Country
        df[self.COUNTRY] = "Japan"
        return df.loc[:, self.JAPAN_COLS]

    def set_variables(self):
        raise NotImplementedError
