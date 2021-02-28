#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from covsirphy.cleaning.country_data import CountryData


class JapanData(CountryData):
    """
    Japan-specific dataset.

    Args:
        filename (str or pathlib.path): CSV filename to save the raw dataset
        force (bool): if True, always download the dataset from the server
        verbose (int): level of verbosity

    Note:
        Columns of JapanData.cleaned():
            - Date (pandas.TimeStamp): date
            - Country (pandas.Category): 'Japan'
            - Province (pandas.Category): '-' (country level), 'Entering' or province names
            - Confirmed (int): the number of confirmed cases
            - Infected (int): the number of currently infected cases
            - Fatal (int): the number of fatal cases
            - Recovered (int): the number of recovered cases
            - Tests (int): the number of tested persons
            - Moderate (int): the number of cases who requires hospitalization but not severe
            - Severe (int): the number of severe cases
            - Vaccinations (int): cumulative number of vaccinations
            - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
            - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
    """
    GITHUB_URL = "https://raw.githubusercontent.com"
    URL_C = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_total.csv"
    URL_P = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_prefecture.csv"
    URL_M = f"{GITHUB_URL}/lisphilar/covid19-sir/master/data/japan/covid_jpn_metadata.csv"
    # Moderate: cases who requires hospitalization but not severe
    MODERATE = "Moderate"
    # Severe
    SEVERE = "Severe"
    # Column names
    JAPAN_VALUE_COLS = [
        CountryData.C, CountryData.CI, CountryData.F, CountryData.R,
        CountryData.TESTS, MODERATE, SEVERE,
        CountryData.VAC, CountryData.V_ONCE, CountryData.V_FULL,
    ]
    JAPAN_COLS = [
        CountryData.DATE, CountryData.COUNTRY, CountryData.PROVINCE,
        *JAPAN_VALUE_COLS,
    ]
    # Meta data
    JAPAN_META_CAT = ["Prefecture", "Admin_Capital", "Admin_Region"]
    JAPAN_META_INT = [
        "Admin_Num", "Area_Habitable", "Area_Total",
        "Clinic_bed_Care", "Clinic_bed_Total",
        "Hospital_bed_Care", "Hospital_bed_Specific", "Hospital_bed_Total",
        "Hospital_bed_Tuberculosis", "Hospital_bed_Type-I", "Hospital_bed_Type-II",
        "Population_Female", "Population_Male", "Population_Total",
    ]
    JAPAN_META_FLT = ["Location_Latitude", "Location_Longitude"]
    JAPAN_META_COLS = [*JAPAN_META_CAT, *JAPAN_META_INT, *JAPAN_META_FLT]

    def __init__(self, filename, force=False, verbose=1):
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        if Path(filename).exists() and not force:
            try:
                self._raw = self.load(filename)
            except KeyError:
                self._raw = self._retrieve(filename=filename, verbose=verbose)
        else:
            self._raw = self._retrieve(filename=filename, verbose=verbose)
        self._cleaned_df = self._cleaning()
        self._country = "Japan"
        self._citation = "Lisphilar (2020), COVID-19 dataset in Japan, GitHub repository, " \
            "https://github.com/lisphilar/covid19-sir/data/japan"
        self.dir_path = Path(filename).parent
        self.verbose = verbose
        # Directory that save the file
        self._dirpath = Path(filename or "input").resolve().parent

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
            print("Retrieving COVID-19 dataset in Japan from https://github.com/lisphilar/covid19-sir/data/japan")
        # Download the dataset at country level
        cols = [
            "Area", "Date", "Positive",
            "Tested", "Discharged", "Fatal", "Hosp_require", "Hosp_severe",
        ]
        cols_v = ["Vaccinated_1st", "Vaccinated_2nd"]
        c_df = self.load(self.URL_C, header=0).rename({"Location": "Area"}, axis=1)[cols + cols_v]
        # Download the datset at province level
        p_df = self.load(self.URL_P, header=0).rename({"Prefecture": "Area"}, axis=1)[cols]
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
        # Moderate
        df[self.MODERATE] = df["Hosp_require"] - df[self.SEVERE]
        # Vaccinations
        v_raw_cols = ["Vaccinated_1st", "Vaccinated_2nd"]
        v_df = self._raw.loc[:, ["Area", "Date", *v_raw_cols]].rename(
            columns={"Area": self.PROVINCE, "Date": self.DATE})
        v_df[self.DATE] = pd.to_datetime(v_df[self.DATE])
        for col in v_raw_cols:
            v_df[col] = pd.to_numeric(v_df[col], errors="coerce").fillna(0)
        v_1st = v_df.groupby(self.PROVINCE)["Vaccinated_1st"].cumsum().fillna(0)
        v_2nd = v_df.groupby(self.PROVINCE)["Vaccinated_2nd"].cumsum().fillna(0)
        v_sum_df = pd.DataFrame(
            {
                self.PROVINCE: v_df[self.PROVINCE],
                self.DATE: v_df[self.DATE],
                self.VAC: v_1st + v_2nd,
                self.V_ONCE: v_1st,
                self.V_FULL: v_2nd,
            }
        )
        df = df.drop(v_raw_cols, axis=1).merge(
            v_sum_df, how="left", on=[self.PROVINCE, self.DATE]).reset_index(drop=True)
        # Records at country level (Domestic/Airport/Returnee) and entering Japan (Airport/Returnee)
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
        # Value columns
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df[self.JAPAN_VALUE_COLS] = df[self.JAPAN_VALUE_COLS].astype(np.int64)
        # Country
        df[self.COUNTRY] = "Japan"
        # Update data types to reduce memory
        df[self.AREA_COLUMNS] = df[self.AREA_COLUMNS].astype("category")
        return df.loc[:, self.JAPAN_COLS]

    def set_variables(self):
        raise NotImplementedError

    def _retrieve_meta(self, filename, verbose=1):
        """
        Retrieve meta data from server.

        Args:
            filename (str or pathlib.path): CSV filename to save the raw dataset
            verbose (int): level of verbosity

        Returns:
            pandas.DataFrame: raw dataset
        """
        # Show URL
        if verbose:
            print("Retrieving Metadata of Japan dataset from https://github.com/lisphilar/covid19-sir/data/japan")
        df = self.load(self.URL_M)
        df.to_csv(filename, index=False)
        return df

    def meta(self, basename="covid_japan_metadata.csv", cleaned=True, force=False):
        """
        Return metadata of Japan-specific dataset.

        Args:
            basename (str): basename of the CSV file to save the raw dataset
            cleaned (bool): return cleaned (True) or raw (False) dataset
            force (bool): if True, always download the dataset from the server

        Returns:
            pandas.DataFrame: (cleaned or raw) dataset
                Index
                    reset index
                Columns for cleaned dataset,
                    - Prefecture (pandas.Category)
                    - Admin_Capital (pandas.Category)
                    - Admin_Region (pandas.Category)
                    - Admin_Num (int)
                    - Area_Habitable (int)
                    - Area_Total (int)
                    - Clinic_bed_Care (int)
                    - Clinic_bed_Total (int)
                    - Hospital_bed_Care (int)
                    - Hospital_bed_Specific (int)
                    - Hospital_bed_Total (int)
                    - Hospital_bed_Tuberculosis (int)
                    - Hospital_bed_Type-I (int)
                    - Hospital_bed_Type-II (int)
                    - Population_Female (int)
                    - Population_Male (int)
                    - Population_Total (int)
                    - Location_Latitude (float)
                    - Location_Longitude (float)

        Note:
            Please refer to https://github.com/lisphilar/covid19-sir/tree/master/data
        """
        filepath = self.dir_path.joinpath(basename)
        if filepath.exists() and not force:
            df = self.load(filepath)
        else:
            df = self._retrieve_meta(filename=filepath, verbose=self.verbose)
        if not cleaned:
            return df
        # Data cleaning
        df["Title"] = df["Category"].str.cat(df["Item"], sep="_")
        df = df.pivot_table(
            values="Value", index="Prefecture", columns="Title", aggfunc="last")
        df.columns.name = None
        df = df.reset_index()
        df[self.JAPAN_META_CAT] = df[self.JAPAN_META_CAT].astype("category")
        df[self.JAPAN_META_INT] = df[self.JAPAN_META_INT].astype(np.int64)
        df[self.JAPAN_META_FLT] = df[self.JAPAN_META_FLT].astype(np.float64)
        df = df.sort_values("Admin_Num").reset_index(drop=True)
        return df.loc[:, self.JAPAN_META_COLS]
