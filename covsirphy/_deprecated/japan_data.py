#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy._deprecated.cbase import CleaningBase


class JapanData(CleaningBase):
    """Deprecated. Japan-specific dataset.

    Args:
        filename (str or pathlib.path): CSV filename to save the raw dataset
        force (bool): if True, always download the dataset from the server
        verbose (int): level of verbosity

    Note:
        Columns of JapanData.cleaned():
            - Date (pandas.TimeStamp): date
            - Country (pandas.Category): 'Japan'
            - ISO3: ISO3 code
            - Province (pandas.Category): '-' (country level), 'Entering' or province names
            - Confirmed (int): the number of confirmed cases
            - Infected (int): the number of currently infected cases
            - Fatal (int): the number of fatal cases
            - Recovered (int): the number of recovered cases
            - Tests (int): the number of tested persons
            - Moderate (int): the number of cases who requires hospitalization but not severe
            - Severe (int): the number of severe cases
            - Vaccinations (int): cumulative number of vaccinations
            - Vaccinations_boosters (int): cumulative number of booster vaccinations
            - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
            - Vaccinated_full (int): cumulative number of people who received all doses prescribed by the protocol
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
        CleaningBase.C, CleaningBase.CI, CleaningBase.F, CleaningBase.R,
        CleaningBase.TESTS, MODERATE, SEVERE,
        CleaningBase.VAC, CleaningBase.VAC_BOOSTERS, CleaningBase.V_ONCE, CleaningBase.V_FULL,
    ]
    JAPAN_COLS = [
        CleaningBase.DATE, CleaningBase.COUNTRY, CleaningBase.ISO3, CleaningBase.PROVINCE,
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

    @deprecate("JapanData", version="2.26.2-epsilon")
    def __init__(self, filename, force=False, verbose=1):
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        if Path(filename).exists() and not force:
            try:
                self._raw = self._parse_raw(filename, None, [*self.JAPAN_COLS, "Hosp_require"])
            except KeyError:
                self._raw = self._retrieve(filename=filename, verbose=verbose)
        else:
            self._raw = self._retrieve(filename=filename, verbose=verbose)
        self._cleaned_df = self._cleaning()
        self._country = "Japan"
        self._citation = "Hirokazu Takaya (2020-2022), COVID-19 dataset in Japan, GitHub repository, " \
            "https://github.com/lisphilar/covid19-sir/data/japan"
        self.dir_path = Path(filename).parent
        self.verbose = verbose
        # Directory that save the file
        self._dirpath = Path(filename or "input").resolve().parent

    @property
    def country(self):
        """
        str: country name
        """
        return self._country

    def raw_columns(self):
        """
        Return the column names of the raw data.

        Returns:
            list[str]: the list of column names of the raw data
        """
        return self._raw.columns.tolist()

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
        cols_v = ["Vaccinated_1st", "Vaccinated_2nd", "Vaccinated_3rd"]
        c_df = pd.read_csv(self.URL_C, header=0).rename({"Location": "Area"}, axis=1)[cols + cols_v]
        # Download the dataset at province level
        p_df = pd.read_csv(self.URL_P, header=0).rename({"Prefecture": "Area"}, axis=1)[cols]
        # Combine the datasets
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
        rename_dict = {
            "Area": self.PROVINCE,
            "Date": self.DATE,
            "Positive": self.C,
            "Fatal": self.F,
            "Discharged": self.R,
            "Hosp_severe": self.SEVERE,
            "Tested": self.TESTS,
            "Vaccinated_1st": self.V_ONCE,
            "Vaccinated_2nd": self.V_FULL,
            "Vaccinated_3rd": self.VAC_BOOSTERS,
        }
        df = df.rename(rename_dict, axis=1)
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
        v_raw_cols = [self.V_ONCE, self.V_FULL, self.VAC_BOOSTERS]
        v_df = self._raw.rename(rename_dict, axis=1)
        v_df = v_df.loc[:, [self.PROVINCE, self.DATE, *v_raw_cols]]
        v_df[self.DATE] = pd.to_datetime(v_df[self.DATE])
        for col in v_raw_cols:
            v_df[col] = pd.to_numeric(v_df[col], errors="coerce").fillna(0)
        v_1st = v_df.groupby(self.PROVINCE)[self.V_ONCE].cumsum().fillna(0)
        v_2nd = v_df.groupby(self.PROVINCE)[self.V_FULL].cumsum().fillna(0)
        v_3rd = v_df.groupby(self.PROVINCE)[self.VAC_BOOSTERS].cumsum().fillna(0)
        v_sum_df = pd.DataFrame(
            {
                self.PROVINCE: v_df[self.PROVINCE],
                self.DATE: v_df[self.DATE],
                self.VAC: v_1st + v_2nd + v_3rd,
                self.VAC_BOOSTERS: v_3rd,
                self.V_ONCE: v_1st,
                self.V_FULL: v_2nd,
            }
        )
        df = df.drop([*v_raw_cols, self.VAC], axis=1, errors="ignore").merge(
            v_sum_df, how="left", on=[self.PROVINCE, self.DATE]).reset_index(drop=True)
        # Records at country level (Domestic/Airport/Returnee) and entering Japan (Airport/Returnee)
        e_cols = ["Airport", "Returnee"]
        e_df = df.loc[df[self.PROVINCE].isin(e_cols)].groupby(self.DATE).sum()
        e_df[self.PROVINCE] = "Entering"
        c_cols = ["Domestic", "Airport", "Returnee"]
        c_df = df.loc[df[self.PROVINCE].isin(c_cols)].groupby(self.DATE).sum()
        c_df[self.PROVINCE] = self.NA
        df = pd.concat(
            [
                df.loc[~df[self.PROVINCE].isin(c_cols)],
                e_df.reset_index(),
                c_df.reset_index(),
            ],
            ignore_index=True, sort=True)
        # Value columns
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df[self.JAPAN_VALUE_COLS] = df[self.JAPAN_VALUE_COLS].fillna(0).astype(np.int64)
        # Country
        df[self.COUNTRY] = "Japan"
        df[self.ISO3] = "JPN"
        # Update data types to reduce memory
        df[self.AREA_COLUMNS] = df[self.AREA_COLUMNS].astype("category")
        return df.loc[:, self.JAPAN_COLS]

    @deprecate("PCRData.replace()", new="DataLoader.read_dataframe()", version="sigma",
               ref="https://lisphilar.github.io/covid19-sir/markdown/LOADING.html")
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
        df = pd.read_csv(self.URL_M)
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
            df = pd.read_csv(filepath)
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

    def total(self):
        """
        Return a dataframe to show chronological change of number and rates.

        Returns:
            pandas.DataFrame: group-by Date, sum of the values

                Index
                    Date (pd.Timestamp): Observation date
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Fatal per Confirmed (int)
                    - Recovered per Confirmed (int)
                    - Fatal per (Fatal or Recovered) (int)
        """
        df = self.cleaned()
        # Calculate total values at country level if not registered
        c_level_df = df.groupby(self.DATE).sum().reset_index()
        c_level_df[self.PROVINCE] = self.NA
        df = pd.concat([df, c_level_df], axis=0, ignore_index=True)
        df = df.drop_duplicates(subset=[self.DATE, self.PROVINCE])
        df = df.loc[df[self.PROVINCE] == self.NA, :]
        df = df.drop([self.COUNTRY, self.PROVINCE], axis=1)
        df = df.set_index(self.DATE)
        # Calculate rates
        total_series = df.sum(axis=1)
        r_cols = self.RATE_COLUMNS[:]
        df[r_cols[0]] = df[self.F] / total_series
        df[r_cols[1]] = df[self.R] / total_series
        df[r_cols[2]] = df[self.F] / (df[self.F] + df[self.R])
        return df.loc[:, [*self.VALUE_COLUMNS, *r_cols]]

    def countries(self):
        """
        Return names of countries where records are registered.

        Returns:
            list[str]: list of country names
        """
        return [self._country]

    def register_total(self):
        """
        Register total value of all provinces as country level data.

        Returns:
            covsirphy.JapanData: self

        Note:
            If country level data was registered, this will be overwritten.
        """
        # Calculate total values at province level
        clean_df = self.cleaned()
        clean_df = clean_df.loc[clean_df[self.PROVINCE] != self.NA]
        total_df = clean_df.groupby(self.DATE).sum().reset_index()
        total_df[self.COUNTRY] = "Japan"
        total_df[self.ISO3] = "JPN"
        total_df[self.PROVINCE] = self.NA
        # Add/overwrite country level data
        df = clean_df.loc[clean_df[self.PROVINCE] != self.NA]
        df = pd.concat([df, total_df], ignore_index=True, sort=True)
        df[[self.COUNTRY, self.PROVINCE]] = df[[self.COUNTRY, self.PROVINCE]].astype("category")
        self._cleaned_df = df.loc[:, self.JAPAN_COLS]
        return self

    def map(self, country=None, variable="Confirmed", date=None, **kwargs):
        """
        Create colored map to show the values.

        Args:
            country (None): None
            variable (str): variable name to show
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()

        Raises:
            NotImplementedError: @country was specified
        """
        if country is not None:
            raise NotImplementedError("@country cannot be specified, always None.")
        # Date
        date_str = date or self._cleaned_df[self.DATE].max().strftime(self.DATE_FORMAT)
        title = f"{self._country}: the number of {variable.lower()} cases on {date_str}"
        # Country-specific map
        return self._colored_map_country(
            country=self._country, variable=variable, title=title, date=date, **kwargs)
