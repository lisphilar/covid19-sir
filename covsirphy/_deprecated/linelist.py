#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy._deprecated.cbase import CleaningBase


class LinelistData(CleaningBase):
    """
    Deprecated. Linelist of case reports.

    Args:
        filename (str or pathlib.path): CSV filename to save the raw dataset
        force (bool): if True, always download the dataset from the server
        verbose (int): level of verbosity
    """
    GITHUB_URL = "https://raw.githubusercontent.com"
    URL = f"{GITHUB_URL}/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.tar.gz"
    # Column names
    AGE = "Age"
    SEX = "Sex"
    HOSPITAL_DATE = "Hospitalized_date"
    CONFIRM_DATE = "Confirmation_date"
    SYMPTOM = "Symptoms"
    CHRONIC = "Chronic_disease"
    OUTCOME = "Outcome"
    OUTCOME_DATE = "Outcome_date"
    R_DATE = "Recovered_date"
    F_DATE = "Fatal_date"
    LINELIST_COLS = [
        CleaningBase.COUNTRY, CleaningBase.PROVINCE,
        HOSPITAL_DATE, CONFIRM_DATE, OUTCOME_DATE,
        CleaningBase.C, CleaningBase.CI, CleaningBase.R, CleaningBase.F,
        SYMPTOM, CHRONIC, AGE, SEX,
    ]
    # Raw dataset
    RAW_COL_DICT = {
        "age": AGE,
        "sex": SEX,
        "province": CleaningBase.PROVINCE,
        "country": CleaningBase.COUNTRY,
        "date_admission_hospital": HOSPITAL_DATE,
        "date_confirmation": CONFIRM_DATE,
        "symptoms": SYMPTOM,
        "chronic_disease": CHRONIC,
        "outcome": OUTCOME,
        "date_death_or_discharge": OUTCOME_DATE,
    }

    @deprecate("LinelistData()", version="2.21.0-theta")
    def __init__(self, filename, force=False, verbose=1):
        self._filename = filename
        raw_df = self._read_raw(filename, force, verbose)
        self._cleaned_df = self._cleaning(raw_df)
        self._citation = "Xu, B., Gutierrez, B., Mekaru, S. et al. " \
            "Epidemiological data from the COVID-19 outbreak, real-time case information. " \
            "Sci Data 7, 106 (2020). https://doi.org/10.1038/s41597-020-0448-0"

    @property
    def raw(self):
        """
        pandas.DataFrame: raw dataset
        """
        return self._read_raw(self._filename, force=False, verbose=0)

    def _read_raw(self, filename, force, verbose):
        """
        Get raw dataset from a CSV file or retrieve the dataset from server.

        Args:
            filename (str or pathlib.path): CSV filename to save the raw dataset
            force (bool): whether force retrieving from server or not when we have CSV file
            verbose (int): level of verbosity

        Returns:
            pd.DataFrame: raw dataset
        """
        if Path(filename).exists() and not force:
            return self._parse_raw(filename, None, list(self.RAW_COL_DICT))
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        return self._retrieve(filename=filename, verbose=verbose)

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
                "Retrieving linelist from Open COVID-19 Data Working Group repository: https://github.com/beoutbreakprepared/nCoV2019")
        # Download the dataset
        df = pd.read_csv(self.URL, header=1, usecols=list(self.RAW_COL_DICT))
        # Save the raw data
        df.to_csv(filename, index=False)
        return df

    def _cleaning(self, raw_df):
        """
        Perform data cleaning of the raw data.

        Args:
            raw_df (pandas.DataFrame): raw data

        Returns:
            pandas.DataFrame: cleaned data
        """
        df = raw_df.copy()
        # Rename columns
        df = df.rename(self.RAW_COL_DICT, axis=1)
        # Location
        df = df.dropna(subset=[self.COUNTRY])
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.NA)
        # Date
        for col in [self.HOSPITAL_DATE, self.CONFIRM_DATE, self.OUTCOME_DATE]:
            df[col] = pd.to_datetime(
                df[col], infer_datetime_format=True, errors="coerce")
        df = df.loc[~df[self.CONFIRM_DATE].isna()]
        # Outcome
        df[self.C] = ~df[self.CONFIRM_DATE].isna()
        df[self.R] = df[self.OUTCOME].str.lower().isin(
            [
                "recovered", "discharge", "discharged", "released from quarantine",
                "discharged from hospital", "not hospitalized"
            ]
        )
        df[self.F] = df[self.OUTCOME].str.lower().isin(
            ["deceased", "died", "death", "dead"]
        )
        df[self.CI] = df[[self.C, self.R, self.F]].apply(
            lambda x: x[0] and not x[1] and x[2], axis=1)
        # Symptoms
        df[self.SYMPTOM] = df[self.SYMPTOM].str.replace(", ", ":")
        # Chronic disease
        df[self.CHRONIC] = df[self.CHRONIC].str.replace(", ", ":")
        df.loc[df[self.CHRONIC].str.contains(
            "http", na=False), self.CHRONIC] = None
        # Age
        df[self.AGE] = pd.to_numeric(df[self.AGE], errors="coerce")
        # Sex
        df[self.SEX] = df[self.SEX].fillna(self.NA)
        # Update data types to reduce memory
        cat_cols = [
            self.AGE, self.SEX, self.SYMPTOM, self.CHRONIC, *self.AREA_COLUMNS]
        df[cat_cols] = df[cat_cols].astype("category")
        # Select columns
        return df.loc[:, self.LINELIST_COLS]

    def layer(self, **kwargs):
        raise NotImplementedError

    def subset(self, country, province=None):
        """
        Return subset of the country/province.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Hospitalized_date (pandas.TimeStamp or NT)
                    - Confirmation_date (pandas.TimeStamp or NT)
                    - Outcome_date (pandas.TimeStamp or NT)
                    - Confirmed (bool)
                    - Infected (bool)
                    - Recovered (bool)
                    - Fatal (bool)
                    - Symtoms (str)
                    - Chronic_disease (str)
                    - Age (int or None)
                    - Sex (str)
        """
        df = self._cleaned_df.copy()
        # Subset by country name
        country = self.ensure_country_name(country)
        df = df.loc[df[self.COUNTRY] == country]
        # Subset by province name
        if province not in (None, self.NA):
            df = df.loc[df[self.PROVINCE] == province]
        # Check records are registered
        country_alias = self.ensure_country_name(country)
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province)
        df = df.drop([self.COUNTRY, self.PROVINCE], axis=1)
        return df.reset_index(drop=True)

    def closed(self, outcome="Recovered"):
        """
        Return subset of global outcome data (recovered/fatal).

        Args:
            outcome (str): 'Recovered' or 'Fatal'

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Country (pandas.Category): country name
                    - Province (pandas.Category): province name or "-"
                    - Hospitalized_date (pandas.TimeStamp or NT)
                    - Confirmation_date (pandas.TimeStamp)
                    - Recovered_date (pandas.TimeStamp): if outcome is Recovered
                    - Fatal_date (pandas.TimeStamp): if outcome is Fatal
                    - Symptoms (str)
                    - Chronic_disease (str)
                    - Age (int or None)
                    - Sex (str)
        """
        column_dict = {self.R: self.R_DATE, self.F: self.F_DATE}
        if outcome not in column_dict:
            raise KeyError(
                f"@outcome should be selected from {self.R} or {self.F}, but {outcome} was applied.")
        df = self._cleaned_df.copy()
        df = df.loc[df[outcome]]
        # Subset by date
        df = df.loc[df[self.CONFIRM_DATE] < df[self.OUTCOME_DATE]]
        # Column names
        df = df.rename({self.OUTCOME_DATE: column_dict[outcome]}, axis=1)
        df = df.drop([self.C, self.CI, self.F, self.R], axis=1)
        return df.reset_index(drop=True)

    def recovery_period(self):
        """
        Calculate median value of recovery period (from confirmation to recovery).

        Returns:
            int: recovery period [days]
        """
        df = self.closed(outcome=self.R)
        series = (df[self.R_DATE] - df[self.CONFIRM_DATE]).dt.days
        return int(series.median())

    def total(self):
        """
        This is not defined for this child class.
        """
        raise NotImplementedError
