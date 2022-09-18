#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import country_converter as coco
import numpy as np
import pandas as pd
import wbdata
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy._deprecated.cbase import CleaningBase


class PopulationPyramidData(CleaningBase):
    """
    Population pyramid dataset.
    World Bank Group (2020), World Bank Open Data, https://data.worldbank.org/

    Args:
        filename (str or None): CSV filename to save the dataset
        force (bool): if True, always download the dataset from the server
        verbose (int): level of verbosity

    Returns:
        If @filename is None, empty dataframe will be set as raw data.
        If @citation is None, citation will be empty string.
    """
    # Indicators of the raw dataset
    AGE_KEYS = [
        "0004", "0509", "1014", "1519", "2024", "2529", "3034", "3539",
        "4044", "4549", "5054", "5559", "6064", "6569", "7579", "80UP",
    ]
    INDICATOR_DICT = {
        f"SP.POP.{age}.{sex}": f"{age[:2]}-{age[2:]}-{sex}"
        for age in AGE_KEYS for sex in ["MA", "FE"]
    }
    ELDEST = 122
    # Columns
    SEX = "Sex"
    YEAR = "Year"
    AGE = "Age"
    PYRAMID_COLS = [CleaningBase.COUNTRY, YEAR, SEX, AGE, CleaningBase.N]
    PORTION = "Per_total"

    @deprecate("CleaningBase", version="2.27.0-zeta")
    def __init__(self, filename, force=False, verbose=1):
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        if Path(filename).exists() and not force:
            self._raw = pd.read_csv(filename)
        else:
            self._raw = pd.DataFrame(columns=self.PYRAMID_COLS)
        self._cleaned_df = self._raw.copy()
        self._citation = "World Bank Group (2020), World Bank Open Data, https://data.worldbank.org/"
        self._filename = filename
        self.verbose = verbose

    def _retrieve_from_server(self, country):
        """
        Retrieve the dataset of the country from the server.

        Args:
            country (str): country name

        Returns:
            pandas.DataFrame: retrieved data
                Index
                    reset index
                Columns
                    - Country (object): country name
                    - Year (int): year
                    - Sex (object): Female/Male
                    - Age (object): age
                    - Population (object): population value
        """
        if self.verbose:
            print(
                f"Retrieving population pyramid dataset ({country}) from https://data.worldbank.org/")
        # Retrieve from World Bank Open Data
        iso3_code = coco.convert(country, to="ISO3", not_found=None)
        try:
            df = wbdata.get_dataframe(
                self.INDICATOR_DICT, country=iso3_code, convert_date=True)
        except RuntimeError:
            raise SubsetNotFoundError(country=country) from None
        # Preprocessing (-> Country, Population, Min, Max, Sex, Year)
        df = df.stack().reset_index()
        df.insert(0, self.COUNTRY, country)
        df.columns = [self.COUNTRY, "Date", "Attribute", self.N]
        df2 = df["Attribute"].str.split("-", expand=True)
        df2.columns = ["Min", "Max", self.SEX]
        df = pd.concat([df.drop("Attribute", axis=1), df2], axis=1)
        df["Max"] = df["Max"].replace("UP", self.ELDEST)
        for col in [self.N, "Min", "Max"]:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        df[self.SEX].replace({"FE": "Female", "MA": "Male"}, inplace=True)
        df[self.YEAR] = df["Date"].dt.year
        df = df.drop("Date", axis=1)
        # Preprocessing (-> Country, Year, Sex, Age, Population)
        df[self.AGE] = df[["Min", "Max"]].apply(
            lambda x: range(x[0], x[1] + 1), axis=1)
        df[self.N] = df[["Min", "Max", self.N]].apply(
            lambda x: x[2] / (x[1] - x[0] + 1), axis=1)
        df = df.explode(self.AGE).reset_index(drop=True)
        df[self.N] = df[self.N].astype(np.int64)
        return df.loc[:, self.PYRAMID_COLS]

    def retrieve(self, country):
        """
        Retrieve the dataset of the country from the local file or the server.

        Args:
            country (str): country name

        Returns:
            pandas.DataFrame: retrieved data
                Index
                    reset index
                Columns
                    - Country (pandas.Category): country name
                    - Year (int): year
                    - Sex (str): Female/Male
                    - Age (int): age
                    - Population (int): population value
        """
        if not self._cleaned_df.empty and country in self._cleaned_df[self.COUNTRY].unique():
            df = self._cleaned_df.copy()
            df = df.loc[df[self.COUNTRY] == country, :].reset_index(drop=True)
        else:
            # Retrieve from World Bank Open Data
            try:
                df = self._retrieve_from_server(country)
            except SubsetNotFoundError:
                raise SubsetNotFoundError(country=country) from None
            # Add to raw dataset
            self._cleaned_df = pd.concat([self._cleaned_df, df], ignore_index=True, axis=0)
            self._cleaned_df.to_csv(self._filename, index=False)
        # Data types
        cat_cols, int_cols = [self.COUNTRY, self.SEX], [self.AGE, self.N]
        df[cat_cols] = df[cat_cols].astype("category")
        df[int_cols] = df[int_cols].astype(np.int64)
        return df

    def cleaned(self):
        """
        Return the cleaned dataset.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Country (pandas.Category): country name
                    - Year (int): year
                    - Sex (str): Female/Male
                    - Age (int): age
                    - Population (int): population value
        """
        return self._cleaned_df

    def layer(self, country=None):
        raise NotImplementedError

    def subset(self, country, year=None, sex=None):
        """
        Return the subset.

        Args:
            country (str): country name
            year (int or None): year or None (the last records)
            sex (str): Female/Male or None (total)

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Age (int): age
                    - Population (int): population value
                    - Per_total (float): portion of the total
        """
        # Select by country
        df = self.retrieve(country=country)
        # Select by year
        year = year or df[self.YEAR].max()
        df = df.loc[df[self.YEAR] == year, :]
        # Select by sex
        if sex is not None:
            df = df.loc[df[self.SEX] == sex, :]
        df = df.drop([self.COUNTRY, self.YEAR, self.SEX], axis=1)
        # Calculate portion of the total
        df = pd.DataFrame(df.groupby(self.AGE).sum())
        df[self.PORTION] = df[self.N] / df[self.N].sum()
        return df.reset_index()

    def records(self, country, year=None, sex=None):
        """
        Return the subset.

        Args:
            country (str): country name
            year (int or None): year or None (the last records)
            sex (str): Female/Male or None (total)

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Age (int): age
                    - Population (int): population value
                    - Per_total (float): portion of the total
        """
        return self.subset(country=country, year=year, sex=sex)
