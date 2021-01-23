#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from covsirphy.util.error import SubsetNotFoundError
from covsirphy.cleaning.cbase import CleaningBase


class VaccineData(CleaningBase):
    """
    Dataset regarding vaccination retrieved from "Our World In Data".
    https://github.com/owid/covid-19-data/tree/master/public/data
    https://ourworldindata.org/coronavirus

    Args:
        filename (str or pathlib.path): CSV filename to save the raw dataset
        force (bool): if True, always download the dataset from the server
        verbose (int): level of verbosity
    Note:
        Columns of VaccineData.cleaned():
            - Date (pandas.TimeStamp): observation dates
            - Country (pandas.Category): country (or province) names
            - Product (pandas.Category): product names
            - Vaccinations (int): the number of vaccinations
    """
    # URL
    URL = "https://covid.ourworldindata.org/data/vaccinations/"
    URL_REC = f"{URL}vaccinations.csv"
    URL_LOC = f"{URL}locations.csv"
    # Columns
    VAC_COLS = [
        CleaningBase.DATE, CleaningBase.COUNTRY, CleaningBase.PRODUCT, CleaningBase.VAC]

    def __init__(self, filename, force=False, verbose=1):
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        if Path(filename).exists() and not force:
            self._raw = self.load(filename)
        else:
            self._raw = self._retrieve(filename=filename, verbose=verbose)
        self._cleaned_df = self._cleaning()
        self._citation = "Hasell, J., Mathieu, E., Beltekian, D. et al." \
            " A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020)." \
            " https://doi.org/10.1038/s41597-020-00688-8"

    def _retrieve(self, filename, verbose=1):
        """
        Retrieve the dataset from server.
        Args:
            filename (str or pathlib.path): CSV filename to save the raw dataset
            verbose (int): level of verbosity
        Returns:
            pd.DataFrame:
                Index reset index
                Columns Date, Country, Product, Vaccinations
        """
        # Show URL
        if verbose:
            print(
                "Retrieving COVID-19 vaccination dataset from https://covid.ourworldindata.org/data/")
        # Download datasets and merge them
        rec_df = self.load(
            self.URL_REC, columns=["location", "date", "total_vaccinations"])
        loc_df = self.load(self.URL_LOC, columns=["location", "vaccines"])
        df = rec_df.merge(loc_df, how="left", on="location")
        df = df.rename(
            {
                "vaccines": self.PRODUCT, "total_vaccinations": self.VAC,
                "date": self.DATE, "location": self.COUNTRY}, axis=1)
        # Remove "World" records (total values)
        df = df.loc[df[self.COUNTRY] != "World"]
        # Save the dataframe as CSV file
        df.to_csv(filename, index=False)
        return df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            pandas.DataFrame:
                Index reset index
                Columns Date, Country, Product, Vaccinations
        """
        df = self._raw.copy()
        # Date
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Location (Country)
        df[self.COUNTRY] = df[self.COUNTRY].astype("category")
        # Company
        df[self.PRODUCT] = df[self.PRODUCT].astype("category")
        # Vaccinations
        df[self.VAC] = pd.to_numeric(df[self.VAC], errors="coerce")
        df[self.VAC] = df[self.VAC].fillna(method="ffill").fillna(0)
        df[self.VAC] = df[self.VAC].astype(np.int64)
        return df.loc[:, self.VAC_COLS]

    def subset(self, country, product=None, start_date=None, end_date=None):
        """
        Return subset of the country/province and start/end date.

        Args:
            country (str or None): country name or ISO3 code
            product (str or None): product name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.TimeStamp): observation date
                    - Vaccinations (int): the number of vaccinations
        """
        df = self._cleaned_df.copy()
        # Subset by country
        country_alias = self.ensure_country_name(country)
        df = df.loc[df[self.COUNTRY] == country_alias]
        # Subset by product name
        if product is not None:
            df = df.loc[df[self.PRODUCT] == product]
        # Subset with start date
        if start_date is not None:
            df = df.loc[df[self.DATE] >= self.date_obj(start_date)]
        # Subset with end date
        if end_date is not None:
            df = df.loc[df[self.DATE] <= self.date_obj(end_date)]
        # Resampling
        df = df.set_index(self.DATE).resample("D").sum().reset_index()
        # Fill in the blanks
        df[self.VAC] = df[self.VAC].replace(0, None)
        df[self.VAC] = df[self.VAC].fillna(method="ffill").fillna(0)
        # Check records were found
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=product,
                start_date=start_date, end_date=end_date)
        return df

    def records(self, country, product=None, start_date=None, end_date=None):
        """
        Return subset of the country/province and start/end date.

        Args:
            country (str or None): country name or ISO3 code
            product (str or None): product name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.TimeStamp): observation date
                    - Vaccinations (int): the number of vaccinations
        """
        return self.subset(
            country=country, product=product, start_date=start_date, end_date=end_date)

    def total(self):
        """
        Calculate total values of the cleaned dataset.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.TimeStamp): observation date
                    - Vaccinations (int): the number of vaccinations
        """
        df = self._cleaned_df.copy()
        # Remove duplication
        df = df.loc[df[self.COUNTRY] == "United Kingdom"]
        # Resampling
        df = df.set_index(self.DATE).resample("D").sum()
        return df.reset_index()
