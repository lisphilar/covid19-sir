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
            - ISO3 (pandas.Category): ISO3 codes
            - Product (pandas.Category): product names
            - Vaccinations (int): cumulative number of vaccinations
            - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
            - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
    """
    # URL
    URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/"
    URL_REC = f"{URL}vaccinations.csv"
    URL_LOC = f"{URL}locations.csv"
    # Columns
    VAC_COLS = [
        CleaningBase.DATE, CleaningBase.COUNTRY, CleaningBase.ISO3, CleaningBase.PRODUCT,
        CleaningBase.VAC, CleaningBase.V_ONCE, CleaningBase.V_FULL]
    VAC_SUBSET_COLS = [CleaningBase.DATE, CleaningBase.VAC, CleaningBase.V_ONCE, CleaningBase.V_FULL]

    def __init__(self, filename, force=False, verbose=1):
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        if Path(filename).exists() and not force:
            try:
                self._raw = self.load(filename)
            except KeyError:
                # Error when the local dataset does not have necessary columns
                # Raised when new CovsirPhy version requires additional columns
                self._raw = self._retrieve(filename=filename, verbose=verbose)
        else:
            self._raw = self._retrieve(filename=filename, verbose=verbose)
        self._cleaned_df = self._cleaning()
        self._citation = "Hasell, J., Mathieu, E., Beltekian, D. et al." \
            " A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020)." \
            " https://doi.org/10.1038/s41597-020-00688-8"
        # Directory that save the file
        self._dirpath = Path(filename or "input").resolve().parent

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
            print("Retrieving COVID-19 vaccination dataset from https://github.com/owid/covid-19-data/")
        # Download datasets and merge them
        rename_dict = {
            "date": self.DATE, "location": self.COUNTRY, "iso_code": self.ISO3,
            "vaccines": self.PRODUCT, "total_vaccinations": self.VAC,
            "people_vaccinated": self.V_ONCE,
            "people_fully_vaccinated": self.V_FULL,
        }
        rec_df = self.load(self.URL_REC, columns=list(set(rename_dict) - set(["vaccines"])))
        loc_df = self.load(self.URL_LOC, columns=["location", "vaccines"])
        df = rec_df.merge(loc_df, how="left", on="location")
        df = df.rename(rename_dict, axis=1)
        # Save the dataframe as CSV file
        df.to_csv(filename, index=False)
        return df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                - Date (pandas.TimeStamp): observation dates
                - Country (pandas.Category): country (or province) names
                - ISO3 (pandas.Category): ISO3 codes
                - Product (pandas.Category): product names
                - Vaccinations (int): cumulative number of vaccinations
                - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
        """
        df = self._raw.copy()
        # Date
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        for col in [self.COUNTRY, self.ISO3, self.PRODUCT]:
            df[col] = df[col].astype("category")
        # Fill in NA values
        for col in [self.VAC, self.V_ONCE, self.V_FULL]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df.groupby(self.ISO3)[col].fillna(method="ffill").fillna(0).astype(np.int64)
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
                    - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
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
        # Check records were found
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=product,
                start_date=start_date, end_date=end_date)
        return df.loc[:, self.VAC_SUBSET_COLS].reset_index(drop=True)

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
                    - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
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
                    - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
        """
        df = self._cleaned_df.copy()
        # Select 'World' data
        df = df.loc[df[self.COUNTRY] == "World"]
        # Resampling
        df = df.set_index(self.DATE).resample("D").sum()
        return df.reset_index()

    def map(self, country=None, variable="Vaccinations", date=None, **kwargs):
        """
        Create colored map with the number of vaccinations.

        Args:
            country (None): always None
            variable (str): variable to show
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()

        Raises:
            NotImplementedError: @country was specified
        """
        if country is not None:
            raise NotImplementedError("@country cannot be specified, always None.")
        # Date
        date_str = date or self.cleaned()[self.DATE].max().strftime(self.DATE_FORMAT)
        country_str = "Global"
        title = f"{country_str}: the number of {variable.lower()} on {date_str}"
        # Global map
        return self._colored_map_global(variable=variable, title=title, date=date, **kwargs)
