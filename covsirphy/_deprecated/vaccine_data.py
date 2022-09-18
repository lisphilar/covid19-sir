#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy._deprecated.cbase import CleaningBase


class VaccineData(CleaningBase):
    """
    Data cleaning of vaccination dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        data (pandas.DataFrame or None):
            Index
                reset index
            Columns
                - Date: Observation date
                - Country: country/region name
                - ISO3: ISO 3166-1 alpha-3, like JPN
                - Province: province/prefecture/state name
                - Product: vaccine product names
                - Vaccinations: cumulative number of vaccinations
                - Vaccinations: cumulative number of booster vaccinations
                - Vaccinated_once: cumulative number of people who received at least one vaccine dose
                - Vaccinated_full: cumulative number of people who received all doses prescribed by the protocol
        citation (str or None): citation or None (empty)
        kwargs: the other arguments will be ignored

    Note:
        Either @filename (high priority) or @data must be specified.

    Note:
        Columns of VaccineData.cleaned():
            - Date (pandas.TimeStamp): observation dates
            - Country (pandas.Category): country (or province) names
            - ISO3 (pandas.Category): ISO3 codes
            - Province (pandas.Category): province/prefecture/state name
            - Product (pandas.Category): vaccine product names
            - Vaccinations (int): cumulative number of vaccinations
            - Vaccinations (int): cumulative number of booster vaccinations
            - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
            - Vaccinated_full (int): cumulative number of people who received all doses prescribed by the protocol
    """

    @deprecate(old="VaccineData", new="DataEngineer", version="2.24.0-xi")
    def __init__(self, filename=None, data=None, citation=None, **kwargs):
        self._subset_cols = [self.DATE, self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL]
        self._raw_cols = [
            self.DATE, self.ISO3, self.COUNTRY, self.PROVINCE, self.PRODUCT,
            self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL]
        # Raw data
        self._raw = self._parse_raw(filename, data, self._raw_cols)
        # Backward compatibility
        if self._raw.empty:
            self._raw = self._retrieve(filename, **kwargs)
        if self.PROVINCE not in self._raw:
            self._raw[self.PROVINCE] = self.NA
        # Data cleaning
        self._cleaned_df = pd.DataFrame(columns=self._raw_cols) if self._raw.empty else self._cleaning()
        # Citation
        self._citation = citation or ""
        # Directory that save the file
        if filename is None:
            self._dirpath = Path("input")
        else:
            Path(filename).parent.mkdir(exist_ok=True, parents=True)
            self._dirpath = Path(filename).resolve().parent

    @deprecate(
        "vaccine_data = cs.VaccineData()",
        new="vaccine_data = cs.DataLoader().vaccine()", version="2.21.0-iota-fu1")
    def _retrieve(self, filename, verbose=1, **kwargs):
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
        URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/"
        URL_REC = f"{URL}vaccinations.csv"
        URL_LOC = f"{URL}locations.csv"
        # Show URL
        if verbose:
            print("Retrieving COVID-19 vaccination dataset from https://github.com/owid/covid-19-data/")
        # Download datasets and merge them
        rename_dict = {
            "date": self.DATE, "location": self.COUNTRY, "iso_code": self.ISO3,
            "vaccines": self.PRODUCT, "total_vaccinations": self.VAC, "total_boosters": self.VAC_BOOSTERS,
            "people_vaccinated": self.V_ONCE,
            "people_fully_vaccinated": self.V_FULL,
        }
        rec_df = pd.read_csv(URL_REC, usecols=list(set(rename_dict) - {"vaccines"}))
        loc_df = pd.read_csv(URL_LOC, usecols=["location", "vaccines"])
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
                - Province: province/prefecture/state name
                - Product (pandas.Category): vaccine product names
                - Vaccinations (int): cumulative number of vaccinations
                - Vaccinations (int): cumulative number of booster vaccinations
                - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                - Vaccinated_full (int): cumulative number of people who received all doses prescribed by the protocol
        """
        df = self._raw.copy()
        # Date
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Fill in NA values
        for col in [self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        country_df = df.loc[:, [self.COUNTRY, self.ISO3, self.PRODUCT]].drop_duplicates()
        # Extent dates to today
        today_date = datetime.now().replace(
            hour=00, minute=00, second=00, microsecond=00
        )

        df = df.pivot_table(
            values=[self.VAC, self.VAC_BOOSTERS, self.V_ONCE, self.V_FULL],
            index=self.DATE, columns=[self.COUNTRY, self.PROVINCE], aggfunc="last")
        df = df.reindex(pd.date_range(df.index[0], today_date, freq="D"))
        df.index.name = self.DATE
        df = df.ffill().fillna(0).astype(np.int64).stack().stack().reset_index()
        df.sort_values(by=[self.COUNTRY, self.PROVINCE, self.DATE], ignore_index=True, inplace=True)
        df = df.merge(country_df, on=self.COUNTRY)
        # Set dtype for category data
        for col in [self.COUNTRY, self.ISO3, self.PROVINCE, self.PRODUCT]:
            df[col] = df[col].astype("category")
        return df.loc[:, self._raw_cols]

    def subset(self, country, province=None, product=None, start_date=None, end_date=None):
        """
        Return subset of the country/province and start/end date.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            product (str or None): vaccine product name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - Vaccinations (int): the number of vaccinations
                    - Vaccinations_boosters (int): the number of booster vaccinations
                    - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (int): cumulative number of people who received all doses prescribed by the protocol
        """
        df = self._cleaned_df.copy()
        # Subset by country
        iso3 = self.ensure_country_name(country)
        df = df.loc[(df[self.ISO3] == iso3) & (df[self.PROVINCE] == (province or self.NA))]
        # Subset by product name
        if product is not None:
            df = df.loc[df[self.PRODUCT] == product]
        # Subset with start date
        if start_date is not None:
            df = df.loc[df[self.DATE] >= self._ensure_date(start_date)]
        # Subset with end date
        if end_date is not None:
            df = df.loc[df[self.DATE] <= self._ensure_date(end_date)]
        # Check records were found
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=iso3, province=product, start_date=start_date, end_date=end_date)
        return df.loc[:, self._subset_cols].reset_index(drop=True)

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
                    - Vaccinations (int): the number of booster vaccinations
                    - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (int): cumulative number of people who received all doses prescribed by the protocol
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
                    - Vaccinations_boosters (int): the number of booster vaccinations
                    - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
                    - Vaccinated_full (int): cumulative number of people who received all doses prescribed by the protocol
        """
        df = self._cleaned_df.copy()
        # Select 'World' data
        df = df.loc[df[self.COUNTRY] == "World"]
        # Resampling
        df = df.set_index(self.DATE).resample("D").sum()
        return df.reset_index()[self._subset_cols]

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
        title = f'Global: the number of {variable.lower()} on {date_str}'
        # Global map
        return self._colored_map_global(variable=variable, title=title, date=date, **kwargs)
