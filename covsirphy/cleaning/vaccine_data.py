#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import datetime
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy.cleaning.cbase import CleaningBase


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
                - Product: vaccine product names
                - Vaccinations: cumulative number of vaccinations
                - Vaccinated_once: cumulative number of people who received at least one vaccine dose
                - Vaccinated_full: cumulative number of people who received all doses prescrived by the protocol
        citation (str or None): citation or None (empty)
        kwargs: the other arguments will be ignored

    Note:
        Either @filename (high priority) or @data must be specified.

    Note:
        Columns of VaccineData.cleaned():
            - Date (pandas.TimeStamp): observation dates
            - Country (pandas.Category): country (or province) names
            - ISO3 (pandas.Category): ISO3 codes
            - Product (pandas.Category): vaccine product names
            - Vaccinations (int): cumulative number of vaccinations
            - Vaccinated_once (int): cumulative number of people who received at least one vaccine dose
            - Vaccinated_full (int): cumulative number of people who received all doses prescrived by the protocol
    """
    # Columns of self._raw and self._clean_df
    RAW_COLS = [
        CleaningBase.DATE, CleaningBase.COUNTRY, CleaningBase.ISO3, CleaningBase.PRODUCT,
        CleaningBase.VAC, CleaningBase.V_ONCE, CleaningBase.V_FULL]
    # Columns of self.cleaned()
    CLEANED_COLS = RAW_COLS[:]
    # Columns of self.subset()
    SUBSET_COLS = [CleaningBase.DATE, CleaningBase.VAC, CleaningBase.V_ONCE, CleaningBase.V_FULL]

    def __init__(self, filename=None, data=None, citation=None, **kwargs):
        # Raw data
        if data is not None and self.PROVINCE in data:
            data_c = data.loc[data[self.PROVINCE] == self.UNKNOWN]
            self._raw = self._parse_raw(filename, data_c, self.RAW_COLS)
        else:
            self._raw = self._parse_raw(filename, data, self.RAW_COLS)
        # Backward compatibility
        if self._raw.empty:
            self._raw = self._retrieve(filename, **kwargs)
        # Data cleaning
        self._cleaned_df = pd.DataFrame(columns=self.RAW_COLS) if self._raw.empty else self._cleaning()
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
            "vaccines": self.PRODUCT, "total_vaccinations": self.VAC,
            "people_vaccinated": self.V_ONCE,
            "people_fully_vaccinated": self.V_FULL,
        }
        rec_df = self.load(URL_REC, columns=list(set(rename_dict) - set(["vaccines"])))
        loc_df = self.load(URL_LOC, columns=["location", "vaccines"])
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
                - Product (pandas.Category): vaccine product names
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
        today_date = datetime.datetime.today().replace(hour=00, minute=00, second=00, microsecond=00)
        for country in df.Country.unique():
            subset_df = df.loc[df[self.COUNTRY] == country]
            # Add any missing dates up until today
            if subset_df[self.DATE].max() < today_date:
                new_dates = pd.date_range(
                    subset_df[self.DATE].max() + datetime.timedelta(days=1), today_date)
                subset_df = subset_df.reset_index(drop=True)
                keep_index = subset_df[self.VAC].idxmax() + 1
                new_df = pd.DataFrame(index=new_dates, columns=subset_df.columns)
                new_df.index.name = self.DATE
                new_df = new_df.drop(self.DATE, axis=1).reset_index()
                subset_df = pd.concat([subset_df, new_df], axis=0, ignore_index=True).ffill()
                subset_df = subset_df.loc[keep_index:]
                df = pd.concat([df, subset_df], axis=0, ignore_index=True)
        df.sort_values(by=[self.COUNTRY, self.DATE], ignore_index=True, inplace=True)
        return df.loc[:, self.RAW_COLS]

    def subset(self, country, product=None, start_date=None, end_date=None):
        """
        Return subset of the country/province and start/end date.

        Args:
            country (str or None): country name or ISO3 code
            product (str or None): vaccine product name
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
            df = df.loc[df[self.DATE] >= self._ensure_date(start_date)]
        # Subset with end date
        if end_date is not None:
            df = df.loc[df[self.DATE] <= self._ensure_date(end_date)]
        # Check records were found
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=product,
                start_date=start_date, end_date=end_date)
        return df.loc[:, self.SUBSET_COLS].reset_index(drop=True)

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
