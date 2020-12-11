#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from dask import dataframe as dd
from covsirphy.util.plotting import line_plot
from covsirphy.util.error import SubsetNotFoundError
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.country_data import CountryData


class PCRData(CleaningBase):
    """
    Data cleaning of PCR dataset.

    Args:
        filename (str): CSV filename of the dataset
        citation (str): citation
    """
    # Column names
    TESTS = "Tests"
    TESTS_JPN = "Tested"
    
    PCR_COLS = [
        CleaningBase.DATE, CleaningBase.COUNTRY,
        CleaningBase.PROVINCE, TESTS, CleaningBase.C
    ]
    
    PCR_NLOC_COLUMNS = [CleaningBase.DATE, TESTS, CleaningBase.C]
    
    PCR_COLUMNS = [*CleaningBase.STR_COLUMNS, TESTS, CleaningBase.C]
    PCR_JPN_COLUMNS = [*CleaningBase.STR_COLUMNS, TESTS_JPN, CleaningBase.C]
    PCR_VALUE_COLUMNS = [TESTS, CleaningBase.C]
    
    def __init__(self, filename, interval=2, citation=None):
        if filename is None:
            self._raw = pd.DataFrame()
            self._cleaned_df = pd.DataFrame(columns=self.PCR_COLUMNS)
        else:
            self._raw = dd.read_csv(
                filename, dtype={"Province/State": "object"}
            ).compute()
            self._cleaned_df = self._cleaning()
        self.interval = self.ensure_natural_int(interval, name="interval")
        self._citation = citation or ""

    def cleaned(self, **kwargs):
        """
        Return the cleaned dataset of PCRData with tests and confirmed data.

        Note:
            Cleaning method is defined by self._cleaning() method.

        Args:
            kwargs: keword arguments will be ignored.

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
        """
        df = self._cleaned_df.copy()
        df = df.loc[:, self.PCR_COLS]
        return df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrites super()._cleaning() method.

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - ISO3 (str): ISO3 code
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
        """
        df = super()._cleaning()
        # Rename the columns
        df = df.rename(
            {
                "ObservationDate": self.DATE,
                "ISO3": self.ISO3,
                "Country/Region": self.COUNTRY,
                "Province/State": self.PROVINCE,
                "Tests": self.TESTS,
                "Confirmed": self.C
            },
            axis=1
        )
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.DATE, self.ISO3, self.COUNTRY, self.PROVINCE, self.TESTS, self.C
        ]
        self.ensure_dataframe(df, name="the raw data", columns=expected_cols)
        # Datetime columns
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Country
        df[self.COUNTRY] = df[self.COUNTRY].replace(
            {
                # COD
                "Congo, the Democratic Republic of the": "Democratic Republic of the Congo",
                # COG
                "Congo": "Republic of the Congo",
                # South Korea
                "Korea, South": "South Korea",
            }
        )
        # Province
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.UNKNOWN)
        df.loc[df[self.COUNTRY] == "Diamond Princess", [
            self.COUNTRY, self.PROVINCE]] = ["Others", "Diamond Princess"]
        # Values
        df = df.fillna(0)
        df[self.TESTS] = df[self.TESTS].astype(np.int64)
        df[self.C] = df[self.C].astype(np.int64)
        df = df.loc[:, [self.ISO3, *self.PCR_COLUMNS]].reset_index(drop=True)
        return df

    def replace(self, country_data):
        """
        Replace a part of cleaned dataset with a dataframe.

        Args:
            country_data (covsirphy.CountryData): dataset object of the country
                Index: reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Province (str): province name
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
                    - The other columns will be ignored

        Returns:
            covsirphy.PCRData: self
        """
        self.ensure_instance(country_data, CountryData, name="country_data")
        # Read new dataset
        country = country_data.country
        if country != "Japan":
            new = country_data.cleaned().loc[:, self.PCR_COLUMNS]
        else:
            new = country_data.cleaned().loc[:, self.PCR_JPN_COLUMNS]
            # Rename the columns
            new = new.rename(
                {
                    "Date": self.DATE,
                    "Country": self.COUNTRY,
                    "Province": self.PROVINCE,
                    "Tested": self.TESTS,
                    "Confirmed": self.C
                },
                axis=1
            )
            # Confirm the expected columns are in raw data
            expected_cols = self.PCR_COLUMNS
            self.ensure_dataframe(new, name="the raw data", columns=expected_cols)
            
        new[self.ISO3] = self.country_to_iso3(country)
        # Remove the data in the country from JHU dataset
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] != country]
        # Combine JHU data and the new data
        df = pd.concat([df, new], axis=0, sort=False)
        self._cleaned_df = df.copy()
        # Citation
        self._citation += f"\n{country_data.citation}"
        return self

    def _subset(self, country, province, start_date, end_date):
        """
        Return the subset of dataset or empty dataframe (when no records were found).

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
        """
        try:
            return super().subset(
                country=country, province=province, start_date=start_date, end_date=end_date)
        except SubsetNotFoundError:
            return pd.DataFrame(columns=self.PCR_NLOC_COLUMNS)

    def subset(self, country, province=None, start_date=None, end_date=None):
        """
        Return the subset of dataset

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
        """
        country_alias = self.ensure_country_name(country)
        # Subset with area, start/end date and calculate Susceptible
        subset_df = self._subset(
            country=country, province=province, start_date=start_date, end_date=end_date)
        if subset_df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date)
        return subset_df
    
    def pcr_monotonic(self, df, variable):
        """
        Force the variable show monotonic increasing.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed
            variable (str): variable name to show monotonic increasing

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed
        """
        # Whether complement is necessary or not
        if df[variable].is_monotonic_increasing:
            return df
        # Complement
        decreased_dates = df[df[variable].diff() < 0].index.tolist()
        for date in decreased_dates:
            # Raw value on the decreased date
            raw_last = df.loc[date, variable]
            # Extrapolated value on the date
            series = df.loc[:date, variable]
            series.iloc[-1] = None
            series.interpolate(method="spline", order=1, inplace=True)
            series.fillna(method="ffill", inplace=True)
            # Reduce values to the previous date
            df.loc[:date, variable] = series * raw_last / series.iloc[-1]
            df[variable] = df[variable].fillna(0).astype(np.int64)
        return df
    
    def pcr_check_complement(self, df, variable):
        """
        If variable values do not change for more than applied 'self.interval' days,
        indicate compliment action is needed.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed, Tests_diff, C_diff
            variable: the desired column to use

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed, Tests_diff, C_diff
            bool: True if complement is needed or False
        """
        df.fillna(0, inplace=True)
        max_frequency = df[variable].value_counts().max()
        need_complement = max_frequency > self.interval or not df.loc[df.index[-1], variable]
        return (df, need_complement)
    
    def pcr_partial_complement(self, df, variable, method=None):
        """
        If there are missing values in variable column,
        apply partial compliment (bfill, ffill) to all columns

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed, Tests_diff, C_diff
            variable: the desired column to use

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed, Tests_diff, C_diff
            bool: True if complement is needed or False
        """
        df, is_complemented = self.pcr_check_complement(df, variable)
        
        if not is_complemented:
            return (df, is_complemented)
        
        for col in df:
            df[col].replace(0,np.nan, inplace=True)
            df[col].fillna(method="ffill", inplace=True)
            df[col].fillna(method="bfill", inplace=True)
        return (df, is_complemented)

    def records(self, country, province=None, start_date=None, end_date=None, **kwargs):
        """
        JHU-style dataset for the area from the start date to the end date.

        Args:
            country(str): country name or ISO3 code
            province(str or None): province name
            start_date(str or None): start date, like 22Jan2020
            end_date(str or None): end date, like 01Feb2020

        Returns:
            tuple(pandas.DataFrame, bool):
                pandas.DataFrame:
                    Index: reset index
                    Columns:
                        - Date(pd.TimeStamp): Observation date
                        - Tests (int): the number of total tests performed
                        - Confirmed(int): the number of confirmed cases
                str/bool: kind of complement or False
        """
        country_alias = self.ensure_country_name(country)
        subset_arg_dict = {
            "country": country, "province": province,
            "start_date": start_date, "end_date": end_date,
        }
        try:
            return (self.subset(**subset_arg_dict), False)
        except ValueError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date) from None

    def positive_rate(self, country, province=None, window=3, show_figure=True, filename=None):
        """
        Return the PCR rate of a country as a dataframe.

        Args:
            country(str): country name or ISO3 code
            window (int): window of moving average, >= 1
            show_figure (bool): if True, show the records as a line-plot.
            filename (str): filename of the figure, or None (show figure)

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
                    - PCR_Rate (float): positive rate (%) of the daily cases over the total daily tests performed

        Notes:
            If non monotonic records were found for either cases or tests,
            "(complemented)" will be added to the title of the figure.
        """
        window = self.ensure_natural_int(window, name="window")
        df = self.records(country, province)[0]
        
        if not df[self.TESTS].max():
            print("No tests records found for country " + country) if not province else print("No tests records found for province " + province)
            df["PCR_Rate"] = 0
            return df

        # Check if there are too many missing value, more than half
        if (df[self.TESTS] == 0).mean() >= 0.5:
            print("Too many missing tests records for country " + country) if not province else print("Too many missing tests records for province " + province) 
            df["PCR_Rate"] = 0
            return df
        
        # Confirmed must be monotonically increasing
        if df[self.C].iloc[-1] == 0:
            df[self.C].iloc[-1] = df[self.C].iloc[-2]
        df = self.pcr_monotonic(df, self.C)
        df, is_complemented = self.pcr_partial_complement(df, self.TESTS)
        # If Tests values are all valid, with no missing values in-between,
        # they must be monotonically increasing as well
        df = self.pcr_monotonic(df, self.TESTS)
        
        # Calculate daily values for tests and confirmed (with window=1)
        df["Tests_diff"] = df[self.TESTS].diff()
        df["C_diff"] = df[self.C].diff()
        
        # Ensure that tests > confirmed in daily basis
        df.loc[df["Tests_diff"].abs() < df["C_diff"].abs(), "Tests_diff"] = None
        
        # Keep valid non-zero values by ignoring zeros at the beginning
        df = df.replace(0,np.nan)
        non_zero_index_start = df["Tests_diff"].first_valid_index()
        df = df.loc[non_zero_index_start:].reset_index(drop=True)
        non_zero_index_end = df["Tests_diff"].last_valid_index()
        
        # Keep valid non-zero values by complementing zeros at the end
        if non_zero_index_end < (len(df)-1):
            df.loc[non_zero_index_end+1:, "Tests_diff"] = None
        df, is_complemented = self.pcr_partial_complement(df, "Tests_diff")
        
        # Use rolling window for averaging tests and confirmed
        df["Tests_diff"] = df["Tests_diff"].rolling(window).mean()
        df["C_diff"] = df["C_diff"].rolling(window).mean()
        df, is_complemented = self.pcr_partial_complement(df, "Tests_diff")
        
        # Remove first zero lines due to window
        df = df.replace(0,np.nan)
        non_zero_index_start = df["Tests_diff"].first_valid_index()
        df = df.loc[non_zero_index_start:].reset_index(drop=True)
        
        # Calculate PCR values
        df["PCR_Rate"] = [(i / j) * 100 for i, j in zip(df["C_diff"], df["Tests_diff"])]
        
        if not show_figure:
            return df
        title = f"{country if not province else province}: Positive Rate (%) over time{' (Tests partially complemented)' if is_complemented else ''}"
        line_plot(
            df.set_index(self.DATE)["PCR_Rate"],
            title,
            ylabel="PCR_Rate (%)",
            y_integer=True,
            filename=filename
        )
        return df
