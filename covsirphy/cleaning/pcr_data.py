#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from dask import dataframe as dd
import swifter
from covsirphy.util.plotting import line_plot
from covsirphy.util.error import PCRIncorrectPreconditionError
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.country_data import CountryData


class PCRData(CleaningBase):
    """
    Data cleaning of PCR dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        interval (int): expected update interval of the number of confirmed cases and tests [days]
        citation (str): citation
    """
    # Column names
    PCR_VALUE_COLUMNS = [CleaningBase.TESTS, CleaningBase.C]
    PCR_NLOC_COLUMNS = [CleaningBase.DATE, *PCR_VALUE_COLUMNS]
    PCR_COLUMNS = [*CleaningBase.STR_COLUMNS, *PCR_VALUE_COLUMNS]
    # Daily values
    T_DIFF = "Tests_diff"
    C_DIFF = "Confirmed_diff"
    PCR_RATE = "Test_positive_rate"

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
        # To avoid "imported but unused"
        self.__swifter = swifter

    def cleaned(self):
        """
        Return the cleaned dataset of PCRData with tests and confirmed data.

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

        Notes:
            Cleaning method is defined by self._cleaning() method.
        """
        return self._cleaned_df.loc[:, self.PCR_COLUMNS]

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
                "Country/Region": self.COUNTRY,
                "Province/State": self.PROVINCE,
            },
            axis=1
        )
        # Confirm the expected columns are in raw data
        self.ensure_dataframe(
            df, name="the raw data", columns=self.PCR_COLUMNS)
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
        df = df.fillna(method="ffill").fillna(0)
        df[self.TESTS] = df[self.TESTS].astype(np.int64)
        df[self.C] = df[self.C].astype(np.int64)
        df = df.loc[:, [self.ISO3, *self.PCR_COLUMNS]].reset_index(drop=True)
        return df

    @classmethod
    def from_dataframe(cls, dataframe):
        """
        Create PCRData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): cleaned dataset

        Returns:
            covsirphy.PCRData: PCR dataset
        """
        instance = cls(filename=None)
        instance._cleaned_df = cls.ensure_dataframe(
            dataframe, name="dataframe", columns=cls.PCR_COLUMNS)
        return instance

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
        new = self.ensure_dataframe(
            country_data.cleaned(), name="the raw data", columns=self.PCR_COLUMNS)
        new = new.loc[:, self.PCR_COLUMNS]
        new[self.ISO3] = self.country_to_iso3(country)
        # Remove the data in the country from the current datset
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] != country]
        # Add the new data
        self._cleaned_df = pd.concat([df, new], axis=0, sort=False)
        # Citation
        self._citation += f"\n{country_data.citation}"
        return self

    @staticmethod
    def _pcr_monotonic(df, variable):
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

    def _pcr_check_complement(self, df, variable):
        """
        If variable values do not change for more than applied 'self.interval' days,
        indicate compliment action is needed.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed, Tests_diff, C_diff
            variable: the desired column to use

        Returns:
            bool: whether complement is necessary or not
        """
        max_frequency = df[variable].value_counts().max()
        return max_frequency > self.interval or not df.loc[df.index[-1], variable]

    def _pcr_partial_complement(self, before_df, variable):
        """
        If there are missing values in variable column,
        apply partial compliment (bfill, ffill) to all columns.

        Args:
            before_df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed, Tests_diff, C_diff
            variable: the desired column to use

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed, Tests_diff, C_diff

        Notes:
            Filling NA with 0 will be always applied.
        """
        df = before_df.fillna(0)
        if not self._pcr_check_complement(df, variable):
            return df
        for col in df:
            df[col].replace(0, np.nan, inplace=True)
            df[col].fillna(method="ffill", inplace=True)
            df[col].fillna(method="bfill", inplace=True)
        return df

    def _pcr_processing(self, before_df, window):
        """
        Return the processed pcr data

        Args:
            before_df (pandas.DataFrame):
                Index: reset index
                Columns: Date (pandas.TimeStamp), Tests, Confirmed
            window (int): window of moving average, >= 1

        Returns:
            tuple(pandas.DataFrame, bool):
                pandas.DataFrame
                    Index: reset index
                    Columns:
                        - Date (pd.TimeStamp): Observation date
                        - Tests (int): the number of total tests performed
                        - Confirmed (int): the number of confirmed cases
                        - Tests_diff (int): daily tests performed
                        - Confirmed_diff (int): daily confirmed cases
                bool: True if complement is needed or False
        """
        df = before_df.fillna(method="ffill")
        # Confirmed must show monotonic increasing
        df = self._pcr_monotonic(df, self.C)
        df = self._pcr_partial_complement(df, self.TESTS)
        # If Tests values are all valid, with no missing values in-between,
        # they must be monotonically increasing as well
        df = self._pcr_monotonic(df, self.TESTS)
        # Calculate daily values for tests and confirmed (with window=1)
        df[self.T_DIFF] = df[self.TESTS].diff()
        df[self.C_DIFF] = df[self.C].diff()
        # Ensure that tests > confirmed in daily basis
        df.loc[
            df[self.T_DIFF].abs() < df[self.C_DIFF].abs(), self.T_DIFF] = None
        # Keep valid non-zero values by ignoring zeros at the beginning
        df = df.replace(0, np.nan)
        non_zero_index_start = df[self.T_DIFF].first_valid_index()
        df = df.loc[non_zero_index_start:].reset_index(drop=True)
        non_zero_index_end = df[self.T_DIFF].last_valid_index()
        # Keep valid non-zero values by complementing zeros at the end
        if non_zero_index_end < (len(df) - 1):
            df.loc[non_zero_index_end + 1:, self.T_DIFF] = None
        df = self._pcr_partial_complement(df, self.T_DIFF)
        # Use rolling window for averaging tests and confirmed
        df[self.T_DIFF] = df[self.T_DIFF].rolling(window).mean()
        df[self.C_DIFF] = df[self.C_DIFF].rolling(window).mean()
        df = self._pcr_partial_complement(df, self.T_DIFF)
        # Remove first zero lines due to window
        df = df.replace(0, np.nan)
        non_zero_index_start = df[self.T_DIFF].first_valid_index()
        df = df.loc[non_zero_index_start:].reset_index(drop=True)
        # Complemented or not
        is_complemented = df.drop(
            [self.T_DIFF, self.C_DIFF], axis=1).equals(before_df)
        return (df, is_complemented)

    def _pcr_check_preconditions(self, df, country, province):
        """
        Check preconditions in order to proceed with PCR data processing.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Tests, Confirmed
            country(str): country name or ISO3 code
            province(str or None): province name

        Raises:
            PCRIncorrectPreconditionError: the dataset has too many missing values
        """
        if (not df[self.TESTS].max()) or ((df[self.TESTS] == 0).mean() >= 0.5):
            raise PCRIncorrectPreconditionError(
                country=country, province=province, message="Too many missing Tests records") from None

    def positive_rate(self, country, province=None, window=3, show_figure=True, filename=None):
        """
        Return the PCR rate of a country as a dataframe.

        Args:
            country(str): country name or ISO3 code
            province(str or None): province name
            window (int): window of moving average, >= 1
            show_figure (bool): if True, show the records as a line-plot.
            filename (str): filename of the figure, or None (display figure)

        Raises:
            PCRIncorrectPreconditionError: the dataset has too many missing values

        Returns:
            pandas.DataFrame
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
                    - Tests_diff (int): daily tests performed
                    - Confirmed_diff (int): daily confirmed cases
                    - Test_positive_rate (float): positive rate (%) of the daily cases over the total daily tests performed

        Notes:
            If non monotonic records were found for either confirmed cases or tests,
            "with partially complemented tests data" will be added to the title of the figure.
        """
        window = self.ensure_natural_int(window, name="window")
        subset_df = self.subset(country, province=province)
        # Check PCR data preconditions
        self._pcr_check_preconditions(subset_df, country, province)
        # Process PCR data
        df, is_complemented = self._pcr_processing(subset_df, window)
        # Calculate PCR values
        df[self.PCR_RATE] = df[[self.C_DIFF, self.T_DIFF]].swifter.progress_bar(False).apply(
            lambda x: x[0] / x[1] * 100, axis=1)
        if not show_figure:
            return df
        # Create figure
        area = self.area_name(country, province=province)
        comp_status = "\nwith partially complemented tests data" if is_complemented else ""
        line_plot(
            df.set_index(self.DATE)[self.PCR_RATE],
            title=f"{area}: Test positive rate (%) over time {comp_status}",
            ylabel="Test positive rate (%)",
            y_integer=True,
            filename=filename,
            show_legend=False,
        )
        return df
