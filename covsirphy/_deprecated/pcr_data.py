#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.util.error import PCRIncorrectPreconditionError, SubsetNotFoundError, deprecate
from covsirphy.visualization.line_plot import line_plot
from covsirphy._deprecated.cbase import CleaningBase
from covsirphy._deprecated.country_data import CountryData


class PCRData(CleaningBase):
    """
    Data cleaning of PCR dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        data (pandas.DataFrame or None):
            Index
                reset index
            Columns
                - Date: Observation date
                - ISO3: ISO3 code
                - Country: country/region name
                - Province: province/prefecture/state name
                - Tests: the number of tests
        interval (int): expected update interval of the number of confirmed cases and tests [days]
        min_pcr_tests (int): minimum number of valid daily tests performed in order to calculate positive rate
        citation (str): citation
    """
    # Daily values
    C_DIFF = "Confirmed_diff"
    PCR_RATE = "Test_positive_rate"

    @deprecate(old="PCRData", new="DataEngineer", version="2.24.0-xi")
    def __init__(self, filename=None, data=None, interval=2, min_pcr_tests=100, citation=None):
        variables = [self.TESTS, self.C]
        super().__init__(filename=filename, data=data, citation=citation, variables=variables)
        # Settings
        self.interval = self._ensure_natural_int(interval, name="interval")
        self.min_pcr_tests = self._ensure_natural_int(min_pcr_tests, name="min_pcr_tests")

    def cleaned(self):
        """
        Return the cleaned dataset of PCRData with tests and confirmed data.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
        """
        df = self._cleaned_df.loc[:, self._raw_cols]
        return df.drop(self.ISO3, axis=1)

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrites super()._cleaning() method.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - ISO3 (str): ISO3 code
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
        """
        df = self._raw.copy()
        df = df.loc[:, self._raw_cols].reset_index(drop=True)
        # Datetime columns
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Province
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.NA)
        # Values
        df = df.dropna(subset=[self.TESTS, self.C], how="any")
        for col in [self.TESTS, self.C]:
            df[col] = df.groupby([self.COUNTRY, self.PROVINCE])[col].ffill().fillna(0).astype(np.int64)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        return df

    @classmethod
    @deprecate("PCRData.from_dataframe()", new="PCRData(data)", version="2.21.0-iota")
    def from_dataframe(cls, dataframe, directory="input"):
        """
        Create PCRData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): cleaned dataset
                Index
                    reset index
                Columns
                    - Date: Observation date
                    - ISO3: ISO3 code (optional)
                    - Country: country/region name
                    - Province: province/prefecture/state name
                    - Tests: the number of tests
            directory (str): directory to save geography information (for .map() method)

        Returns:
            covsirphy.PCRData: PCR dataset
        """
        df = cls._ensure_dataframe(dataframe, name="dataframe")
        df[cls.ISO3] = df[cls.ISO3] if cls.ISO3 in df.columns else cls.NA
        instance = cls(filename=None)
        instance.directory = str(directory)
        instance._cleaned_df = cls._ensure_dataframe(df, name="dataframe", columns=cls._raw_cols)
        return instance

    @deprecate("PCRData.use_ourworldindata()", new="DataLoader.pcr()", version="2.21.0-iota-fu4")
    def use_ourworldindata(self, filename, force=False):
        """
        Deprecated and removed.
        Set the cleaned dataset retrieved from "Our World In Data" site.
        https://github.com/owid/covid-19-data/tree/master/public/data
        https://ourworldindata.org/coronavirus

        Args:
            filename (str): CSV filename to save the datasetretrieved from "Our World In Data"
            force (bool): if True, always download the dataset from "Our World In Data"
        """
        raise NotImplementedError

    @deprecate("PCRData.replace()", new="DataLoader.read_dataframe()", version="sigma",
               ref="https://lisphilar.github.io/covid19-sir/markdown/LOADING.html")
    def replace(self, country_data):
        """
        Replace a part of cleaned dataset with a dataframe.

        Args:
            country_data (covsirphy.CountryData): dataset object of the country

                Index
                    reset index
                Columns
                    - Date (pandas.TimeStamp): Observation date
                    - Province (pandas.Category): province name
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
                    - The other columns will be ignored

        Returns:
            covsirphy.PCRData: self
        """
        self._ensure_instance(country_data, CountryData, name="country_data")
        # Read new dataset
        country = country_data.country
        new = country_data.cleaned()
        new[self.ISO3] = self.country_to_iso3(country)
        self._ensure_dataframe(new, name="the raw data", columns=self._raw_cols)
        new = new.loc[:, self._raw_cols]
        # Remove the data in the country from the current datset
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] != country]
        # Add the new data
        df = pd.concat([df, new], axis=0, sort=False)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        self._cleaned_df = df.copy()
        # Citation
        self._citation += f"\n{country_data.citation}"
        return self

    @staticmethod
    def _pcr_monotonic(df, variable):
        """
        Force the variable show monotonic increasing.

        Args:
            df (pandas.DataFrame):
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Tests, Confirmed
            variable (str): variable name to show monotonic increasing

        Returns:
            pandas.DataFrame: complemented records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Tests, Confirmed
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
            series.interpolate(method="linear", inplace=True, limit_direction="both")
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
                Index
                    Sate (pandas.TimeStamp)
                Columns
                    Tests, Confirmed, Tests_diff, C_diff
            variable: the desired column to use

        Returns:
            bool: whether complement is necessary or not
        """
        max_frequency = df[variable].value_counts().max()
        return max_frequency > self.interval or not df.loc[df.index[-1], variable]

    def _pcr_partial_complement_ending(self, df, window):
        """
        If ending test values do not change daily, while there are new cases,
        apply previous diff() only to these ending unupdated values
        and keep the previous valid ones.

        Args:
            df (pandas.DataFrame):
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Tests, Confirmed
            window (int): window of moving average, >= 1

        Returns:
            pandas.DataFrame: complemented test records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Tests, Confirmed
        """
        # Whether complement is necessary or not
        tests_max = df[self.TESTS].max()
        check_tests_ending = (
            df[self.TESTS] == tests_max).sum() > self.interval
        last_new_C = df[self.C].diff().rolling(window).mean().iloc[-1]
        check_C = last_new_C > self.min_pcr_tests
        if not (check_tests_ending and check_C):
            return df
        # Complement any ending unupdated test values
        # that are not updated daily, by keeping and propagating forward previous valid diff()
        # min_index: index for first ending max test reoccurrence
        min_index = df[self.TESTS].idxmax() + 1
        first_value = df.loc[min_index, self.TESTS]
        df_ending = df.copy()
        df_ending.loc[df_ending.duplicated([self.TESTS], keep="first"), self.TESTS] = None
        diff_series = df_ending[self.TESTS].diff().ffill().fillna(0).astype(np.int64)
        diff_series.loc[diff_series.duplicated(keep="last")] = None
        diff_series.interpolate(
            method="linear", inplace=True, limit_direction="both")
        df.loc[min_index:, self.TESTS] = first_value + diff_series.loc[min_index:].cumsum()
        return df

    def _pcr_partial_complement(self, before_df, variable):
        """
        If there are missing values in variable column,
        apply partial compliment (bfill, ffill) to all columns.

        Args:
            before_df (pandas.DataFrame):
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Tests, Confirmed, Tests_diff, C_diff
            variable: the desired column to use

        Returns:
            pandas.DataFrame: complemented records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Tests, Confirmed, Tests_diff, C_diff

        Note:
            Filling NA with 0 will be always applied.
        """
        df = before_df.copy()
        df[self.TESTS].fillna(0, inplace=True)
        if self.TESTS_DIFF in df.columns:
            df[self.TESTS_DIFF].fillna(0, inplace=True)
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
                Index reset index
                Columns Date (pandas.TimeStamp), Tests, Confirmed
            window (int): window of moving average, >= 1

        Returns:
            tuple (pandas.DataFrame, bool):
                pandas.DataFrame
                    Index
                        reset index
                    Columns
                        - Date (pd.Timestamp): Observation date
                        - Tests (int): the number of total tests performed
                        - Confirmed (int): the number of confirmed cases
                        - Tests_diff (int): daily tests performed
                        - Confirmed_diff (int): daily confirmed cases
                bool: True if complement is needed or False
        """
        df = before_df.copy()
        df[self.TESTS].fillna(method="ffill", inplace=True)
        # Confirmed must show monotonic increasing
        df = self._pcr_monotonic(df, self.C)
        df = self._pcr_partial_complement(df, self.TESTS)
        # If Tests values are all valid, with no missing values in-between,
        # they must be monotonically increasing as well
        compare_df = df.copy()
        df = self._pcr_monotonic(df, self.TESTS)
        # Complement any ending unupdated test records
        df = self._pcr_partial_complement_ending(df, window)
        # Complemented or not
        is_complemented = not df.equals(compare_df)
        # Calculate daily values for tests and confirmed (with window=1)
        df[self.TESTS_DIFF] = df[self.TESTS].diff()
        df[self.C_DIFF] = df[self.C].diff()
        # Ensure that tests > confirmed in daily basis
        df.loc[df[self.TESTS_DIFF].abs() < df[self.C_DIFF].abs(), self.TESTS_DIFF] = None
        # Keep valid non-zero values by ignoring zeros at the beginning
        df = df.replace(0, np.nan)
        non_zero_index_start = df[self.TESTS_DIFF].first_valid_index()
        df = df.loc[non_zero_index_start:].reset_index(drop=True)
        non_zero_index_end = df[self.TESTS_DIFF].last_valid_index()
        # Keep valid non-zero values by complementing zeros at the end
        if non_zero_index_end < (len(df) - 1):
            df.loc[non_zero_index_end + 1:, self.TESTS_DIFF] = None
        df = self._pcr_partial_complement(df, self.TESTS_DIFF)
        # Use rolling window for averaging tests and confirmed
        df[self.TESTS_DIFF] = df[self.TESTS_DIFF].rolling(window).mean()
        df[self.C_DIFF] = df[self.C_DIFF].rolling(window).mean()
        df = self._pcr_partial_complement(df, self.TESTS_DIFF)
        # Remove first zero lines due to window
        df = df.replace(0, np.nan)
        non_zero_index_start = df[self.TESTS_DIFF].first_valid_index()
        df = df.loc[non_zero_index_start:].reset_index(drop=True)
        return (df, is_complemented)

    def _pcr_check_preconditions(self, df):
        """
        Check preconditions in order to proceed with PCR data processing.

        Args:
            df (pandas.DataFrame):
                Index Date (pandas.TimeStamp)
                Columns Tests, Confirmed

        Return:
            bool: whether the dataset has sufficient data or not
        """
        df[self.TESTS].fillna(0, inplace=True)
        if self.TESTS_DIFF in df.columns:
            df[self.TESTS_DIFF].fillna(0, inplace=True)
        # Check if the values are zero or nan
        check_zero = df[self.TESTS].max()
        # Check if the number of the missing values
        # is more than 50% of the total values
        check_missing = (df[self.TESTS] == 0).mean() < 0.5
        # Check if the number of the positive unique values
        # is less than 1% of the total values
        positive_df = df.loc[df[self.TESTS] > 0, self.TESTS]
        try:
            check_unique = (positive_df.nunique() / positive_df.size) >= 0.01
        except ZeroDivisionError:
            return False
        # Result
        return check_zero and check_missing and check_unique

    def _subset_select(self, country, province):
        """
        Return subset if available.

        Args:
            country (str): country name
            province (str): province name or "-"
        """
        df = self._subset_by_area(country, province)
        if self._pcr_check_preconditions(df):
            return df
        # Failed in retrieving sufficient data
        raise PCRIncorrectPreconditionError(
            country=country, province=province, details="Too many missing Tests records")

    def positive_rate(self, country, province=None, window=7, last_date=None, show_figure=True, filename=None):
        """
        Return the PCR rate of a country as a dataframe.

        Args:
            country(str): country name or ISO3 code
            province(str or None): province name
            window (int): window of moving average, >= 1
            last_date (str or None): the last date of the total tests records or None (max date of main dataset)
            show_figure (bool): if True, show the records as a line-plot.
            filename (str): filename of the figure, or None (display figure)

        Raises:
            covsirphy.PCRIncorrectPreconditionError: the dataset has too many missing values

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.TimeStamp): Observation date
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
                    - Tests_diff (int): daily tests performed
                    - Confirmed_diff (int): daily confirmed cases
                    - Test_positive_rate (float): positive rate (%) of the daily cases over the total daily tests performed

        Note:
            If non monotonic records were found for either confirmed cases or tests,
            "with partially complemented tests data" will be added to the title of the figure.
        """
        window = self._ensure_natural_int(window, name="window")
        # Subset with area
        country_alias = self.ensure_country_name(country)
        province = province or self.NA
        try:
            subset_df = self._subset_select(country_alias, province)
        except PCRIncorrectPreconditionError:
            raise PCRIncorrectPreconditionError(
                country=country, province=province, details="Too many missing Tests records") from None
        # Limit tests records to last date
        if last_date is not None:
            subset_df = subset_df.loc[subset_df[self.DATE] <= pd.to_datetime(last_date)]
        # Process PCR data
        df, is_complemented = self._pcr_processing(subset_df, window)
        # Calculate PCR values
        df[self.PCR_RATE] = df[[self.C_DIFF, self.TESTS_DIFF]].apply(
            lambda x: x[0] / x[1] * 100 if x[1] > self.min_pcr_tests else 0, axis=1)
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

    def subset(self, country, province=None, start_date=None, end_date=None):
        """
        Return subset of the country/province and start/end date.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Tests (int): the number of total tests performed
                    - Tests_diff (int): daily number of tests on date
                    - Confirmed (int): the number of confirmed cases
        """
        country_alias = self.ensure_country_name(country)
        df = self._subset_select(country=country_alias, province=province or self.NA)
        # Calculate Tests_diff
        df[self.TESTS_DIFF] = df[self.TESTS].diff().fillna(0)
        df.loc[df[self.TESTS_DIFF] < 0, self.TESTS_DIFF] = 0
        df[self.TESTS_DIFF] = df[self.TESTS_DIFF].astype(np.int64)
        df = df.loc[:, [self.DATE, self.TESTS, self.TESTS_DIFF, self.C]]
        # Subset with Start/end date
        if start_date is None and end_date is None:
            return df.reset_index(drop=True)
        series = df[self.DATE].copy()
        start_obj = self._ensure_date(start_date, default=series.min())
        end_obj = self._ensure_date(end_date, default=series.max())
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date)
        return df.reset_index(drop=True)

    def map(self, country=None, variable="Tests", date=None, **kwargs):
        """
        Create colored map with the number of tests.

        Args:
            country (str or None): country name or None (global map)
            variable (str): always 'vaccinations'
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()

        Raises:
            NotImplementedError: @variable was specified

        Note:
            When @country is None, country level data will be shown on global map.
            When @country is a country name, province level data will be shown on country map.
        """
        if variable != self.TESTS:
            raise NotImplementedError(f"@variable cannot be changed, always {self.TESTS}.")
        # Date
        date_str = date or self.cleaned(
        )[self.DATE].max().strftime(self.DATE_FORMAT)
        country_str = country or "Global"
        title = f"{country_str}: the number of {variable.lower()} on {date_str}"
        # Global map
        if country is None:
            return self._colored_map_global(
                variable=variable, title=title, date=date, **kwargs)
        # Country-specific map
        return self._colored_map_country(
            country=country, variable=variable, title=title, date=date, **kwargs)
