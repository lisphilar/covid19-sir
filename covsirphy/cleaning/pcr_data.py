#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from dask import dataframe as dd
from covsirphy.util.plotting import line_plot
from covsirphy.util.error import PCRIncorrectPreconditionError, SubsetNotFoundError
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.country_data import CountryData


class PCRData(CleaningBase):
    """
    Data cleaning of PCR dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        interval (int): expected update interval of the number of confirmed cases and tests [days]
        min_pcr_tests (int): minimum number of valid daily tests performed in order to calculate positive rate
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

    def __init__(self, filename, interval=2, min_pcr_tests=100, citation=None):
        if filename is None:
            self._raw = pd.DataFrame()
            self._cleaned_df = pd.DataFrame(columns=self.PCR_COLUMNS)
        else:
            self._raw = dd.read_csv(
                filename, dtype={"Province/State": "object"}
            ).compute()
            self._cleaned_df = self._cleaning()
        self.interval = self._ensure_natural_int(interval, name="interval")
        self.min_pcr_tests = self._ensure_natural_int(
            min_pcr_tests, name="min_pcr_tests")
        self._citation = citation or ""
        # Cleaned dataset of "Our World In Data"
        self._cleaned_df_owid = pd.DataFrame()
        # Directory that save the file
        if filename is None:
            self._dirpath = Path("input")
        else:
            self._dirpath = Path(filename).resolve().parent

    def cleaned(self):
        """
        Return the cleaned dataset of PCRData with tests and confirmed data.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases

        Note:
            Cleaning method is defined by self._cleaning() method.
        """
        return self._cleaned_df.loc[:, self.PCR_COLUMNS]

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrites super()._cleaning() method.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - ISO3 (str): ISO3 code
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
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
        self._ensure_dataframe(
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
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        return df

    @classmethod
    def from_dataframe(cls, dataframe, directory="input"):
        """
        Create PCRData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): cleaned dataset
            directory (str): directory to save geometry information (for .map() method)

        Returns:
            covsirphy.PCRData: PCR dataset
        """
        instance = cls(filename=None)
        instance.directory = str(directory)
        instance._cleaned_df = cls._ensure_dataframe(
            dataframe, name="dataframe", columns=cls.PCR_COLUMNS)
        return instance

    def _download_ourworldindata(self, filename):
        """
        Download the dataset (ISO code/date/the number of tests) from "Our World In Data" site.
        https://github.com/owid/covid-19-data/tree/master/public/data
        https://ourworldindata.org/coronavirus

        Args:
            filename (str): CSV filename to save the datasetretrieved from "Our World In Data"
        """
        url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv"
        col_dict = {
            "ISO code": self.ISO3,
            "Date": self.DATE,
            "Cumulative total": self.TESTS,
            "Daily change in cumulative total": self.T_DIFF,
        }
        # Download the dataset
        df = self.load(url, columns=list(col_dict))
        # Data cleaning
        df = df.rename(col_dict, axis=1)
        df[self.TESTS] = pd.to_numeric(df[self.TESTS], errors="coerce")
        df[self.TESTS] = df[self.TESTS].fillna(method="ffill").astype(np.int64)
        # Calculate cumulative values if necessary
        df[self.T_DIFF] = df[self.T_DIFF].fillna(0).astype(np.int64)
        na_last_df = df.loc[
            (df[self.TESTS].isna()) & (df[self.DATE] == df[self.DATE].max())]
        re_countries_set = set(na_last_df[self.ISO3].unique())
        df["cumsum"] = df.groupby(self.ISO3)[self.T_DIFF].cumsum()
        df[self.TESTS] = df[[self.ISO3, self.TESTS, "cumsum"]].apply(
            lambda x: x[1] if x[0] in re_countries_set else x[2], axis=1)
        df = df.drop("cumsum", axis=1)
        # Drop duplicated records
        df = df.drop_duplicates(subset=[self.ISO3, self.DATE])
        # Save as CSV file
        df.to_csv(filename, index=False)
        return df

    def use_ourworldindata(self, filename, force=False):
        """
        Set the cleaned dataset retrieved from "Our World In Data" site.
        https://github.com/owid/covid-19-data/tree/master/public/data
        https://ourworldindata.org/coronavirus

        Args:
            filename (str): CSV filename to save the datasetretrieved from "Our World In Data"
            force (bool): if True, always download the dataset from "Our World In Data"
        """
        # Retrieve dataset from "Our World In Data" if necessary
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        if Path(filename).exists() and not force:
            df = self.load(filename, dtype={self.TESTS: np.int64})
        else:
            df = self._download_ourworldindata(filename)
        # Add "Country" and "Confirmed" column using "COVID-19 Data Hub" dataset
        df[self.COUNTRY] = None
        df[self.C] = None
        df.index = df[self.ISO3].str.cat(df[self.DATE], sep="_")
        series = df.loc[:, self.TESTS]
        hub_df = self._cleaned_df.copy()
        hub_df = hub_df.loc[hub_df[self.PROVINCE] == self.UNKNOWN]
        hub_df.index = hub_df[self.ISO3].str.cat(
            hub_df[self.DATE].astype(str), sep="_")
        df.update(hub_df)
        df[self.TESTS] = series
        df = df.dropna().reset_index(drop=True)
        # Add "Province" column (Unknown because not)
        df[self.PROVINCE] = self.UNKNOWN
        # Data types
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        df[self.COUNTRY] = df[self.COUNTRY].astype("category")
        df[self.PROVINCE] = df[self.PROVINCE].astype("category")
        df[self.C] = df[self.C].astype(np.int64)
        # Save the dataframe as the cleaned dataset
        self._cleaned_df_owid = df.reset_index(drop=True)
        # Update citation
        self._citation += "\nHasell, J., Mathieu, E., Beltekian, D. et al." \
            " A cross-country database of COVID-19 testing. Sci Data 7, 345 (2020)." \
            " https://doi.org/10.1038/s41597-020-00688-8"
        return self

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
        new = self._ensure_dataframe(
            country_data.cleaned(), name="the raw data", columns=self.PCR_COLUMNS)
        new = new.loc[:, self.PCR_COLUMNS]
        new[self.ISO3] = self.country_to_iso3(country)
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
        # that are not updated daily, by keeping and
        # propagating forward previous valid diff()
        # min_index: index for first ending max test reoccurrence
        min_index = df[self.TESTS].idxmax() + 1
        first_value = df.loc[min_index, self.TESTS]
        df_ending = df.copy()
        df_ending.loc[df_ending.duplicated(
            [self.TESTS], keep="first"), self.TESTS] = None
        diff_series = df_ending[self.TESTS].diff(
        ).ffill().fillna(0).astype(np.int64)
        diff_series.loc[diff_series.duplicated(keep="last")] = None
        diff_series.interpolate(
            method="linear", inplace=True, limit_direction="both")
        df.loc[min_index:, self.TESTS] = first_value + \
            diff_series.loc[min_index:].cumsum()
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
        if self.T_DIFF in df.columns:
            df[self.T_DIFF].fillna(0, inplace=True)
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
                        - Date (pd.TimeStamp): Observation date
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
        if self.T_DIFF in df.columns:
            df[self.T_DIFF].fillna(0, inplace=True)
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

    def _subset_by_area(self, country, province, dataset="COVID-19 Data Hub"):
        """
        Return the subset of "Our World In Data".

        Args:
            country (str): country name
            province (str): province name or "-"
            dataset (str): 'COVID-19 Data Hub' or 'Our World In Data'
        """
        dataset_dict = {
            "COVID-19 Data Hub": self._cleaned_df,
            "Our World In Data": self._cleaned_df_owid,
        }
        df = dataset_dict[dataset].copy()
        return df.loc[(df[self.COUNTRY] == country) & (df[self.PROVINCE] == province)]

    def _subset_select(self, country, province):
        """
        When only "Our World In Data" has sufficient data, the subset of this dataset will be returned.
        If not, "COVID-19 Data Hub" will be selected.

        Args:
            country (str): country name
            province (str): province name or "-"
        """
        # If 'COVID-19 Data Hub' has sufficient data for the area, it will be used
        hub_df = self._subset_by_area(
            country, province, dataset="COVID-19 Data Hub")
        if self._pcr_check_preconditions(hub_df):
            return hub_df
        # If 'Our World In Data' has sufficient data for the area, it will be used
        owid_df = self._subset_by_area(
            country, province, dataset="Our World In Data")
        if self._pcr_check_preconditions(owid_df):
            return owid_df
        # Failed in retrieving sufficient data
        raise PCRIncorrectPreconditionError(
            country=country, province=province, message="Too many missing Tests records")

    def positive_rate(self, country, province=None, window=7, show_figure=True, filename=None):
        """
        Return the PCR rate of a country as a dataframe.

        Args:
            country(str): country name or ISO3 code
            province(str or None): province name
            window (int): window of moving average, >= 1
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
        province = province or self.UNKNOWN
        try:
            subset_df = self._subset_select(country_alias, province)
        except PCRIncorrectPreconditionError:
            raise PCRIncorrectPreconditionError(
                country=country, province=province, message="Too many missing Tests records") from None
        # Process PCR data
        df, is_complemented = self._pcr_processing(subset_df, window)
        # Calculate PCR values
        df[self.PCR_RATE] = df[[self.C_DIFF, self.T_DIFF]].apply(
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
                    - Date (pd.TimeStamp): Observation date
                    - Tests (int): the number of total tests performed
                    - Confirmed (int): the number of confirmed cases
        """
        country_alias = self.ensure_country_name(country)
        df = self._subset_select(country=country_alias, province=province or self.UNKNOWN)
        df = df.drop(
            [self.COUNTRY, self.ISO3, self.PROVINCE], axis=1)
        # Subset with Start/end date
        if start_date is None and end_date is None:
            return df.reset_index(drop=True)
        series = df[self.DATE].copy()
        start_obj = self.date_obj(date_str=start_date, default=series.min())
        end_obj = self.date_obj(date_str=end_date, default=series.max())
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
