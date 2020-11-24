#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.country_data import CountryData


class JHUData(CleaningBase):
    """
    Data cleaning of JHU-style dataset.

    Args:
        filename (str): CSV filename of the dataset
        citation (str): citation
    """

    def __init__(self, filename, citation=None):
        super().__init__(filename, citation)
        self._closing_period = None

    def cleaned(self, **kwargs):
        """
        Return the cleaned dataset.

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
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        if "population" in kwargs.keys():
            raise ValueError(
                "@population was removed in JHUData.cleaned(). Please use JHUData.subset()")
        df = self._cleaned_df.copy()
        df = df.loc[:, self.COLUMNS]
        return df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrite super()._cleaning() method.

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - ISO3 (str): ISO3 code
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = super()._cleaning()
        # Rename the columns
        df = df.rename(
            {
                "ObservationDate": self.DATE,
                "ISO3": self.ISO3,
                "Country/Region": self.COUNTRY,
                "Province/State": self.PROVINCE,
                "Confirmed": self.C,
                "Deaths": self.F,
                "Recovered": self.R
            },
            axis=1
        )
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.DATE, self.ISO3, self.COUNTRY, self.PROVINCE, self.C, self.F, self.R
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
        df = df.fillna(method="ffill").fillna(0)
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df[self.VALUE_COLUMNS] = df[self.VALUE_COLUMNS].astype(np.int64)
        df = df.loc[:, [self.ISO3, *self.COLUMNS]].reset_index(drop=True)
        return df

    def replace(self, country_data):
        """
        Replace a part of cleaned dataset with a dataframe.

        Args:
            country_data (covsirphy.CountryData): dataset object of the country

        Returns:
            covsirphy.JHUData: self
        """
        self.ensure_instance(country_data, CountryData, name="country_data")
        # Read new dataset
        country = country_data.country
        new = country_data.cleaned()
        new[self.ISO3] = self.country_to_iso3(country)
        # Remove the data in the country from JHU dataset
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] != country, :]
        # Combine JHU data and the new data
        df = pd.concat([df, new], axis=0, sort=False)
        self._cleaned_df = df.copy()
        return self

    def _subset(self, country, province, start_date, end_date, population):
        """
        Return the subset of dataset.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int or None): population value

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Notes:
            If @population is not None, the number of susceptible cases will be calculated.
        """
        # Subset with area and start/end date
        try:
            df = super().subset(
                country=country, province=province, start_date=start_date, end_date=end_date)
        except KeyError:
            columns = self.NLOC_COLUMNS if population is None else self.SUB_COLUMNS
            return pd.DataFrame(columns=columns)
        # Calculate Susceptible if population value was applied
        if population is None:
            return df
        df.loc[:, self.S] = population - df.loc[:, self.C]
        return df

    def subset(self, country, province=None, start_date=None, end_date=None, population=None):
        """
        Return the subset of dataset with Recovered > 0.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int or None): population value

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Notes:
            If @population is not None, the number of susceptible cases will be calculated.
        """
        area = self.area_name(country, province=province)
        # Subset with area, start/end date and calculate Susceptible
        subset_df = self._subset(
            country=country, province=province,
            start_date=start_date, end_date=end_date, population=population)
        if subset_df.empty:
            area = self.area_name(country, province=province)
            s_fr = "" if start_date is None else f" from {start_date}"
            s_to = "" if end_date is None else f" from {end_date}"
            raise KeyError(f"Records in {area}{s_fr}{s_to} are un-registered.")
        # Select records where Recovered > 0
        df = subset_df.loc[subset_df[self.R] > 0, :]
        df = df.reset_index(drop=True)
        if df.empty:
            series = subset_df[self.DATE]
            start_date = start_date or series.min().strftime(self.DATE_FORMAT)
            end_date = end_date or series.max().strftime(self.DATE_FORMAT)
            area = self.area_name(country, province=province)
            raise ValueError(
                f"Records with 'Recovered > 0' in {area} from {start_date} to {end_date} are un-registered.")
        return df

    def to_sr(self, country, province=None,
              start_date=None, end_date=None, population=None):
        """

        Args:
            country (str): country name
            province (str): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int): population value

        Returns:
            pandas.DataFrame
                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases

        Notes:
            @population must be specified.
            Records with Recovered > 0 will be used.
        """
        population = self.ensure_population(population)
        subset_df = self.subset(
            country=country, province=province,
            start_date=start_date, end_date=end_date, population=population
        )
        return subset_df.set_index(self.DATE).loc[:, [self.R, self.S]]

    @classmethod
    def from_dataframe(cls, dataframe):
        """
        Create JHUData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): Cleaned dataset

        Returns:
            covsirphy.JHUData: JHU-style dataset
        """
        instance = cls(filename=None)
        instance._cleaned_df = cls.ensure_dataframe(
            dataframe, name="dataframe", columns=cls.COLUMNS)
        return instance

    def total(self):
        """
        Calculate total number of cases and rates.

        Returns:
            pandas.DataFrame: group-by Date, sum of the values

                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Fatal per Confirmed (int)
                    - Recovered per Confirmed (int)
                    - Fatal per (Fatal or Recovered) (int)
        """
        df = self._cleaned_df.copy()
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
        df = df.groupby(self.DATE).sum()
        total_series = df.loc[:, self.C]
        r_cols = self.RATE_COLUMNS[:]
        df[r_cols[0]] = df[self.F] / total_series
        df[r_cols[1]] = df[self.R] / total_series
        df[r_cols[2]] = df[self.F] / (df[self.F] + df[self.R])
        return df.loc[:, [*self.VALUE_COLUMNS, *r_cols]]

    def countries(self, complement=True, **kwargs):
        """
        Return names of countries where records.

        Args:
            complement (bool): whether say OK for complement or not
            interval (int): expected update interval of the number of recovered cases [days]
            kwargs: the other keyword arguments of JHUData.subset_complement()

        Returns:
            list[str]: list of country names
        """
        df = self._cleaned_df.copy()
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
        # All countries
        all_set = set((df[self.COUNTRY].unique()))
        # Selectable countries without complement
        raw_ok_set = set(df.loc[df[self.R] > 0, self.COUNTRY].unique())
        if not complement:
            return sorted(raw_ok_set)
        # Selectable countries
        comp_ok_list = [
            country for country in all_set - raw_ok_set
            if not self.subset_complement(country=country, **kwargs)[0].empty]
        return sorted(raw_ok_set | set(comp_ok_list))

    def calculate_closing_period(self):
        """
        Calculate mode value of closing period, time from confirmation to get outcome.

        Returns:
            int: closing period [days]
        """
        # Get cleaned dataset at country level
        df = self._cleaned_df.copy()
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
        # Select records of countries where recovered values are reported
        df = df.groupby(self.COUNTRY).filter(lambda x: x[self.R].sum() != 0)
        # Total number of confirmed/closed cases of selected records
        df = df.groupby(self.DATE).sum()
        df[self.FR] = df[[self.F, self.R]].sum(axis=1)
        df = df.loc[:, [self.C, self.FR]]
        # Calculate how many days to confirmed, closed
        df = df.unstack().reset_index()
        df.columns = ["Variable", self.DATE, "Number"]
        df["Days"] = (df[self.DATE] - df[self.DATE].min()).dt.days
        df = df.pivot_table(values="Days", index="Number", columns="Variable")
        df = df.interpolate(limit_direction="both").fillna(method="ffill")
        df["Elapsed"] = df[self.FR] - df[self.C]
        df = df.loc[df["Elapsed"] > 0]
        # Calculate mode value of closing period
        return df["Elapsed"].mode().astype(np.int64).values[0]

    def _complement_non_monotonic(self, subset_df, monotonic_columns):
        """
        Make the number of cases show monotonic increasing, if necessary.

        Args:
            subset_df (pandas.DataFrame): subset records with country, province and start/end dates
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Fatal (int): the number of fatal cases
                    - the other columns will be ignored
            monotonic_columns (list[str]): columns that show monotonic increasing

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases, if @subset_df has
        """
        # If all variables show monotic increasing, complement will not be done
        columns = [
            col for col in monotonic_columns if not subset_df[col].is_monotonic_increasing]
        if not columns:
            return subset_df
        # Complement
        df = subset_df.set_index(self.DATE)
        for col in columns:
            decreased_dates = df[df[col].diff() < 0].index.tolist()
            for date in decreased_dates:
                # Raw value on the decreased date
                raw_last = df.loc[date, col]
                # Extrapolated value on the date
                series = df.loc[:date, col]
                series.iloc[-1] = None
                series.interpolate(method="spline", order=1, inplace=True)
                series.fillna(method="ffill", inplace=True)
                # Reduce values to the previous date
                df.loc[:date, col] = series * raw_last / series.iloc[-1]
                df[col] = df[col].astype(np.int64)
        # Calculate Infected
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        return df.reset_index()

    def _complement_recovered_full(self, subset_df, max_ignored):
        """
        Estimate the number of recovered cases with closing period, if necessary.

        Args:
            subset_df (pandas.DataFrame): subset records with country, province and start/end dates
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Fatal (int): the number of fatal cases
                    - the other columns will be ignored
            max_ignored (int): Max number of recovered cases to be ignored [cases]

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases, if @subset_df has
        """
        c, f, r = subset_df[[self.C, self.F, self.R]].max().tolist()
        if r > max_ignored and r > (c - f) * 0.1:
            return subset_df
        df = subset_df.set_index(self.DATE)
        # Closing period
        self._closing_period = self._closing_period or self.calculate_closing_period()
        # Estimate recovered records
        shifted = df[self.C].shift(periods=self._closing_period, freq="D")
        df[self.R] = shifted - df[self.F]
        df.loc[df[self.R] < 0, self.R] = None
        df[self.R].interpolate(method="spline", order=1, inplace=True)
        df[self.R] = df[self.R].fillna(0).astype(np.int64)
        df = self._complement_non_monotonic(
            df.reset_index(), [self.R]).set_index(self.DATE)
        # Re-calculate infected records
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        return df.reset_index()

    def _complement_recovered_partial(self, subset_df, interval, max_ignored):
        """
        Complement the number of recovered cases when not updated, if necessary.

        Args:
            subset_df (pandas.DataFrame): subset records with country, province and start/end dates
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Fatal (int): the number of fatal cases
                    - the other columns will be ignored
            interval (int): expected update interval of the number of recovered cases [days]
            max_ignored (int): Max number of recovered cases to be ignored [cases]

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases, if @subset_df has

        Notes:
            If the number of recovered cases did not change
            for more than @interval days after reached @max_ignored cases,
            complement will be applied to the number of recovered cases.
        """
        df = subset_df.copy()
        # If updated, do not perform complement
        series = df.loc[df[self.R] > max_ignored, self.R]
        max_frequency = series.value_counts().max()
        if max_frequency <= interval:
            return subset_df
        # Complement
        df.loc[df.duplicated([self.R], keep="last"), self.R] = None
        df[self.R].interpolate(
            method="linear", inplace=True, limit_direction="both")
        df[self.R] = df[self.R].fillna(method="bfill").round().astype(np.int64)
        # Calculate Infected
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        return df

    def subset_complement(self, country, province=None,
                          start_date=None, end_date=None, population=None,
                          interval=2, max_ignored=100):
        """
        Return the subset of dataset and complement recovered data, if necessary.
        Records with Recovered > 0 will be selected.

        Args:
            country(str): country name or ISO3 code
            province(str or None): province name
            start_date(str or None): start date, like 22Jan2020
            end_date(str or None): end date, like 01Feb2020
            population(int or None): population value
            interval (int): expected update interval of the number of recovered cases [days]
            max_ignored (int): Max number of recovered cases to be ignored [cases]

        Returns:
            tuple(pandas.DataFrame, bool):
                pandas.DataFrame:
                    Index:
                        reset index
                    Columns:
                        - Date(pd.TimeStamp): Observation date
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases ( > 0)
                        - Susceptible(int): the number of susceptible cases, if calculated
                bool: whether recovered data complemented or not

        Notes:
            If @ population is not None, the number of susceptible cases will be calculated.
        """
        # Arguments
        interval = self.ensure_natural_int(interval, name="interval")
        max_ignored = self.ensure_natural_int(max_ignored, name="max_ignored")
        # Subset with area, start/end date and calculate Susceptible
        subset_df = self._subset(
            country=country, province=province,
            start_date=start_date, end_date=end_date, population=population)
        if subset_df.empty:
            area = self.area_name(country, province=province)
            s_fr = "" if start_date is None else f" from {start_date}"
            s_to = "" if end_date is None else f" from {end_date}"
            raise KeyError(f"Records in {area}{s_fr}{s_to} are un-registered.")
        # Complement recovered value if necessary
        df = self._complement_non_monotonic(
            subset_df, monotonic_columns=self.MONO_COLUMNS)
        df = self._complement_recovered_full(df, max_ignored=max_ignored)
        df = self._complement_recovered_partial(
            df, interval=interval, max_ignored=max_ignored)
        # Whether complemented or not
        is_complemented = not df.equals(subset_df)
        # Select records where Recovered > 0
        df = df.loc[df[self.R] > 0, :]
        df = df.reset_index(drop=True)
        return (df, is_complemented)
