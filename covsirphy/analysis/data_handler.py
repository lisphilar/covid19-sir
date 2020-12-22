#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import timedelta
from covsirphy.util.plotting import line_plot
from covsirphy.cleaning.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.population import PopulationData


class DataHandler(Term):
    """
    Data handler for scenario analysis.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        country (str): country name
        province (str or None): province name
        auto_complement (bool): if True and necessary, the number of cases will be complemented
    """

    def __init__(self, jhu_data, population_data, country, province=None, auto_complement=True):
        # Population
        population_data = self.ensure_instance(
            population_data, PopulationData, name="population_data")
        self.population = population_data.value(country, province=province)
        # Records
        self.jhu_data = self.ensure_instance(
            jhu_data, JHUData, name="jhu_data")
        # Area name
        self.country = country
        self.province = province or self.UNKNOWN
        self.area = JHUData.area_name(country, province)
        # Whether complement the number of cases or not
        self._auto_complement = bool(auto_complement)
        self._complemented = False
        # Create {scenario_name: PhaseSeries} and set records
        self.record_df = pd.DataFrame()
        self._first_date = None
        self._last_date = None

    def init_records(self):
        """
        Set records.
        Only when auto-complement mode, complement records if necessary.
        """
        # Set records (complement records, if necessary)
        self.record_df, self._complemented = self.jhu_data.records(
            country=self.country, province=self.province,
            start_date=self._first_date, end_date=self._last_date,
            population=self.population,
            auto_complement=self._auto_complement
        )
        # First/last date of the records
        if self._first_date is None:
            series = self.record_df.loc[:, self.DATE]
            self._first_date = series.min().strftime(self.DATE_FORMAT)
            self._last_date = series.max().strftime(self.DATE_FORMAT)

    @property
    def first_date(self):
        """
        str: the first date of the records
        """
        return self._first_date

    @first_date.setter
    def first_date(self, date):
        self.ensure_date_order(self._first_date, date, name="date")
        self.ensure_date_order(date, self._last_date, name="date")
        self._first_date = date
        self.init_records()

    @property
    def last_date(self):
        """
        str: the last date of the records
        """
        return self._last_date

    @last_date.setter
    def last_date(self, date):
        self.ensure_date_order(self._first_date, date, name="date")
        self.ensure_date_order(date, self._last_date, name="date")
        self._last_date = date
        self.init_records()

    def complement(self, interval=2, max_ignored=100):
        """
        Complement the number of recovered cases, if necessary.

        Args:
            interval (int): expected update interval of the number of recovered cases [days]
            max_ignored (int): Max number of recovered cases to be ignored [cases]

        Returns:
            covsirphy.Scenario: self
        """
        self.record_df, self._complemented = self.jhu_data.records(
            country=self.country, province=self.province,
            start_date=self._first_date, end_date=self._last_date,
            population=self.population,
            auto_complement=True, interval=interval, max_ignored=max_ignored
        )
        return self

    def complement_reverse(self):
        """
        Restore the raw records. Reverse method of covsirphy.Scenario.complement().

        Returns:
            covsirphy.Scenario: self
        """
        self.record_df, self._complemented = self.jhu_data.records(
            country=self.country, province=self.province,
            start_date=self._first_date, end_date=self._last_date,
            population=self.population,
            auto_complement=False
        )
        return self

    def records(self, variables=None, show_figure=True, filename=None, **kwargs):
        """
        Return the records as a dataframe.

        Args:
            show_figure (bool): if True, show the records as a line-plot.
            variables (list[str] or None): variables to include, Infected/Fatal/Recovered when None
            filename (str): filename of the figure, or None (show figure)
            kwargs: the other keyword arguments of covsirphy.line_plot()

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Columns set by @variables (int)
        Notes:
            - Records with Recovered > 0 will be selected.
            - If complement was performed by Scenario.complement() or Scenario(auto_complement=True),
            The kind of complement will be added to the title of the figure.
            - @variables can be selected from Susceptible/Confirmed/Infected/Fatal/Recovered.
        """
        variables = self.ensure_list(
            variables or [self.CI, self.F, self.R],
            candidates=[self.S, *self.VALUE_COLUMNS], name="variables")
        df = self.record_df.loc[:, [self.DATE, *variables]]
        if not show_figure:
            return df
        if self._complemented:
            title = f"{self.area}: Cases over time\nwith {self._complemented}"
        else:
            title = f"{self.area}: Cases over time"
        line_plot(
            df.set_index(self.DATE), title, y_integer=True, filename=filename, **kwargs)
        return df

    def _records_diff_fill_ending(self, before_df, variables, max_ignored=100, eval_days=30):
        """
        If there are missing values in variables columns (diff values expected),
        remove ending duplicates and apply partial compliment (bfill, ffill).

        Args:
            before_df (pandas.DataFrame):
                Index:
                    - Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): daily new cases of Confirmed, if calculated
                    - Infected (int):  daily new cases of Infected, if calculated
                    - Fatal (int):  daily new cases of Fatal, if calculated
                    - Recovered (int):  daily new cases of Recovered, if calculated
            variables: the desired columns to use
            max_ignored (int): Max number of [col] cases to be ignored [cases]
            eval_days (int): the number of last days to check if the pandemic is currently outbreaking

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered (all daily values)
        """
        check_df = before_df.copy()
        # Check first if it is currently
        # (in the last eval_days days) outbreaking or not
        # the values are expected to contain already diff values
        check_df = check_df[-eval_days:]
        check_df.replace(0, np.nan, inplace=True)
        last_index = check_df[self.C].last_valid_index()
        check_df = check_df[:last_index]

        # Stopping is considered when either mean value of daily fatal is
        # at most one person or if the daily rate of confirmed and recovered is negative
        check_df.fillna(0, inplace=True)
        sel_C = check_df[self.C] > max_ignored
        sel_F = check_df[self.F].mean() <= 1
        sel_R = check_df[self.R] > max_ignored
        check_C = check_df[self.C].diff().mean() > 0
        check_R = check_df[self.R].diff().mean() > 0
        s_df_1 = check_df.loc[sel_C & check_C]
        s_df_2 = check_df.loc[sel_R & check_R]
        if s_df_1.empty and s_df_2.empty or sel_F:
            return before_df

        df = before_df.copy()
        for col in variables:
            df[col].replace(0, np.nan, inplace=True)
            df_ending = df.copy()
            # Replace duplicate values with NaN
            df_ending.loc[df_ending.duplicated([col], keep=False), col] = None
            # Find last valid index
            non_zero_index_end = df_ending[col].last_valid_index()
            df_ending.loc[non_zero_index_end + timedelta(days=1):, col] = None
            # Update df with only the ending NaN values
            df.loc[non_zero_index_end:, col] = df_ending.loc[non_zero_index_end:, col]
            # Keep the previous valid value for the ending NaN
            df[col].fillna(method="ffill", inplace=True)
        return df

    def records_diff(self, variables=None, window=7, show_figure=True, filename=None, **kwargs):
        """
        Return the number of daily new cases (the first discreate difference of records).

        Args:
            variables (str or None): variables to show
            window (int): window of moving average, >= 1
            show_figure (bool): if True, show the records as a line-plot.
            filename (str): filename of the figure, or None (show figure)
            kwargs: the other keyword arguments of covsirphy.line_plot()

        Returns:
            pandas.DataFrame
                Index:
                    - Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): daily new cases of Confirmed, if calculated
                    - Infected (int):  daily new cases of Infected, if calculated
                    - Fatal (int):  daily new cases of Fatal, if calculated
                    - Recovered (int):  daily new cases of Recovered, if calculated

        Notes:
            @variables will be selected from Confirmed, Infected, Fatal and Recovered.
            If None was set as @variables, ["Confirmed", "Fatal", "Recovered"] will be used.
        """
        variables = self.ensure_list(
            variables or [self.C, self.F, self.R], candidates=self.VALUE_COLUMNS, name="variables")
        window = self.ensure_natural_int(window, name="window")
        df = self.record_df.set_index(self.DATE)[variables]
        df = df.diff().dropna()
        df = self._records_diff_fill_ending(df, variables)
        df = df.rolling(window=window).mean().dropna().astype(np.int64)
        if self.CI in df.columns:
            # Negative infected means greater recovery rate than infection rate
            df.loc[df[self.CI] < 0, self.CI] = 0
        if not show_figure:
            return df
        if self._complemented:
            title = f"{self.area}: Daily new cases\nwith {self._complemented}"
        else:
            title = f"{self.area}: Daily new cases"
        line_plot(df, title, y_integer=True, filename=filename, **kwargs)
        return df
