#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from covsirphy.util.error import NotInteractiveError
from covsirphy.util.plotting import line_plot
from covsirphy.util.term import Term
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
        population_data = self._ensure_instance(
            population_data, PopulationData, name="population_data")
        self.population = population_data.value(country, province=province)
        # Records
        self.jhu_data = self._ensure_instance(
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
        # Interactive (True) / script (False) mode
        self._interactive = hasattr(sys, "ps1")

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
        self._ensure_date_order(self._first_date, date, name="date")
        self._ensure_date_order(date, self._last_date, name="date")
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
        self._ensure_date_order(self._first_date, date, name="date")
        self._ensure_date_order(date, self._last_date, name="date")
        self._last_date = date
        self.init_records()

    @property
    def interactive(self):
        """
        bool: interactive mode (display figures) or not

        Note:
            When running scripts, interactive mode cannot be selected.
        """
        return self._interactive

    @interactive.setter
    def interactive(self, is_interactive):
        if not hasattr(sys, "ps1") and is_interactive:
            raise NotInteractiveError
        self._interactive = hasattr(sys, "ps1") and bool(is_interactive)

    def line_plot(self, df, show_figure=True, filename=None, **kwargs):
        """
        Display or save a line plot of the dataframe.

        Args:
            show_figure (bool): whether show figure when interactive mode or not
            filename (str or None): filename of the figure or None (not save) when script mode

        Note:
            When interactive mode and @show_figure is True, display the figure.
            When script mode and filename is not None, save the figure.
            When using interactive shell, we can change the modes by Scenario.interactive = True/False.
        """
        if self._interactive and show_figure:
            return line_plot(df=df, filename=None, **kwargs)
        if not self._interactive and filename is not None:
            return line_plot(df=df, filename=filename, **kwargs)

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

    def show_complement(self, **kwargs):
        """
        Show the details of complement that was (or will be) performed for the records.

        Args:
            kwargs: keyword arguments of JHUDataComplementHandler() i.e. control factors of complement

        Returns:
            pandas.DataFrame: as the same as `JHUData.show_complement()
        """
        return self.jhu_data.show_complement(
            country=self.country, province=self.province,
            start_date=self._first_date, end_date=self._last_date, **kwargs)

    def records(self, variables=None, **kwargs):
        """
        Return the records as a dataframe.

        Args:
            show_figure (bool): if True, show the records as a line-plot.
            variables (list[str] or None): variables to include, Infected/Fatal/Recovered when None
            kwargs: the other keyword arguments of Scenario.line_plot()

        Returns:
            pandas.DataFrame

                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Columns set by @variables (int)

        Note:
            - Records with Recovered > 0 will be selected.
            - If complement was performed by Scenario.complement() or Scenario(auto_complement=True),
            The kind of complement will be added to the title of the figure.
            - @variables can be selected from Susceptible/Confirmed/Infected/Fatal/Recovered.
        """
        variables = self._ensure_list(
            variables or [self.CI, self.F, self.R],
            candidates=[self.S, *self.VALUE_COLUMNS], name="variables")
        df = self.record_df.loc[:, [self.DATE, *variables]]
        if self._complemented:
            title = f"{self.area}: Cases over time\nwith {self._complemented}"
        else:
            title = f"{self.area}: Cases over time"
        self.line_plot(
            df=df.set_index(self.DATE), title=title, y_integer=True, **kwargs)
        return df

    def records_diff(self, variables=None, window=7, **kwargs):
        """
        Return the number of daily new cases (the first discreate difference of records).

        Args:
            variables (str or None): variables to show
            window (int): window of moving average, >= 1
            kwargs: the other keyword arguments of Scenario.line_plot()

        Returns:
            pandas.DataFrame
                Index
                    - Date (pd.TimeStamp): Observation date
                Columns
                    - Confirmed (int): daily new cases of Confirmed, if calculated
                    - Infected (int):  daily new cases of Infected, if calculated
                    - Fatal (int):  daily new cases of Fatal, if calculated
                    - Recovered (int):  daily new cases of Recovered, if calculated

        Note:
            @variables will be selected from Confirmed, Infected, Fatal and Recovered.
            If None was set as @variables, ["Confirmed", "Fatal", "Recovered"] will be used.
        """
        variables = self._ensure_list(
            variables or [self.C, self.F, self.R], candidates=self.VALUE_COLUMNS, name="variables")
        window = self._ensure_natural_int(window, name="window")
        df = self.record_df.set_index(self.DATE)[variables]
        df = df.diff().dropna()
        df = df.rolling(window=window).mean().dropna().astype(np.int64)
        if self._complemented:
            title = f"{self.area}: Daily new cases\nwith {self._complemented}"
        else:
            title = f"{self.area}: Daily new cases"
        self.line_plot(df=df, title=title, y_integer=True, **kwargs)
        return df
