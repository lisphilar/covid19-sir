#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.util.term import Term
from covsirphy.trend.trend_plot import trend_plot
from covsirphy.trend.trend_detector import TrendDetector


class PhaseTracker(Term):
    """
    Track phase information of one scenario.

    Args:
        data (pandas.DataFrame):
            Index
                reset index
            Columns
                - Date (pandas.Timestamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - Susceptible (int): the number of susceptible cases
        today (str or pandas.Timestamp): reference date to determine whether a phase is a past phase or not
        area (str): area name, like Japan/Tokyo
    """

    def __init__(self, data, today, area):
        self._ensure_dataframe(data, name="data", columns=self.SUB_COLUMNS)
        self._today = self._ensure_date(today, name="today")
        self._area = str(area)
        # Tracker of phase information: index=Date, records of C/I/F/R/S, phase ID (0: not defined)
        self._track_df = data.set_index(self.DATE)
        self._track_df[self.ID] = 0

    def define_phase(self, start, end):
        """
        Define a phase with the series of dates.

        Args:
            start (str or pandas.Timestamp): start date of the new phase
            end (str or pandas.Timestamp): end date of the new phase

        Returns:
            covsirphy.PhaseTracker: self

        Note:
            When today is in the range of (start, end), a past phase and a future phase will be created.
        """
        start = self._ensure_date(start, name="start")
        end = self._ensure_date(end, name="end")
        # Start date must be over the first date of records
        self._ensure_date_order(self._track_df.index.min(), start, name="start")
        # Add a past phase (start -> min(end, today))
        if start <= self._today:
            self._track_df.loc[start:min(self._today, end), self.ID] = self._track_df[self.ID].max() + 1
        # Add a future phase (tomorrow -> end)
        if self._today < end:
            phase_start = max(self._today + timedelta(days=1), start)
            df = pd.DataFrame(
                index=pd.date_range(phase_start, end), columns=self._track_df.columns)
            df.index.name = self.DATE
            df[self.ID] = self._track_df[self.ID].max() + 1
            self._track_df = pd.concat([self._track_df, df], axis=0).resample("D").last()
        # Fill in blanks
        series = self._track_df[self.ID].copy()
        self._track_df.loc[(series.index <= end) & (series == 0), self.ID] = series.max() + 1
        return self

    def remove_phase(self, start, end):
        """
        Remove phase information from the date range.

        Args:
            start (str or pandas.Timestamp): start date of the phase to remove
            end (str or pandas.Timestamp): end date of the phase to remove

        Returns:
            covsirphy.PhaseTracker: self
        """
        start = self._ensure_date(start, name="start")
        end = self._ensure_date(end, name="end")
        self._track_df.loc[start:end, self.ID] = 0
        return self

    def track(self):
        """
        Track data with all dates.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases
                    - If available,
                        - Rt (float): phase-dependent reproduction number
                        - (str, float): estimated parameter values, including rho
                        - (int or float): day parameters, including 1/beta [days]
                        - {metric}: score with the estimated parameter values
                        - Trials (int): the number of trials
                        - Runtime (str): runtime of optimization
        """
        return self._track_df.drop(self.ID, axis=1).reset_index()

    def summary(self):
        """
        Summarize phase information.

        Returns:
            pandas.DataFrame
                Index
                    str: phase names
                Columns
                    - Type: 'Past' or 'Future'
                    - Start: start date of the phase
                    - End: end date of the phase
                    - Population: population value of the start date
                    - If available,
                        - ODE (str): ODE model names
                        - Rt (float): phase-dependent reproduction number
                        - (str, float): estimated parameter values, including rho
                        - tau (int): tau value [min]
                        - (int or float): day parameters, including 1/beta [days]
                        - {metric}: score with the estimated parameter values
                        - Trials (int): the number of trials
                        - Runtime (str): runtime of optimization
        """
        # Remove un-registered phase
        track_df = self._track_df.reset_index()
        track_df = track_df.loc[track_df[self.ID] != 0]
        # -> index=phase names, columns=Start/variables,.../End
        track_df[self.ID], _ = track_df[self.ID].factorize()
        first_df = track_df.groupby(self.ID).first()
        df = first_df.join(track_df.groupby(self.ID).last(), rsuffix="_last")
        df = df.rename(columns={self.DATE: self.START, f"{self.DATE}_last": self.END})
        df.index.name = None
        df.index = [self.num2str(num) for num in df.index]
        df = df.loc[:, [col for col in df.columns if "_last" not in col]]
        # Calculate phase types: Past or Future
        df[self.TENSE] = (df[self.START] <= self._today).map({True: self.PAST, False: self.FUTURE})
        # Calculate population values
        df[self.N] = df[[self.S, self.C]].sum(axis=1)
        # Set the order of columns
        df = df.drop([self.C, self.CI, self.F, self.R, self.S], axis=1)
        fixed_cols = self.TENSE, self.START, self.END, self.N
        others = [col for col in df.columns if col not in set(fixed_cols)]
        return df.loc[:, [*fixed_cols, *others]]

    def trend(self, force, show_figure, **kwargs):
        """
        Define past phases with S-R trend analysis.

        Args:
            force (bool): if True, change points will be over-written
            show_figure (bool): if True, show the result as a figure
            kwargs: keyword arguments of covsirphy.TrendDetector(), .TrendDetector.sr() and .trend_plot()

        Returns:
            covsirphy.PhaseTracker: self
        """
        df = self._track_df.loc[:self._today].reset_index()[self.SUB_COLUMNS]
        detector = TrendDetector(data=df, area=self._area, **find_args(TrendDetector, **kwargs))
        # Perform S-R trend analysis
        detector.sr(**find_args(TrendDetector.sr, **kwargs))
        # Register phases
        if force:
            start_dates, end_dates = detector.dates()
            [self.define_phase(start, end) for (start, end) in zip(start_dates, end_dates)]
        # Show S-R plane
        if show_figure:
            detector.show(**find_args(trend_plot, **kwargs))
        return self
