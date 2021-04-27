#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import pandas as pd
from covsirphy.util.term import Term


class PhaseTracker(Term):
    """
    Track phase information of one scenario.

    Args:
        data (pandas.DataFrame):
            Index
                reset index
            Columns:
                - Date (pandas.Timestamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases ( > 0)
                - Susceptible (int): the number of susceptible cases
        today (str or pandas.Timestamp): reference date to determine whether a phase is a past phase or not
    """

    def __init__(self, data, today):
        self._ensure_dataframe(data, name="data", columns=self.SUB_COLUMNS)
        self._today = self._ensure_date(today, name="today")
        # Tracker of phase information: index=Date, records of C/I/F/R/S, phase ID (0: not defined)
        self._track_df = data.set_index(self.DATE)
        self._track_df[self.ID] = 0

    def define_phase(self, start, end):
        """
        Define a phase with the series of dates.

        Args:
            start (str or pandas.Timestamp): start date of the new phase
            end (str or pandas.Timestamp): end date of the new phase

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
        if end <= self._today:
            return
        df = pd.DataFrame(
            index=pd.date_range(self._today + timedelta(days=1), end),
            columns=self._track_df.columns)
        df[self.ID] = self._track_df[self.ID].max() + 1
        self._track_df = pd.concat([self._track_df, df], axis=0).resample("D").last()
