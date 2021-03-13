#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term


class TrendDetector(Term):
    """
    Interface for trend analysis (change point analysis).

    Args:
        data (pandas.DataFrame): data to analyse
            Index:
                reset index
            Column:
                - Date(pd.Timestamp): Observation date
                - Confirmed(int): the number of confirmed cases
                - Infected(int): the number of currently infected cases
                - Fatal(int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - Susceptible(int): the number of susceptible cases

    Note:
        "Phase" means a sequential dates in which the parameters of SIR-derived models are fixed.
        "Change points" means the dates when trend was changed.
        "Change points" is the same as the start dates of phases except for the 0th phase.
    """

    def __init__(self, data):
        self._ensure_dataframe(data, name="data", columns=self.SUB_COLUMNS)
        # Index: Date, Columns: the number cases
        self._record_df = data.groupby(self.DATE).last()
        # The first/last date
        self._first_date = self._record_df.index.min()
        self._last_date = self._record_df.index.max()
        # Change points: list[pandas.Timestamp]
        self._points = []
        self.reset()

    def reset(self):
        """
        Reset the phase setting with the end dates of the records.

        Returns:
            covsirphy.TrendDetector: self
        """
        self._points = [self._last_date]
        return self

    def dates(self):
        """
        Return the list of start dates and end dates.

        Returns:
            tuple(list[str], list[str]): list of start dates and end dates
        """
        points = self._points[:]
        # Start dates
        start_dates = [self._first_date, *points]
        # End dates
        end_dates = [self.yesterday(date) for date in points] + [self._last_date]
        return (start_dates, end_dates)

    def summary(self):
        """
        Summarize the phases with a dataframe.

        Returns:
            pandas.Dataframe:
                Index
                    reset index
                Columns
                    - Start (pandas.Timestamp): star dates
                    - End (pandas.Timestamp): end dates
                    - RMSLE_S-R: RMSLE score in S-R plane
        """
        start_dates, end_dates = self.dates()
        return pd.DataFrame(
            {
                self.START: start_dates,
                self.END: end_dates,
                "RMSLE_S-R": None,
            }
        )

    def sr(self):
        """
        Perform S-R trend analysis.

        Returns:
            covsirphy.TrendDetector: self
        """
        return self

    def sr_show(self):
        """
        Show the trend in S-R plane.
        """
        pass
