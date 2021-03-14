#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy.trend.sr_change import _SRChange


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
        min_size (int): minimum value of phase length [days], over 2

    Note:
        "Phase" means a sequential dates in which the parameters of SIR-derived models are fixed.
        "Change points" means the dates when trend was changed.
        "Change points" is the same as the start dates of phases except for the 0th phase.
    """

    def __init__(self, data, min_size=5):
        self._ensure_dataframe(data, name="data", columns=self.SUB_COLUMNS)
        # Index: Date, Columns: the number cases
        self._record_df = data.groupby(self.DATE).last()
        # Minimum size of phases
        self._min_size = self._ensure_int_range(min_size, name="min_size", value_range=(3, None))
        if len(self._record_df) < self._min_size * 2:
            raise ValueError(f"More than {min_size * 2} records must be included because @min_size is {min_size}.")
        # The first/last date
        self._first_point = self._record_df.index.min()
        self._last_point = self._record_df.index.max()
        # Change points: list[pandas.Timestamp]
        self._points = []

    def reset(self):
        """
        Reset the phase setting with the end dates of the records.

        Returns:
            covsirphy.TrendDetector: self
        """
        self._points = []
        return self

    def dates(self):
        """
        Return the list of start dates and end dates.

        Returns:
            tuple(list[str], list[str]): list of start dates and end dates
        """
        points = self._points[:]
        # Start dates
        start_dates = [date.strftime(self.DATE_FORMAT) for date in [self._first_point, *points]]
        # End dates
        end_dates = [self.yesterday(date.strftime(self.DATE_FORMAT)) for date in points]
        end_dates.append(self._last_point.strftime(self.DATE_FORMAT))
        return (start_dates, end_dates)

    def summary(self):
        """
        Summarize the phases with a dataframe.

        Returns:
            pandas.Dataframe:
                Index
                    (str): phase names
                Columns
                    - Start (pandas.Timestamp): star dates
                    - End (pandas.Timestamp): end dates
                    - RMSLE_S-R: RMSLE score on S-R plane
        """
        start_dates, end_dates = self.dates()
        return pd.DataFrame(
            {
                self.START: start_dates,
                self.END: end_dates,
                "RMSLE_S-R": _SRChange(sr_df=self._record_df).score(change_points=self._points),
            },
            index=[self.num2str(num) for num in range(len(self._points) + 1)]
        )

    def sr(self):
        """
        Perform S-R trend analysis.

        Returns:
            covsirphy.TrendDetector: self
        """
        finder = _SRChange(sr_df=self._record_df)
        points = finder.run(min_size=self._min_size)
        self._points = sorted(set(self._points) | set(points))
        return self

    def sr_show(self, **kwargs):
        """
        Show the trend on S-R plane.
        """
        pass
