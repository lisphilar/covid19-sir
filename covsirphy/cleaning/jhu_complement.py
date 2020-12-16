#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import warnings
import numpy as np
from covsirphy.cleaning.term import Term


class JHUDataComplementHandler(Term):
    """
    Complement JHU dataset, if necessary.

    Args:
        recovery_period (int): expected value of recovery period [days]
        interval (int): expected update interval of the number of recovered cases [days]
        max_ignored (int): Max number of recovered cases to be ignored [cases]

    Notes:
        To add new complement solutions, we need to update cls.STATUS_NAME_DICT and self._protocol().
        Status names with high socres will be prioritized when status code will be determined.
        Status code: 'fully complemented recovered data' and so on as noted in self.run() docstring.
    """
    RAW_COLS = [Term.C, Term.F, Term.R]
    # Kind of complement: {score: name}
    STATUS_NAME_DICT = {
        1: "sorting",
        2: "monotonic increasing",
        3: "partially",
        4: "fully",
    }

    def __init__(self, recovery_period, interval, max_ignored):
        # Arguments for complement
        self.recovery_period = self.ensure_natural_int(
            recovery_period, name="recovery_period")
        self.interval = self.ensure_natural_int(interval, name="interval")
        self.max_ignored = self.ensure_natural_int(
            max_ignored, name="max_ignored")

    def _protocol(self):
        """
        Return the list of complement solutions and scores.

        Returns:
            list[str/None, function, int]: nested list of variables to be updated, methods and scores
        """
        return [
            (None, self._pre_processing, 0),
            (self.C, functools.partial(self._monotonic, variable=self.C), 2),
            (self.F, functools.partial(self._monotonic, variable=self.F), 2),
            (self.R, functools.partial(self._monotonic, variable=self.R), 2),
            (self.R, self._recovered_full, 4),
            (self.R, self._recovered_sort, 1),
            (self.R, functools.partial(self._monotonic, variable=self.R), 2),
            (self.R, self._recovered_partial, 3),
            (self.R, functools.partial(self._monotonic, variable=self.R), 2),
            (self.R, self._recovered_sort, 1),
            (None, self._post_processing, 0)
        ]

    def run(self, subset_df):
        """
        Perform complement.

        Args:
            subset_df (pandas.DataFrame): Subset of records
                Index: reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - The other columns will be ignored

        Returns:
            tuple(pandas.DataFrame, str):
                pandas.DataFrame:
                    Index: reset index
                    Columns:
                        - Date(pd.TimeStamp): Observation date
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases
                str: status code

        Notes:
            Status code will be selected from:
                - '' (not complemented)
                - 'monotonic increasing complemented confirmed data'
                - 'monotonic increasing complemented fatal data'
                - 'monotonic increasing complemented recovered data'
                - 'fully complemented recovered data'
                - 'partially complemented recovered data'
        """
        self.ensure_dataframe(
            subset_df, name="subset_df", columns=[self.DATE, *self.RAW_COLS])
        # Initialize
        after_df = subset_df.copy()
        status_dict = dict.fromkeys(self.RAW_COLS, 0)
        # Perform complement one by one
        for (variable, func, score) in self._protocol():
            before_df, after_df = after_df.copy(), func(after_df)
            if after_df.equals(before_df) or variable is None:
                continue
            status_dict[variable] = max(status_dict[variable], score)
        # Create status code
        status_list = [
            f"{self.STATUS_NAME_DICT[score]} complemented {v.lower()} data"
            for (v, score) in status_dict.items() if score]
        status = " and \n".join(status_list)
        return (after_df, status)

    def _pre_processing(self, subset_df):
        """
        Select Confirmed/Fatal/Recovered class from the dataset.

        Args:
            subset_df (pandas.DataFrame): Subset of records
                Index: reset index
                Columns: Date, Confirmed, Fatal, Recovered (the others will be ignored)

        Returns:
            pandas.DataFrame
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered
        """
        return subset_df.set_index(self.DATE).loc[:, self.RAW_COLS]

    def _post_processing(self, df):
        """
        Select Confirmed/Fatal/Recovered class from the dataset.

        Args:
            df (pandas.DataFrame)
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index: reset index
                Columns: Date (pandas.TimeStamp), Confirmed, Infected, Fatal, Recovered (int)
        """
        df = df.astype(np.int64)
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df = df.loc[:, self.VALUE_COLUMNS]
        return df.reset_index()

    def _monotonic(self, df, variable):
        """
        Force the variable show monotonic increasing.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered
            variable (str): variable name to show monotonic increasing

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered
        """
        warnings.simplefilter("ignore", UserWarning)
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

    def _recovered_full(self, df):
        """
        Estimate the number of recovered cases with the value of recovery period.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered
        """
        # Whether complement is necessary or not
        if df[self.R].max() > self.max_ignored:
            # Necessary if sum of recovered is more than 99%
            # of sum of recovered and infected when outbreaking
            sel_1 = df[self.C] > self.max_ignored
            sel_2 = df[self.C].diff().diff().rolling(14).mean() > 0
            s_df = df.loc[
                sel_1 & sel_2 & (df[self.R] > 0.99 * (df[self.C] - df[self.F]))]
            if s_df.empty:
                return df
        # Estimate recovered records
        df[self.R] = (df[self.C] - df[self.F]).shift(
            periods=self.recovery_period, freq="D")
        return df

    def _recovered_partial(self, df):
        """
        If recovered values do not change for more than applied 'self.interval' days
        after reached 'self.max_ignored' cases, interpolate the recovered values.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered
        """
        # Whether complement is necessary or not
        series = df.loc[df[self.R] > self.max_ignored, self.R]
        max_frequency = series.value_counts().max()
        if max_frequency <= self.interval:
            return df
        # Complement
        df.loc[df.duplicated([self.R], keep="last"), self.R] = None
        df[self.R].interpolate(
            method="linear", inplace=True, limit_direction="both")
        df[self.R] = df[self.R].fillna(method="bfill")
        return df

    def _recovered_sort(self, df):
        """
        Sort the absolute values of recovered data.

        Args:
            df (pandas.DataFrame):
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index: Date (pandas.TimeStamp)
                Columns: Confirmed, Fatal, Recovered
        """
        df.loc[:, self.R] = sorted(df[self.R].abs())
        df[self.R].interpolate(method="time", inplace=True)
        df[self.R] = df[self.R].fillna(0).round()
        return df
