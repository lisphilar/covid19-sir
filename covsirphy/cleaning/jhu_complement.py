#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import numpy as np
from covsirphy.cleaning.term import Term


class JHUDataComplementHandler(Term):
    """
    Complement JHU dataset, if necessary.

    Args:
        subset_df (pandas.DataFrame): Subset of records
            Index: reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - The other columns will be ignored
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

    def __init__(self, subset_df, recovery_period, interval, max_ignored):
        self.ensure_dataframe(
            subset_df, name="subset_df", columns=[self.DATE, *self.RAW_COLS])
        # Before complement: Index=Date, Columns=Confirmed, Fatal, Recovered
        self._raw_df = subset_df.set_index(self.DATE).loc[:, self.RAW_COLS]
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
            list[function, str, int]: nested list of variables to be updated, methods and scores
        """
        return [
            (self.C, functools.partial(self._monotonic(variable=self.C)), 2),
            (self.F, functools.partial(self._monotonic(variable=self.F)), 2),
            (self.R, functools.partial(self._monotonic(variable=self.R)), 2),
            (self.R, self._recovered_full, 4),
            (self.R, self._recovered_partial, 3),
            (self.R, self._recovered_sort, 1),
        ]

    def run(self):
        """
        Perform complement.

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
        # Complement
        df, status = self._run(self._raw_df)
        # Calculate infected data
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df = df.loc[:, self.VALUE_COLUMNS]
        df = df.astype(np.int64).reset_index()
        return (df, status)

    def _run(self):
        """
        Perform complement per protocol defined by self._protocol method.

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
                str: status code ("" or kind of complement defined in self.STATUS_NAME_DICT)
        """
        df = self._raw_df.copy()
        status_dict = dict.fromkeys(self.RA_COLS, 0)
        # Perform complement
        for (variable, func, score) in self._protocol():
            df = func(df)
            if not df.equals(self._raw_df):
                status_dict[variable] = max(status_dict[variable], score)
        # Create status code
        status_list = [
            f"{self.STATUS_NAME_DICT[score]} complemented {v.lower()} data"
            for (v, score) in status_dict.items() if score]
        status = " and ".join(status_list)
        return (df, status)

    def _monotonic(self, df, variable):
        pass

    def _recovered_full(self, df):
        pass

    def _recovered_partial(self, df):
        pass

    def _recovered_sort(self, df):
        pass
