#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import warnings
import numpy as np
from datetime import timedelta
from covsirphy.util.error import deprecate
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class JHUDataComplementHandler(Term):
    """
    Complement JHU dataset, if necessary.

    Args:
        recovery_period (int): expected value of recovery period [days]
        interval (int): expected update interval of the number of recovered cases [days]
        max_ignored (int): Max number of recovered cases to be ignored [cases]
        max_ending_unupdated (int): Max number of days to apply full complement, where max recovered cases are not updated [days]
        upper_limit_days (int): maximum number of valid partial recovery periods [days]
        lower_limit_days (int): minimum number of valid partial recovery periods [days]
        upper_percentage (float): fraction of partial recovery periods with value greater than upper_limit_days
        lower_percentage (float): fraction of partial recovery periods with value less than lower_limit_days

    Note:
        To add new complement solutions, we need to update cls.STATUS_NAME_DICT and self._protocol().
        Status names with high scores will be prioritized when status code will be determined.
        Status code: 'fully complemented recovered data' and so on as noted in self.run() docstring.
    """
    # Column names
    RAW_COLS = [Term.C, Term.F, Term.R]
    MONOTONIC_CONFIRMED = "Monotonic_confirmed"
    MONOTONIC_FATAL = "Monotonic_fatal"
    MONOTONIC_RECOVERED = "Monotonic_recovered"
    FULL_RECOVERED = "Full_recovered"
    PARTIAL_RECOVERED = "Partial_recovered"
    RECOVERED_COLS = [MONOTONIC_RECOVERED, FULL_RECOVERED, PARTIAL_RECOVERED]
    SHOW_COMPLEMENT_FULL_COLS = [
        MONOTONIC_CONFIRMED, MONOTONIC_FATAL, *RECOVERED_COLS]
    # Kind of complement: {score: name}
    STATUS_NAME_DICT = {
        1: "sorting",
        2: "monotonic increasing",
        3: "partially",
        4: "fully",
    }

    @deprecate(old="JHUDataComplementHandler", new="DataEngineer", version="2.24.0-xi")
    def __init__(self, recovery_period, interval=2, max_ignored=100,
                 max_ending_unupdated=14, upper_limit_days=90,
                 lower_limit_days=7, upper_percentage=0.5, lower_percentage=0.5):
        # Arguments for complement
        self.recovery_period = Validator(recovery_period, "recovery_period").int(value_range=(1, None))
        self.interval = Validator(interval, "interval").int(value_range=(0, None))
        self.max_ignored = Validator(max_ignored, "max_ignored").int(value_range=(1, None))
        self.max_ending_unupdated = Validator(max_ending_unupdated, "max_ending_unupdated").int(value_range=(0, None))
        self.upper_limit_days = Validator(upper_limit_days, "upper_limit_days").int(value_range=(0, None))
        self.lower_limit_days = Validator(lower_limit_days, "lower_limit_days").int(value_range=(0, None))
        self.upper_percentage = Validator(upper_percentage, "upper_percentage").float(value_range=(0, 100))
        self.lower_percentage = Validator(lower_percentage, "lower_percentage").float(value_range=(0, 100))
        self.complement_dict = None

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
            (self.R, self._recovered_partial_ending, 3),
            (self.R, self._recovered_sort, 1),
            (self.R, functools.partial(self._monotonic, variable=self.R), 2),
            (self.R, self._recovered_sort, 1),
            (None, self._post_processing, 0)
        ]

    def run(self, subset_df):
        """
        Perform complement.

        Args:
            subset_df (pandas.DataFrame): Subset of records

                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - The other columns will be ignored

        Returns:
            tuple(pandas.DataFrame, str, dict):
                pandas.DataFrame

                    Index
                        reset index
                    Columns
                        - Date(pd.Timestamp): Observation date
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases

                str: status code
                dict: status for each complement type

        Note:
            Status code will be selected from:
            - '' (not complemented)
            - 'monotonic increasing complemented confirmed data'
            - 'monotonic increasing complemented fatal data'
            - 'monotonic increasing complemented recovered data'
            - 'fully complemented recovered data'
            - 'partially complemented recovered data'
        """
        Validator(subset_df, "subset_df").dataframe(columns=[self.DATE, *self.RAW_COLS])
        # Initialize
        after_df = subset_df.copy()
        status_dict = dict.fromkeys(self.RAW_COLS, 0)
        self.complement_dict = dict.fromkeys(
            self.SHOW_COMPLEMENT_FULL_COLS, False)
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
        return (after_df, status, self.complement_dict)

    def _pre_processing(self, subset_df):
        """
        Select Confirmed/Fatal/Recovered class from the dataset.

        Args:
            subset_df (pandas.DataFrame): Subset of records
                Index
                    reset index
                Columns
                    Date, Confirmed, Fatal, Recovered (the others will be ignored)

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered
        """
        sel_invalid_R = subset_df[self.C] - \
            subset_df[self.F] < subset_df[self.R]
        subset_df.loc[sel_invalid_R,
                      self.R] = subset_df[self.C] - subset_df[self.F]
        subset_df.loc[sel_invalid_R, self.CI] = subset_df[self.C] - \
            subset_df[self.F] - subset_df[self.R]
        return subset_df.set_index(self.DATE).loc[:, self.RAW_COLS]

    def _post_processing(self, df):
        """
        Select Confirmed/Fatal/Recovered class from the dataset.

        Args:
            df (pandas.DataFrame)
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index
                    reset index
                Columns
                    Date (pandas.TimeStamp), Confirmed, Infected, Fatal, Recovered (int)
        """
        df = df.astype(np.int64)
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df = df.loc[:, self.VALUE_COLUMNS]
        return df.reset_index()

    def _monotonic(self, df, variable):
        """
        Force the variable show monotonic increasing.

        Args:
            df (pandas.DataFrame)
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered
            variable (str): variable name to show monotonic increasing

        Returns:
            pandas.DataFrame: complemented records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered
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
            series.interpolate(method="linear", inplace=True, limit_direction="both")
            series.fillna(method="ffill", inplace=True)
            # Reduce values to the previous date
            df.loc[:date, variable] = series * raw_last / series.iloc[-1]
            df[variable] = df[variable].fillna(0)
        df[variable] = df[variable].apply(np.ceil).astype(np.int64)
        self.complement_dict[f"Monotonic_{variable.lower()}"] = True
        return df

    def _validate_recovery_period(self, country_df):
        """
        Calculates and validates recovery period for specific country
        as an additional condition in order to apply full complement or not

        Args:
            country_df (pandas.DataFrame)
                Index
                    reset_index
                Columns
                    Date, Confirmed, Recovered, Fatal

        Returns:
            bool: true if recovery period is within valid range or false otherwise

        Note: Passed argument df corresponds to specific country's df
              upper_limit_days has default value of 3 months (90 days)
              lower_limit_days has default value of 1 week (7 days)
        """
        df = country_df.copy()
        # Calculate "Confirmed - Fatal"
        df["diff"] = df[self.C] - df[self.F]
        df = df.loc[:, ["diff", self.R]]
        # Calculate how many days passed to reach the number of cases
        df = df.unstack().reset_index()
        df.columns = ["Variable", "Date", "Number"]
        df["Days"] = (df[self.DATE] - df[self.DATE].min()).dt.days
        # Calculate partial recovery periods
        df = df.pivot_table(values="Days", index="Number", columns="Variable")
        df = df.interpolate(limit_area="inside").dropna().astype(np.int64)
        df["Elapsed"] = df[self.R] - df["diff"]
        df = df.loc[df["Elapsed"] > 0]
        # Check partial recovery periods
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        per_up = (df["Elapsed"] > self.upper_limit_days).sum() / len(df)
        per_lw = (df["Elapsed"] < self.lower_limit_days).sum() / len(df)
        return per_up < self.upper_percentage and per_lw < self.lower_percentage

    def _recovered_full(self, df):
        """
        Estimate the number of recovered cases with the value of recovery period.

        Args:
            df (pandas.DataFrame)
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered
        """
        # Whether complement is necessary or not
        if df[self.R].max() > self.max_ignored:
            # Necessary if sum of recovered is more than 99%
            # or less than 1% of sum of recovered minus infected when out-breaking
            sel_C1 = df[self.C] > self.max_ignored
            sel_R1 = df[self.R] > self.max_ignored
            sel_2 = df[self.C].diff().diff().rolling(14).mean() > 0
            cf_diff = (df[self.C] - df[self.F]).rolling(14).mean()
            sel_3 = df[self.R] < 0.01 * cf_diff
            s_df = df.loc[sel_C1 & sel_R1 & sel_2 & sel_3]
            if s_df.empty and self._validate_recovery_period(df):
                return df
        # Estimate recovered records
        df[self.R] = (df[self.C] - df[self.F]).shift(
            periods=self.recovery_period, freq="D")
        self.complement_dict[self.FULL_RECOVERED] = True
        return df

    def _recovered_partial_ending(self, df):
        """
        If ending recovered values do not change for more than applied 'self.max_ending_unupdated' days
        after reached 'self.max_ignored' cases, apply either previous diff() (with some variance)
        or full complement only to these ending unupdated values and keep the previous valid ones.
        _recovered_partial() does not handle well such values, because
        they are not in-between values and interpolation generates only similar values,
        but small compared to confirmed cases.

        Args:
            df (pandas.DataFrame)
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered

        Note: _recovered_partial_ending() must always be called
              after _recovered_partial()
        """
        # Whether complement is necessary or not
        r_max = df[self.R].max()
        sel_0 = (df[self.R] == r_max).sum() > self.max_ending_unupdated
        if not (r_max and sel_0):
            # full complement will be handled in _recovered_full()
            return df
        # Complement any ending unupdated values that are not updated
        # for more than max_ending_unupdated days,
        # by keeping and propagating forward previous valid diff()
        # min_index: index for first ending max R reoccurrence
        min_index = df[self.R].idxmax() + timedelta(days=1)
        first_value = df.loc[min_index, self.R]
        df_ending = df.copy()
        df_ending.loc[df_ending.duplicated(
            [self.R], keep="first"), self.R] = None
        diff_series = df_ending[self.R].diff(
        ).ffill().fillna(0).astype(np.int64)
        diff_series.loc[diff_series.duplicated(keep="last")] = None
        diff_series.interpolate(
            method="linear", inplace=True, limit_direction="both")
        df.loc[min_index:, self.R] = first_value + \
            diff_series[min_index:].cumsum()
        # Check if the ending complement is valid (too large recovered ending values)
        # If the validity check fails, then fully complement these ending values
        sel_C1 = df[self.C] > self.max_ignored
        sel_R1 = df[self.R] > self.max_ignored
        # check all values one-by-one, no rolling window
        cf_diff = df[self.C] - df[self.F]
        sel_limit = df[self.R] > 0.99 * cf_diff
        s_df_1 = df.loc[sel_C1 & sel_R1 & sel_limit]
        if not s_df_1.empty:
            df.loc[min_index:, self.R] = (df[self.C] - df[self.F]).shift(
                periods=self.recovery_period, freq="D").loc[min_index:]
        self.complement_dict[self.PARTIAL_RECOVERED] = True
        return df

    def _recovered_partial(self, df):
        """
        If recovered values do not change for more than applied 'self.interval' days
        after reached 'self.max_ignored' cases, interpolate the recovered values.

        Args:
            df (pandas.DataFrame)
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered
        """
        # Whether complement is necessary or not
        series = df.loc[df[self.R] > self.max_ignored, self.R]
        max_frequency = series.value_counts().max()
        if max_frequency <= self.interval:
            return df
        # Complement in-between recovered values when confirmed > max_ignored
        sel_C = df[self.C] > self.max_ignored
        sel_duplicate = df.duplicated([self.R], keep="first")
        df.loc[sel_C & sel_duplicate, self.R] = None
        df[self.R].interpolate(
            method="linear", inplace=True, limit_direction="both")
        df[self.R] = df[self.R].fillna(method="bfill")
        self.complement_dict[self.PARTIAL_RECOVERED] = True
        return df

    def _recovered_sort(self, df):
        """
        Sort the absolute values of recovered data.

        Args:
            df (pandas.DataFrame)
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered

        Returns:
            pandas.DataFrame: complemented records
                Index
                    Date (pandas.TimeStamp)
                Columns
                    Confirmed, Fatal, Recovered

        Note:
            _recovered_sort() must always be called after _recovered_partial_ending()
        """
        df.loc[:, self.R] = sorted(df[self.R].abs())
        df[self.R].interpolate(method="time", inplace=True)
        df[self.R] = df[self.R].fillna(0).round()
        return df
