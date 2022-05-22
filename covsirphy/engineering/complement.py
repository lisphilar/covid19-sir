#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import warnings
import numpy as np
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _ComplementHandler(Term):
    """Class for data complement.

    Args:
        data (pandas.DataFrame): raw data
            Index
                reset index
            Column
                - column defined by @date
                - the other columns
        date (str): column name of observation dates of the data

    Note:
        Layers should be removed in advance.
    """

    def __init__(self, data, date):
        self._df = data.set_index(date).resample("D").ffill()
        self._date = date

    def all(self):
        """Return all available data.

        Returns:
            pandas.DataFrame: transformed data
        """
        return self._df.reset_index()

    def assess_monotonic_increase(self, column):
        """Assess the column shows a monotonic increase.

        Args:
            column (str): column name

        Return:
            bool: whether complement is required or not
        """
        Validator(self._df, "data").dataframe(columns=[column])
        return not self._df[column].drop_duplicates().is_monotonic_increasing

    def force_monotonic_increase(self, column):
        """Force the column shows monotonic increase.

        Args:
            column (str): column name

        Returns:
            bool: whether complemented or not
        """
        warnings.simplefilter("ignore", UserWarning)
        if not self.assess_monotonic_increase(column):
            return False
        df = self._df.copy()
        decreased_dates = df[df[column].diff() < 0].index.tolist()
        for date in decreased_dates:
            # Raw value on the decreased date
            raw_last = df.loc[date, column]
            # Extrapolated value on the date
            series = df.loc[:date, column]
            series.iloc[-1] = None
            series = series.ffill()
            # The last value must be equal to or lower than the last value of the raw data
            df.loc[:date, column] = np.ceil(series * raw_last / series.iloc[-1])
        self._df = df.copy()
        return True

    def assess_recovered_full(self, confirmed, fatal, recovered, **kwargs):
        """Assess the recovered curve to decide whether it is acceptable or not.

        Args:
            confirmed (str): column of the number of confirmed cases
            fatal (str): column of the number of fatal cases
            recovered (str): column name of the number of recovered cases
            kwargs: Keyword arguments of the following (all required)
                max_ignored (int): max number of confirmed cases to be ignored [cases]
                upper_limit_days (int): maximum number of valid partial recovery periods [days]
                lower_limit_days (int): minimum number of valid partial recovery periods [days]
                upper_percentage (float): fraction of partial recovery periods with value greater than upper_limit_days
                lower_percentage (float): fraction of partial recovery periods with value less than lower_limit_days

        Return:
            bool: whether complement is required or not
        """
        # Keyword arguments
        max_ignored = self._ensure_natural_int(kwargs["max_ignored"], name="max_ignored")
        upper_limit_days = self._ensure_natural_int(kwargs["upper_limit_days"], name="upper_limit_days")
        lower_limit_days = self._ensure_natural_int(kwargs["lower_limit_days"], name="lower_limit_days")
        upper_percentage = self._ensure_float(kwargs["upper_percentage"], name="upper_percentage")
        lower_percentage = self._ensure_float(kwargs["lower_percentage"], name="lower_percentage")
        # Complement is required if the max value of Recovered is small
        df = self._df.copy()
        if 0 < df[recovered].max() <= max_ignored:
            return True
        # Complement is required if sum of recovered is more than 99%
        # or less than 1% of sum of recovered minus infected when out-breaking
        sel_C1 = df[confirmed] > max_ignored
        sel_R1 = df[recovered] > max_ignored
        sel_2 = df[confirmed].diff().diff().rolling(14).mean() > 0
        sel_3 = df[recovered] < 0.01 * (df[confirmed] - df[fatal]).rolling(14).mean()
        if not df.loc[sel_C1 & sel_R1 & sel_2 & sel_3].empty:
            return True
        # Complement is required if recovered period of the data is not acceptable
        df["diff"] = df[confirmed] - df[fatal]
        df = df.loc[:, ["diff", recovered]]
        df = df.unstack().reset_index()
        df.columns = ["Variable", "Date", "Number"]
        df["Days"] = (df["Date"] - df["Date"].min()).dt.days
        df = df.pivot_table(values="Days", index="Number", columns="Variable")
        df = df.interpolate(limit_area="inside").dropna().astype(np.int64)
        df["Elapsed"] = df[recovered] - df["diff"]
        df = df.loc[df["Elapsed"] > 0]
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        per_up = (df["Elapsed"] > upper_limit_days).sum() / len(df)
        per_lw = (df["Elapsed"] < lower_limit_days).sum() / len(df)
        return per_up >= upper_percentage or per_lw >= lower_percentage

    def force_recovered_full(self, confirmed, fatal, recovered, **kwargs):
        """Force full complement of recovered data.

        Args:
            confirmed (str): column of the number of confirmed cases
            fatal (str): column of the number of fatal cases
            recovered (str): column name of the number of recovered cases
            kwargs: Keyword arguments of the following (all required)
                max_ignored (int): max number of confirmed cases to be ignored [cases]
                recovery_period (int): expected value of recovery period [days]
                upper_limit_days (int): maximum number of valid partial recovery periods [days]
                lower_limit_days (int): minimum number of valid partial recovery periods [days]
                upper_percentage (float): fraction of partial recovery periods with value greater than upper_limit_days
                lower_percentage (float): fraction of partial recovery periods with value less than lower_limit_days

        Returns:
            bool: whether complemented or not
        """
        if not self.assess_recovered_full(confirmed, recovered, **kwargs):
            return False
        df = self._df.copy()
        recovery_period = self._ensure_natural_int(kwargs["recovery_period"], name="recovery_period")
        self._df[recovered] = (df[confirmed] - df[fatal]).shift(periods=recovery_period, freq="D")
        return True

    def assess_partial_internal(self, column, **kwargs):
        """Assess the data includes un-changed values internally.


        Args:
            column (str): column name of the number of recovered cases or the number of tests
            kwargs: Keyword arguments of the following (all required)
                interval (int): expected update interval of the number of recovered cases [days]
                max_ignored (int): max number of confirmed cases to be ignored [cases]

        Return:
            bool: whether complement is required or not
        """
        # Keyword arguments
        max_ignored = self._ensure_natural_int(kwargs["max_ignored"], name="max_ignored")
        interval = self._ensure_natural_int(kwargs["interval"], name="interval")
        # Whether complement is necessary or not
        df = self._df.copy()
        series = df.loc[df[column] > max_ignored, column]
        max_frequency = series.value_counts().max()
        return max_frequency > interval

    def force_partial_internal(self, confirmed, column, **kwargs):
        """Force partial complement of the data (internal).
        If recovered values do not change for more than applied 'interval' days
        after reached 'max_ignored' cases, interpolate the recovered values.

        Args:
            confirmed (str): column of the number of confirmed cases
            column (str): column name of the number of recovered cases or the number of tests
            kwargs: Keyword arguments of the following (all required)
                interval (int): expected update interval of the number of recovered cases [days]
                max_ignored (int): max number of confirmed cases to be ignored [cases]

        Returns:
            bool: whether complemented or not
        """
        if not self.assess_partial_internal(column, **kwargs):
            return False
        # Keyword arguments
        max_ignored = self._ensure_natural_int(kwargs["max_ignored"], name="max_ignored")
        # Complement
        df = self._df.copy()
        df.loc[(df[confirmed] > max_ignored) & df.duplicated(column), column] = None
        df[column] = np.ceil(df[column].interpolate(method="linear", limit_direction="both").bfill())
        self._df = df.copy()
        return True

    def assess_partial_ending(self, column, **kwargs):
        """Assess the data includes un-changed values in the last values.


        Args:
            column (str): column name of the number of recovered cases or the number of tests
            kwargs: Keyword arguments of the following (all required)
                max_ending_unupdated (int): Max number of days to apply full complement, where max recovered cases are not updated [days]

        Return:
            bool: whether complement is required or not
        """
        # Keyword arguments
        max_ending_unupdated = self._ensure_natural_int(kwargs["max_ending_unupdated"], name="max_ending_unupdated")
        # Whether complement is necessary or not
        df = self._df.copy()
        r_max = df[column].max()
        sel_0 = (df[column] == r_max).sum() > max_ending_unupdated
        return r_max and sel_0

    def force_recovered_partial_ending(self, confirmed, fatal, recovered, **kwargs):
        """Force partial complement of recovered data (ending).
        If ending recovered values do not change for more than applied 'max_ending_unupdated' days
        after reached 'max_ignored' cases, apply either previous diff() (with some variance)
        or full complement only to these ending unupdated values and keep the previous valid ones.

        Args:
            confirmed (str): column of the number of confirmed cases
            fatal (str): column of the number of fatal cases
            recovered (str): column name of the number of recovered cases
            kwargs: Keyword arguments of the following (all required)
                max_ignored (int): max number of confirmed cases to be ignored [cases]
                recovery_period (int): expected value of recovery period [days]

        Returns:
            bool: whether complemented or not
        """
        if not self.assess_recovered_partial_ending(recovered, **kwargs):
            return False
        # Keyword arguments
        max_ignored = self._ensure_natural_int(kwargs["max_ignored"], name="max_ignored")
        recovery_period = self._ensure_natural_int(kwargs["recovery_period"], name="recovery_period")
        # Complement
        df = self._df.copy()
        # Keeping and propagating forward previous valid diff()
        # min_index: index for first ending max R reoccurrence
        min_index = df[recovered].idxmax() + timedelta(days=1)
        first_value = df.loc[min_index, recovered]
        df_ending = df.copy()
        df_ending.loc[df_ending.duplicated([recovered], keep="first"), recovered] = None
        diff_series = df_ending[recovered].diff().ffill().fillna(0).astype("Int64")
        diff_series.loc[diff_series.duplicated(keep="last")] = None
        diff_series.interpolate(method="linear", inplace=True, limit_direction="both")
        df.loc[min_index:, recovered] = first_value + diff_series[min_index:].cumsum()
        # Check if the ending complement is valid (too large recovered ending values)
        # If the validity check fails, then fully complement these ending values
        sel_C1 = df[confirmed] > max_ignored
        sel_R1 = df[recovered] > max_ignored
        # check all values one-by-one, no rolling window
        sel_limit = df[recovered] > 0.99 * (df[confirmed] - df[fatal])
        s_df_1 = df.loc[sel_C1 & sel_R1 & sel_limit]
        if not s_df_1.empty:
            df.loc[min_index:, recovered] = (
                df[confirmed] - df[fatal]).shift(periods=recovery_period, freq="D").loc[min_index:]
        # Sorting
        df.loc[:, recovered] = sorted(df[self.R].abs())
        df[recovered] = np.ceil(df[recovered].interpolate(method="time").fillna(0))
        self._df = df.copy()
        return True

    def force_tests_partial_ending(self, tests, **kwargs):
        """Force partial complement of tests data (ending).
        If ending recovered values do not change for more than applied 'max_ending_unupdated' days
        after reached 'max_ignored' cases, apply either previous diff() (with some variance)
        or full complement only to these ending unupdated values and keep the previous valid ones.

        Args:
            tests (str): column of the number of tests
            kwargs: Keyword arguments of the following (all required)
                max_ignored (int): max number of confirmed cases to be ignored [cases]

        Returns:
            bool: whether complemented or not
        """
        if not self.assess_partial_ending(tests, **kwargs):
            return False
        # Keeping and propagating forward previous valid diff()
        # min_index: index for first ending max test reoccurrence
        df = self._df.copy()
        min_index = df[tests].idxmax() + 1
        first_value = df.loc[min_index, tests]
        df_ending = df.copy()
        df_ending.loc[df_ending.duplicated(tests, keep="first"), tests] = None
        diff_series = df_ending[tests].diff().ffill().fillna(0).astype("Int64")
        diff_series.loc[diff_series.duplicated(keep="last")] = None
        diff_series.interpolate(method="linear", inplace=True, limit_direction="both")
        df.loc[min_index:, tests] = np.ceil(first_value + diff_series.loc[min_index:].cumsum())
        self._df = df.copy()
        return True

    @staticmethod
    def _ensure_natural_int(target, name="number", include_zero=False, none_ok=False):
        """
        Ensure a natural (non-negative) number.

        Args:
            target (int or float or str or None): value to ensure
            name (str): argument name of the value
            include_zero (bool): include 0 or not
            none_ok (bool): None value can be applied or not.

        Returns:
            int: as-is the target

        Note:
            When @target is None and @none_ok is True, None will be returned.
            If the value is a natural number and the type was float or string,
            it will be converted to an integer.
        """
        if target is None and none_ok:
            return None
        s = f"@{name} must be a natural number, but {target} was applied"
        try:
            number = int(target)
        except TypeError as e:
            raise TypeError(f"{s} and not converted to integer.") from e
        if number != target:
            raise ValueError(f"{s}. |{target} - {number}| > 0")
        min_value = 0 if include_zero else 1
        if number < min_value:
            raise ValueError(f"{s}. This value is under {min_value}")
        return number

    @staticmethod
    def _ensure_float(target, name="value", value_range=(0, None)):
        """
        Ensure a float value.
        If the value is a float value and the type was string,
        it will be converted to a float.

        Args:
            target (float or str): value to ensure
            name (str): argument name of the value
            value_range(tuple(int or None, int or None)): value range, None means un-specified

        Returns:
            float: as-is the target
        """
        s = f"@{name} must be a float value, but {target} was applied"
        try:
            value = float(target)
        except ValueError:
            raise ValueError(f"{s} and not converted to float.") from None
        # Minimum
        if value_range[0] is not None and value < value_range[0]:
            raise ValueError(f"{name} must be over or equal to {value_range[0]}, but {value} was applied.")
        # Maximum
        if value_range[1] is not None and value > value_range[1]:
            raise ValueError(f"{name} must be under or equal to {value_range[1]}, but {value} was applied.")
        return value
