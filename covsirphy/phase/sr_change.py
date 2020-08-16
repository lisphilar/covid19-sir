#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import numpy as np
import pandas as pd
import ruptures as rpt
from covsirphy.cleaning.term import Term
from covsirphy.phase.trend import Trend


class ChangeFinder(Term):
    """
    Find change points of S-R trend.

    Args:
        sr_df (pandas.DataFrame)
            Index:
                Date (pd.TimeStamp): Observation date
            Columns:
                - Recovered (int): the number of recovered cases (> 0)
                - Susceptible (int): the number of susceptible cases
                - any other columns will be ignored
        min_size (int): minimum value of phase length [days], over 2
        max_rmsle (float): minmum value of RMSLE score

    Notes:
        When RMSLE score > max_rmsle, predicted values will be None
    """

    def __init__(self, sr_df, min_size=7, max_rmsle=20.0):
        # Dataset
        self.sr_df = self.ensure_dataframe(
            sr_df, name="sr_df", time_index=True, columns=[self.S, self.R])
        self.dates = [
            date_obj.strftime(self.DATE_FORMAT) for date_obj in sr_df.index
        ]
        # Minimum size of records
        self.min_size = self.ensure_natural_int(min_size, "min_size")
        # Check length of records
        if self.min_size < 3:
            raise ValueError(
                f"@min_size must be over 2, but {min_size} was applied.")
        if len(self.dates) < self.min_size * 2:
            raise ValueError(
                f"More than {min_size * 2} records must be included.")
        # Minimum value of RMSLE score
        self.max_rmsle = self.ensure_float(max_rmsle)
        # Setting for optimization
        self._change_dates = []

    def run(self):
        """
        Run optimization and find change points.

        Returns:
            self
        """
        # Convert the dataset, index: Recovered, column: log10(Susceptible)
        sr_df = self.sr_df.copy()
        sr_df[self.S] = np.log10(sr_df[self.S].astype(np.float64))
        df = sr_df.pivot_table(index=self.R, values=self.S, aggfunc="last")
        # Convert index to serial numbers
        serial_df = pd.DataFrame(np.arange(1, df.index.max() + 1, 1))
        serial_df.index += 1
        df = pd.merge(
            df, serial_df, left_index=True, right_index=True, how="outer"
        )
        series = df.reset_index(drop=True).iloc[:, 0]
        series = series.interpolate(limit_direction="both")
        # Sampling to reduce run-time of Ruptures
        samples = np.linspace(
            0, series.index.max(), len(self.sr_df), dtype=np.int64
        )
        series = series[samples]
        # Detection with Ruptures
        algorithm = rpt.Pelt(model="rbf", jump=2, min_size=self.min_size)
        results = algorithm.fit_predict(series.values, pen=0.5)
        # Convert index values to Susceptible values
        reset_series = series.reset_index(drop=True)
        reset_series.index += 1
        susceptible_df = reset_series[results].reset_index()
        # Convert Susceptible values to dates
        df = pd.merge_asof(
            susceptible_df.sort_values(self.S),
            sr_df.reset_index().sort_values(self.S),
            on=self.S, direction="nearest"
        )
        found_list = df[self.DATE].sort_values()[:-1]
        # Only use dates when the previous phase has more than {min_size + 1} days
        delta_days = timedelta(days=self.min_size)
        first_obj = self.date_obj(self.dates[0])
        last_obj = self.date_obj(self.dates[-1])
        effective_list = [first_obj]
        for found in found_list:
            if effective_list[-1] + delta_days < found:
                effective_list.append(found)
        # The last change date must be under the last date of records {- min_size} days
        if effective_list[-1] >= last_obj - delta_days:
            effective_list = effective_list[:-1]
        # Set change points
        self._change_dates = [
            date.strftime(self.DATE_FORMAT) for date in effective_list[1:]
        ]
        return self

    def _curve_fitting(self, phase, start_date, end_date):
        """
        Perform curve fitting for the phase.

        Args:
            phase (str): phase name
            start_date (str): start date of the phase
            end_date (str): end date of the phase

        Returns:
            tuple
                (pandas.DataFrame): Result of curve fitting
                    Index: reset index
                    Columns:
                        - (phase name)_predicted: predicted value of Susceptible
                        - (phase_name)_actual: actual value of Susceptible
                        - (phase_name)_Recovered: Recovered
                (int): minimum value of R, which is the change point of the curve
        """
        sr_df = self.sr_df.copy()
        sta = self.date_obj(start_date)
        end = self.date_obj(end_date)
        sr_df = sr_df.loc[(sr_df.index >= sta) & (sr_df.index <= end), :]
        trend = Trend(sr_df)
        df = trend.run()
        if trend.rmsle() > self.max_rmsle:
            df[f"{self.S}{self.P}"] = None
        # Get min value for vline
        r_value = int(df[self.R].min())
        # Rename the columns
        phase = self.INITIAL if phase == "0th" else phase
        df = df.rename({f"{self.S}{self.P}": f"{phase}{self.P}"}, axis=1)
        df = df.rename({f"{self.S}{self.A}": f"{phase}{self.A}"}, axis=1)
        df = df.rename({f"{self.R}": f"{phase}_{self.R}"}, axis=1)
        return (df, r_value)

    def date_range(self, change_dates=None):
        """
        Calculate the list of start/end dates with the list of change dates.

        Args:
            change_dates (list[str] or None): list of change points

        Returns:
            tuple(list[str], list[str]): list of start/end dates

        Notes:
            @change_dates must be specified if ChangeFinder.run() was not done.
        """
        change_dates = change_dates or self._change_dates[:]
        self._change_dates = change_dates[:]
        if not change_dates:
            return([self.dates[0]], [self.dates[-1]])
        # Start dates
        start_dates = [self.dates[0], *change_dates]
        # End dates
        end_dates = [self.yesterday(date) for date in change_dates]
        end_dates.append(self.dates[-1])
        return (start_dates, end_dates)

    def show(self, area, change_dates=None, filename=None):
        """
        show the S-R trend in a figure.

        Args:
            area (str): area name
            change_dates (list[str] or None): list of change points
            filename (str): filename of the figure, or None (display)

        Notes:
            @change_dates must be specified if ChangeFinder.run() was not done.
        """
        # Curve fitting
        start_dates, end_dates = self.date_range(change_dates)
        nested = [
            self._curve_fitting(self.num2str(num), start_date, end_date)
            for (num, (start_date, end_date))
            in enumerate(zip(start_dates, end_dates))
        ]
        df_list, vlines = zip(*nested)
        comp_df = pd.concat([self.sr_df, *df_list], axis=1)
        comp_df = comp_df.rename({self.S: f"{self.S}{self.A}"}, axis=1)
        comp_df = comp_df.apply(
            lambda x: pd.to_numeric(x, errors="coerce", downcast="integer"),
            axis=0
        )
        # Show figure
        pred_cols = [
            col for col in comp_df.columns if col.endswith(self.P)
        ]
        if len(pred_cols) == 1:
            title = f"{area}: S-R trend without change points"
        else:
            _list = self._change_dates[:]
            strings = [
                ", ".join(_list[i: i + 6]) for i in range(0, len(_list), 6)
            ]
            change_str = ",\n".join(strings)
            title = f"{area}: S-R trend changed on\n{change_str}"
        Trend.show_with_many(
            result_df=comp_df,
            predicted_cols=pred_cols,
            title=title,
            vlines=vlines[1:],
            filename=filename
        )
