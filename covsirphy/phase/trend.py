#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from covsirphy.util.plotting import line_plot_multiple
from covsirphy.util.term import Term


class Trend(Term):
    """
    S-R trend analysis in a phase.

    Args:
        sr_df (pandas.DataFrame)
            Index
                Date (pd.TimeStamp): Observation date
            Columns
                - Recovered (int): the number of recovered cases (> 0)
                - Susceptible (int): the number of susceptible cases
                - any other columns will be ignored
    """
    L = "linear"
    N = "negative_exponential"

    def __init__(self, sr_df):
        # Dataset
        self.sr_df = self._ensure_dataframe(
            sr_df, name="sr_df", time_index=True, columns=[self.S, self.R])
        # Dataset for analysis
        if len(self.sr_df) < 3:
            raise ValueError("The length of @sr_df must be over 2.")
        # Setting for analysis
        self._result_df = pd.DataFrame()
        self.fit_fnc = self.linear

    @property
    def result_df(self):
        """
        pandas.DataFrame: results of fitting

            Index
                - Date (pandas.TimeStamp): Observation date
            Columns
                - Recovered: The number of recovered cases
                - Susceptible_actual: Actual values of Susceptible
                - columns defined by @columns
        """
        if self._result_df.empty:
            self.run()
        return self._result_df

    def run(self):
        """
        Perform curve fitting with some functions and select the best solution.
        Then, the result and RMSLE score of the best solution will be saved.

        Returns:
            covsirphy.Trend: self
        """
        L, N = self.L, self.N
        # Perform fitting and calculate RMSLE scores
        dataframe_dict = {func: self._run(func=func) for func in (L, N)}
        score_dict = {
            func: self._rmsle(fit_df) for (func, fit_df) in dataframe_dict.items()}
        # Select the best dataframe
        if 0 < score_dict[L] < score_dict[N] or not score_dict[N]:
            self._result_df = dataframe_dict[L]
        else:
            self._result_df = dataframe_dict[N]
        return self

    def _run(self, func):
        """
        Perform curve fitting of S-R trend with linear or negative exponential function and save the result.

        Args:
            func (function): the selected curve fitting function, either linear or negative exponential

        Returns:
            pandas.DataFrame: results of fitting
                Index
                    - index (Date) (pd.TimeStamp): Observation date
                Columns
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual values of Susceptible
                    - columns defined by @columns
        """
        self.fit_fnc = self.negative_exp if func == self.N else self.linear
        return self._fitting(self.sr_df)

    def _fitting(self, sr_df):
        """
        Perform curve fitting of S-R trend with linear or negative exponential function.

        Args:
            sr_df (pandas.DataFrame): training dataset
                Index
                    - index (Date) (pd.TimeStamp): Observation date
                Columns
                    - Recovered: The number of recovered cases
                    - Susceptible: Actual data of Susceptible

        Returns:
            pandas.DataFrame
                Index
                    - index (Date) (pd.TimeStamp): Observation date
                Columns
                    - Recovered (int): The number of recovered cases
                    - Susceptible_actual (int): Actual values of Susceptible
                    - Susceptible_predicted (int): Predicted values of Susceptible
        """
        df = sr_df.rename({self.S: f"{self.S}{self.A}"}, axis=1)
        df = df.astype(np.float64)
        # Calculate initial values of parameters
        x_series = df[self.R]
        y_series = np.log(df[f"{self.S}{self.A}"]).astype(np.float64)
        a_ini = y_series.max()
        b_ini = y_series.diff().reset_index(drop=True)[1] / a_ini
        # Curve fitting with linear or negative exponential function
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        param, _ = curve_fit(
            self.fit_fnc, x_series, y_series,
            p0=[a_ini, b_ini],
            # Increase mux number of iteration in curve fitting from 600 (default)
            maxfev=10000
        )
        # Predict the values with the parameters
        f_partial = functools.partial(
            self.fit_fnc, a=param[0], b=param[1]
        )
        df[f"{self.S}{self.P}"] = np.exp(
            f_partial(x_series)).astype(np.float64)
        return df.astype(np.int64, errors="ignore")

    def rmsle(self):
        """
        Return the best RMSLE score.

        Returns:
            float: RMSLE score
        """
        if self._result_df.empty:
            self.run()
        return self._rmsle(self._result_df)

    def _rmsle(self, fit_df):
        """
        Calculate RMSLE score of actual/predicted Susceptible.

        Args:
            fit_df (pandas.DataFrame):
                Index
                    - index (Date) (pd.TimeStamp): Observation date
                Columns
                    - Recovered (int): The number of recovered cases
                    - Susceptible_actual (int): Actual values of Susceptible
                    - Susceptible_predicted (int): Predicted values of Susceptible

        Returns:
            float: RMSLE score
        """
        df = fit_df.replace(np.inf, 0)
        df = df.loc[df[f"{self.S}{self.A}"] > 0, :]
        df = df.loc[df[f"{self.S}{self.P}"] > 0, :]
        actual = df[f"{self.S}{self.A}"]
        predicted = df[f"{self.S}{self.P}"]
        # Calculate RMSLE score
        scores = np.abs(
            np.log10(actual + 1) - np.log10(predicted + 1)
        )
        return scores.sum()

    def show(self, area, **kwargs):
        """
        show the result as a figure.

        Args:
            area (str): area name
            kwargs: keyword arguments of covsirphy.line_plot_multiple()
        """
        df = self._result_df.copy()
        df = df.rename({f"{self.S}{self.P}": "Predicted"}, axis=1)
        # Star/end date
        start_date = self.sr_df.index.min().strftime(self.DATE_FORMAT)
        end_date = self.sr_df.index.max().strftime(self.DATE_FORMAT)
        # Plotting
        title = f"{area}: S-R trend from {start_date} to {end_date}"
        self.show_with_many(
            result_df=df, predicted_cols=["Predicted"], title=title, **kwargs)

    @classmethod
    def show_with_many(cls, result_df, predicted_cols, title, **kwargs):
        """
        show the result as a figure.

        Args:
            result_df (pandas.DataFrame): results of fitting

                Index
                    Date (pandas.TimeStamp): Observation date
                Columns
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual values of Susceptible
                    - columns defined by @predicted_cols
            predicted_cols (list[str]): list of columns which have predicted values
            title (str): title of the figure
            kwargs: keyword arguments of covsirphy.line_plot_multiple()
        """
        result_df = cls._ensure_dataframe(
            result_df, name="result_df", time_index=True,
            columns=[cls.R, f"{cls.S}{cls.A}", *predicted_cols]
        )
        result_df.rename(columns={f"{cls.S}{cls.A}": cls.ACTUAL}, inplace=True)
        result_df.rename(columns=lambda x: x.replace(cls.P, ""), inplace=True)
        predicted_cols = [
            col.replace(cls.P, "") for col in predicted_cols]
        # Line plotting
        line_plot_multiple(
            df=result_df.replace(np.inf, np.nan),
            x_col=cls.R, actual_col=cls.ACTUAL, predicted_cols=predicted_cols,
            title=title, ylabel=cls.S, y_logscale=True, **kwargs)
