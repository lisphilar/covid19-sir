#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import sys
import warnings
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from covsirphy.cleaning.term import Term


class Trend(Term):
    """
    S-R trend analysis in a phase.

    Args:
        sr_df (pandas.DataFrame)
            Index:
                Date (pd.TimeStamp): Observation date
            Columns:
                - Recovered (int): the number of recovered cases (> 0)
                - Susceptible (int): the number of susceptible cases
                - any other columns will be ignored
    """

    def __init__(self, sr_df):
        # Dataset
        self.sr_df = self.ensure_dataframe(
            sr_df, name="sr_df", time_index=True, columns=[self.S, self.R])
        # Dataset for analysis
        if len(self.sr_df) < 3:
            raise ValueError("The length of @sr_df must be over 2.")
        # Setting for analysis
        self.result_df = None

    def run(self):
        """
        Perform curve fitting of S-R trend with negative exponential function and save the result.

        Returns:
            (pandas.DataFrame): results of fitting
                Index:
                    - index (Date) (pd.TimeStamp): Observation date
                Columns:
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual values of Susceptible
                    - columns defined by @columns
        """
        self.result_df = self._fitting(self.sr_df)
        return self.result_df

    def _fitting(self, sr_df):
        """
        Perform curve fitting of S-R trend
            with negative exponential function.

        Args:
            sr_df (pandas.DataFrame): training dataset
                Index:
                    - index (Date) (pd.TimeStamp): Observation date
                Columns:
                    - Recovered: The number of recovered cases
                    - Susceptible: Actual data of Susceptible

        Returns:
            (pandas.DataFrame)
                Index:
                    - index (Date) (pd.TimeStamp): Observation date
                Columns:
                    - Recovered (int): The number of recovered cases
                    - Susceptible_actual (int): Actual values of Susceptible
                    - Susceptible_predicted (int): Predicted values of Susceptible
        """
        df = sr_df.rename({self.S: f"{self.S}{self.A}"}, axis=1)
        df = df.astype(np.float64)
        # Calculate initial values of parameters
        x_series = df[self.R]
        y_series = df[f"{self.S}{self.A}"]
        a_ini = y_series.max()
        b_ini = y_series.diff().reset_index(drop=True)[1] / a_ini
        # Curve fitting with negative exponential function
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        param, _ = curve_fit(
            self.negative_exp, x_series, y_series,
            p0=[a_ini, b_ini]
        )
        # Predict the values with the parameters
        f_partial = functools.partial(
            self.negative_exp, a=param[0], b=param[1]
        )
        df[f"{self.S}{self.P}"] = f_partial(x_series)
        return df.astype(np.int64, errors="ignore")

    def rmsle(self):
        """
        Calculate RMSLE score of actual/predicted Susceptible.

        Returns:
            (float): RMSLE score
        """
        df = self.run().replace(np.inf, 0)
        df = df.loc[df[f"{self.S}{self.A}"] > 0, :]
        df = df.loc[df[f"{self.S}{self.P}"] > 0, :]
        actual = df[f"{self.S}{self.A}"]
        predicted = df[f"{self.S}{self.P}"]
        # Calculate RMSLE score
        scores = np.abs(
            np.log10(actual + 1) - np.log10(predicted + 1)
        )
        return scores.sum()

    def show(self, area, filename=None):
        """
        show the result as a figure.

        Args:
            area (str): area name
            filename (str): filename of the figure, or None (display)
        """
        df = self.run()
        df = df.rename({f"{self.S}{self.P}": "Predicted"}, axis=1)
        start_date = self.sr_df.index.min().strftime(self.DATE_FORMAT)
        end_date = self.sr_df.index.max().strftime(self.DATE_FORMAT)
        title = f"{area}: S-R trend from {start_date} to {end_date}"
        self.show_with_many(
            result_df=df,
            predicted_cols=["Predicted"],
            title=title,
            filename=filename
        )

    @classmethod
    def show_with_many(cls, result_df, predicted_cols,
                       title, vlines=None, filename=None):
        """
        show the result as a figure.

        Args:
            result_df (pandas.DataFrame): results of fitting
                Index:
                    - index (Date) (pd.TimeStamp): Observation date
                Columns:
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual values of Susceptible
                    - columns defined by @predicted_cols
            predicted_cols (list[str]): list of columns which have predicted values
            title (str): title of the figure
            vlines (list[int]): list of Recovered values to show vertical lines
            filename (str): filename of the figure, or None (show figure)
        """
        result_df = cls.ensure_dataframe(
            result_df, name="result_df", time_index=True,
            columns=[cls.R, f"{cls.S}{cls.A}", *predicted_cols]
        )
        df = result_df.replace(np.inf, np.nan)
        x_series = df[cls.R]
        actual = df[f"{cls.S}{cls.A}"]
        # Plot the actual values
        plt.plot(
            x_series, actual,
            label="Actual", color="black",
            marker=".", markeredgewidth=0, linewidth=0
        )
        # Plot the predicted values
        for col in predicted_cols:
            plt.plot(x_series, df[col], label=col.replace(cls.P, str()))
        # x-axis
        plt.xlabel(cls.R)
        plt.xlim(0, None)
        # y-axis
        plt.ylabel(cls.S)
        try:
            plt.yscale("log", base=10)
        except Exception:
            plt.yscale("log", basey=10)
        # Delete y-labels of log-scale (minor) axis
        plt.setp(plt.gca().get_yticklabels(minor=True), visible=False)
        plt.gca().tick_params(left=False, which="minor")
        # Set new y-labels of major axis
        ymin, ymax = plt.ylim()
        ydiff_scale = int(np.log10(ymax - ymin))
        yticks = np.linspace(
            round(ymin, - ydiff_scale),
            round(ymax, - ydiff_scale),
            5,
            dtype=np.int64
        )
        plt.gca().set_yticks(yticks)
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        plt.gca().yaxis.set_major_formatter(fmt)
        # Title
        plt.title(title)
        # Vertical lines
        if isinstance(vlines, (list, tuple)):
            for value in vlines:
                plt.axvline(x=value, color="black", linestyle=":")
        # Legend
        plt.legend(
            bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0
        )
        # Save figure or show figure
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
        if filename is None:
            plt.show()
            return None
        plt.savefig(
            filename, bbox_inches="tight", transparent=False, dpi=300
        )
        plt.clf()
        return None
