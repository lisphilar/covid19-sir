#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import functools
import sys
import warnings
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from covsirphy.cleaning.word import Word
from covsirphy.phase.sr_data import SRData


class Trend(Word):
    """
    S-R trend analysis in a phase.

    Args:
        clean_df (pandas.DataFrame): cleaned data

            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Country (str): country/region name
                - Province (str): province/prefecture/sstate name
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
        population (int): total population in the place
        country (str): country name
        province (str): province name
        start_date (str): start date, like 22Jan2020
        end_date (str): end date, like 01Feb2020
    """

    def __init__(self, clean_df, population,
                 country, province=None, start_date=None, end_date=None):
        self.population = population
        if province is None:
            self.area = country
        else:
            self.area = f"{country}{self.SEP}{province}"
        sr_data = SRData(
            clean_df, country=country, province=province
        )
        self.train_df = sr_data.make(
            population, start_date=start_date, end_date=end_date
        )
        self.result_df = None
        # Start date
        self.start_date = self.train_df.index.min()
        if start_date is not None:
            self.start_date = max(
                self.start_date,
                datetime.strptime(start_date, self.DATE_FORMAT)
            )
        self.start_date = self.start_date.strftime(self.DATE_FORMAT)
        # End date
        self.end_date = self.train_df.index.max()
        if end_date is not None:
            self.end_date = min(
                self.end_date,
                datetime.strptime(end_date, self.DATE_FORMAT)
            )
        self.end_date = self.end_date.strftime(self.DATE_FORMAT)

    def analyse(self):
        """
        Perform curve fitting of S-R trend with negative exponential function and save the result.
        """
        self.result_df = self._fitting(self.train_df)

    def _fitting(self, train_df):
        """
        Perform curve fitting of S-R trend
            with negative exponential function.

        Args:
        @train_df (pandas.DataFrame): training dataset

            Index:
                - index (Date) (pd.TimeStamp): Observation date
            Columns:
                - Recovered: The number of recovered cases
                - Susceptible_actual: Actual data of Susceptible

        Returns:
            (pandas.DataFrame)
                Index:
                    - index (Date) (pd.TimeStamp): Observation date
                Columns:
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual values of Susceptible
                    - Susceptible_predicted: Predicted values of Susceptible
        """
        df = train_df.copy()
        # Calculate initial values of parameters
        x_series = df[self.R]
        y_series = df[f"{self.S}{self.A}"]
        a_ini = y_series.max()
        try:
            b_ini = y_series.diff().reset_index(drop=True)[1] / a_ini
        except KeyError:
            raise KeyError("The length of @train_df must be over 2.")
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
        df[f"{self.S}{self.P}"] = x_series.apply(
            lambda x: f_partial(x)
        )
        df = df.astype(np.int64, errors="ignore")
        return df

    def rmsle(self):
        """
        Calculate RMSLE score of actual/predicted Susceptible.

        Returns:
            (float): RMSLE score
        """
        df = self.result_df.copy()
        if df is None:
            raise NameError("Must perform Trend().analyse() in advance.")
        df = df.replace(np.inf, 0)
        df = df.loc[df[f"{self.S}{self.A}"] > 0, :]
        df = df.loc[df[f"{self.S}{self.P}"] > 0, :]
        if df.empty:
            return np.inf
        actual = df[f"{self.S}{self.A}"]
        predicted = df[f"{self.S}{self.P}"]
        # Calculate RMSLE score
        scores = np.abs(
            np.log10(actual + 1) - np.log10(predicted + 1)
        )
        return scores.sum()

    def result(self):
        """
        Show the result as a dataframe.
        """
        return self.result_df

    def show(self, filename=None):
        """
        show the result as a figure.

        Args:
            show_figure (bool): if True, show the history as a pair-plot of parameters.
            filename (str): filename of the figure, or None (show figure)
        """
        df = self.result()
        df["Predicted"] = df[f"{self.S}{self.P}"]
        title = f"{self.area}: S-R trend from {self.start_date} to {self.end_date}"
        self.show_with_many(
            result_df=df, predicted_cols=["Predicted"],
            title=title,
            filename=filename
        )

    @classmethod
    def show_with_many(cls, result_df, predicted_cols,
                       title, vlines=None, filename=None):
        """
        show the result as a figure.

        Args:
            result_df (pandas.DataFrame): training dataset

                Index:
                    - index (Date) (pd.TimeStamp): Observation date
                Columns:
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual values of Susceptible
                    - columns defined by @columns
            predicted_cols (list[str]): list of columns which have predicted values
            title (str): title of the figure
            vlines (list[int]): list of Recovered values to show vertical lines
            filename (str): filename of the figure, or None (show figure)
        """
        df = result_df.copy()
        if df is None:
            raise NameError("Must perform Trend().analyse() in advance.")
        x_series = df[cls.R]
        actual = df[f"{cls.S}{cls.A}"]
        # Plot the actual values
        plt.plot(
            x_series, actual,
            label="Actual", color="black",
            marker=".", markeredgewidth=0, linewidth=0
        )
        # Plot the predicted values
        if len(predicted_cols) == 1:
            plt.plot(x_series, df[predicted_cols[0]], label="Regression")
        else:
            for col in predicted_cols:
                plt.plot(x_series, df[col], label=col.replace(cls.P, str()))
        # x-axis
        plt.xlabel(cls.R)
        plt.xlim(0, None)
        # y-axis
        plt.ylabel(cls.S)
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
