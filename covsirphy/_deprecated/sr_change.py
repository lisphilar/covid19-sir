#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import functools
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated.trend_plot import trend_plot


class _SRChange(Term):
    """
    Perform S-R trend analysis.

    Args:
        sr_df (pandas.DataFrame)
            Index
                Date (pd.TimeStamp): Observation date
            Columns
                - Recovered (int): the number of recovered cases (>0)
                - Susceptible (int): the number of susceptible cases
                - any other columns will be ignored
    """

    def __init__(self, sr_df):
        Validator(sr_df, name="sr_df").dataframe(time_index=True, columns=[self.S, self.R])
        # Index: Date, Columns: Recovered, Susceptible, logS
        self._sr_df = pd.DataFrame(
            {
                self.R: sr_df[self.R],
                self.S: sr_df[self.S],
                "logS": np.log10(sr_df[self.S].astype(np.float64)),
            }
        )

    def run(self, algorithm, **kwargs):
        """
        Run optimization and return the change points.

        Args:
            algorithm (classes of ruptures): detection algorithms
            kwargs: the other arguments of the algorithm class

        Returns:
            list[pandas.Timestamp]: list of change points
        """
        # Index: Recovered, Columns: logS
        df = self._sr_df.pivot_table(index=self.R, values="logS", aggfunc="last")
        df.index.name = None
        # Detect change points with Ruptures package: reset index + 1 values will be returned
        results = algorithm.fit_predict(df.iloc[:, 0].to_numpy(), pen=0.5)[:-1]
        # Convert reset index + 1 values to logS
        logs_df = df.iloc[[result - 1 for result in results]]
        # Convert logS to dates
        merge_df = pd.merge_asof(
            logs_df.sort_values("logS"), self._sr_df.reset_index().sort_values("logS"),
            on="logS", direction="nearest")
        return merge_df[self.DATE].sort_values().tolist()

    def _fitting_in_phase(self, start, end):
        """
        Perform curve fitting with the actual values in a phase on S-R plane.

        Args:
            start (pandas.Timestamp): start date of the phase
            end (pandas.Timestamp): end date of the phase
            func (function): function for fitting

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): observation date
                Columns
                    Fitted (float): fitted values of Susceptible
        """
        # Actual values for the phase
        df = self._sr_df.loc[start: end]
        # Curve fitting
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        param, _ = curve_fit(self._linear, df[self.R], df["logS"], maxfev=10000)
        # Get fitted values
        f_partial = functools.partial(self._linear, a=param[0], b=param[1])
        return pd.DataFrame({self.FITTED: 10 ** f_partial(df[self.R])}, index=df.index)

    @staticmethod
    def _linear(x, a, b):
        """
        Linear function f(x) = A x + b.

        Args:
            x (float): x values
            a (float): the first parameter of the function
            b (float): the second parameter of the function

        Returns:
            float
        """
        return a * x + b

    def _fitting(self, change_points):
        """
        Perform curve fitting with the actual values on S-R plane.

        Args:
            change_points (list[pandas.Timestamp]): list of change points

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): observation date
                Columns
                    - Recovered (int): The number of recovered cases
                    - Actual (int): actual values of Susceptible
                    - 0th, 1st, 2nd,... (float or None): fitted values of Susceptible for phases
        """
        df = self._sr_df.copy()
        # Start dates
        start_points = [df.index.min(), *change_points]
        # End dates
        end_points = [point - timedelta(days=1) for point in change_points] + [df.index.max()]
        # Phase names
        phases = [self.num2str(num) for num in range(len(change_points) + 1)]
        # Fitting
        df = df.rename(columns={self.S: self.ACTUAL}).drop("logS", axis=1)
        for (phase, start, end) in zip(phases, start_points, end_points):
            phase_df = self._fitting_in_phase(start=start, end=end)
            df = df.join(phase_df.rename(columns={self.FITTED: phase}), how="left")
        return df.round(0)

    def score(self, change_points, metric):
        """
        Calculate scores of the phases.

        Args:
            change_points (list[pandas.Timestamp]): list of change points
            metric (str): metric name

        Returns:
            list[float]: scores for phases

        Note:
            Please refer to covsirphy.Evaluator.score() for metric names
        """
        fit_df = self._fitting(change_points)
        phases = [self.num2str(num) for num in range(len(change_points) + 1)]
        scores = []
        for phase in phases:
            df = fit_df[[self.ACTUAL, phase]].dropna()
            evaluator = Evaluator(df[self.ACTUAL], df[phase], how="all")
            scores.append(evaluator.score(metric=metric))
        return scores

    def show(self, change_points, area, **kwargs):
        """
        Show the trend on S-R plane.

        Args:
            change_points (list[pandas.Timestamp]): list of change points
            area (str): area name (used in the figure title)
            kwargs: keyword arguments of covsirphy.trend_plot()
        """
        # Title
        if change_points:
            title = f"{area}: phases detected with S-R trend analysis"
        else:
            title = f"{area}: S-R trend without change points"
        # Curve fitting
        fit_df = self._fitting(change_points)
        # Show S-R plane
        fit_df = fit_df.rename(columns={"0th": self.INITIAL})
        plot_kwargs = {
            "title": title,
            "xlabel": self.R,
            "ylabel": self.S,
            "show_legend": True,
            "v": [fit_df.loc[point, self.R] for point in change_points],
        }
        plot_kwargs.update(kwargs)
        trend_plot(df=fit_df.set_index(self.R), actual_col=self.ACTUAL, **plot_kwargs)
