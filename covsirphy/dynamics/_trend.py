#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from datetime import timedelta
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import ruptures as rpt
from scipy.optimize import curve_fit
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.visualization.vbase import VisualizeBase


class _TrendAnalyzer(Term):
    """Class for S-R trend analysis to find change points of log10(S) - R of model specific variables.

    Args:
        data (pandas.DataFrame): new data to overwrite the current information
            Index
                Date (pandas.Timestamp): Observation date
            Columns
                Susceptible (int): the number of susceptible cases
                Infected (int): the number of currently infected cases
                Fatal (int): the number of fatal cases
                Recovered (int): the number of recovered cases
        model (covsirphy.ODEModel): definition of ODE model
        min_size (int): minimum value of phase length [days], be equal to or over 3

    Note:
        S-R trend analysis is original to Covsirphy, https://www.kaggle.com/code/lisphilar/covid-19-data-with-sir-model/notebook
        "Phase" means a sequential dates in which the parameters of SIR-derived models are fixed.
        "Change points" means the dates when trend was changed.
        "Change points" is the same as the start dates of phases except for the 0th phase.
    """

    def __init__(self, data, model, min_size):
        self._all_df = model.sr(data)
        self._algo_dict = {
            "Pelt-rbf": (rpt.Pelt, {"model": "rbf", "jump": 1, "min_size": min_size}),
            "Binseg-rbf": (rpt.Binseg, {"model": "rbf", "jump": 1, "min_size": min_size}),
            "Binseg-normal": (rpt.Binseg, {"model": "normal", "jump": 1, "min_size": min_size}),
            "BottomUp-rbf": (rpt.BottomUp, {"model": "rbf", "jump": 1, "min_size": min_size}),
            "BottomUp-normal": (rpt.BottomUp, {"model": "normal", "jump": 1, "min_size": min_size}),
        }
        self._r, self._logS = model._r, model._logS
        self._model_name = model._NAME

    def find_points(self, algo="Binseg-normal", **kwargs):
        """Find change points with S-R trend analysis.

        Args:
            algo (str): detection algorithms and models
            **kwargs: keyword arguments of algorithm classes (ruptures.Pelt, .Binseg, BottomUp) except for "model"

        Returns:
            tuple of (pandas.Timestamp): date of change points

        Note:
            Python library `ruptures` will be used for off-line change point detection.
            Refer to documentation of `ruptures` library, https://centre-borelli.github.io/ruptures-docs/
            Candidates of @algo are "Pelt-rbf", "Binseg-rbf", "Binseg-normal", "BottomUp-rbf", "BottomUp-normal".
        """
        algo_dict = self._algo_dict.copy()
        Validator([algo], "algo").sequence(candidates=algo_dict.keys())
        r, logS = self._r, self._logS
        sr_df = self._all_df.pivot_table(index=r, values=logS, aggfunc="last")
        sr_df.index.name = None
        warnings.filterwarnings("ignore", category=UserWarning)
        detector = algo_dict[algo][0](**algo_dict[algo][1], **Validator(kwargs).kwargs(algo_dict[algo][0]))
        results = detector.fit_predict(sr_df.iloc[:, 0].to_numpy(), pen=0.5)[:-1]
        logs_df = sr_df.iloc[[result - 1 for result in results]]
        merge_df = pd.merge_asof(
            logs_df.sort_values(logS), self._all_df.reset_index().sort_values(logS), on=logS, direction="nearest")
        return merge_df[self.DATE].sort_values().tolist()

    def fitting(self, points):
        """Perform fitting of data segmented with change points with linear function.

        Args:
            points (tuple of (pandas.Timestamp)): date of change points

        Returns:
            pandas.Dataframe:
                Index
                    R (int): actual R (R of the ODE model) values
                Columns
                    Actual (float): actual log10(S) (common logarithm of S of the ODE model) values
                    Fitted (float): log10(S) values fitted with y = a * R + b
                    0th (float): log10(S) values fitted with y = a * R + b and 0th phase data
                    1st, 2nd... (float): fitted values of 1st, 2nd phases
        """
        r, logS = self._r, self._logS
        all_df = self._all_df.copy()
        starts = [all_df.index.min(), *points]
        ends = [point - timedelta(days=1) for point in points] + [all_df.index.max()]
        for i, (start, end) in enumerate(zip(starts, ends)):
            phase_df = all_df.loc[start: end, :]
            param, _ = curve_fit(self._linear_f, phase_df[r], phase_df[logS], maxfev=10000)
            all_df[self.num2str(i)] = self._linear_f(phase_df[r], a=param[0], b=param[1])
        all_df[self.FITTED] = all_df.drop([logS, r], axis=1).sum(axis=1)
        return all_df.rename(columns={logS: self.ACTUAL}).set_index(r).groupby(level=0).first()

    @staticmethod
    def _linear_f(x, a, b):
        """
        Linear function f(x) = A * x + b.

        Args:
            x (float): x values
            a (float): the first parameter of the function
            b (float): the second parameter of the function

        Returns:
            float
        """
        return a * x + b

    def display(self, points, fit_df, name, **kwargs):
        """Display data on log10(S) - R plane with phases.

        Args:
            points (tuple of (pandas.Timestamp)): date of change points
            fit_df (pandas.Dataframe):
                Index
                    R (int): actual R (R of the ODE model) values
                Columns
                    Actual (float): actual log10(S) (common logarithm of S of the ODE model) values
                    Fitted (float): log10(S) values fitted with y = a * R + b
                    0th (float): log10(S) values fitted with y = a * R + b and 0th phase data
                    1st, 2nd... (float): fitted values of 1st, 2nd phases
            name (str or None): name of dynamics to show in figures (e.g. "baseline") or None (un-set)
            **kwargs: keyword arguments of covsirphy.VisualizeBase() and matplotlib.legend.Legend()
        """
        with VisualizeBase(**Validator(kwargs, "keyword arguments").kwargs([VisualizeBase, plt.savefig])) as lp:
            # _, lp.ax = plt.subplots(1, 1)
            lp.ax.plot(
                fit_df.index, fit_df[self.ACTUAL], label=self.ACTUAL,
                color="black", marker=".", markeredgewidth=0, linewidth=0)
            for phase in fit_df.drop([self.ACTUAL, self.FITTED], axis=1).columns:
                lp.ax.plot(fit_df.index, fit_df[phase], label=phase)
            for r_value in [self._all_df.loc[point, self._r] for point in points]:
                lp.ax.axvline(x=r_value, color="black", linestyle=":")
            pre = "Phases" if name is None else name + ": phases"
            lp.title = f"{pre} detected by S-R trend analysis of {self._model_name}"
            lp.ax.set_xlim(max(0, min(fit_df.index)), None)
            lp.ax.set_xlabel(xlabel=f"R of {self._model_name}")
            lp.ax.set_ylabel(ylabel=f"log10(S) of {self._model_name}")
            legend_kwargs = Validator(kwargs, "keyword arguments").kwargs(
                matplotlib.legend.Legend, default=dict(bbox_to_anchor=(0.5, -0.5), loc="lower center", borderaxespad=0, ncol=7))
            lp.ax.legend(**legend_kwargs)
