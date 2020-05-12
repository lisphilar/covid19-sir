#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.optimize import curve_fit
import pandas as pd
import optuna
import numpy as np
from collections import defaultdict
from datetime import timedelta
import functools
import warnings
import matplotlib.pyplot as plt
import matplotlib
from covsirphy.selection.area import select_area


class Trend(object):
    """
    Class for S-R trend analysis.
    """
    # TODO: Refactoring with method separation
    SUFFIX_DICT = defaultdict(lambda: "th")
    SUFFIX_DICT.update({1: "st", 2: "nd", 3: "rd"})

    def __init__(self, ncov_df, total_population, name=None, **kwargs):
        """
        @ncov_df <pd.DataFrame>: the clean data
        @total_population <int>: total population in the target area
        @name <str>: name of the area
        @kwargs: keword arguments of select_area()
        """
        df = select_area(ncov_df, **kwargs)
        # Timedelta from the first record date
        self.start_date = df["Date"].min()
        series = (df["Date"] - self.start_date).dt.total_seconds()
        df["day"] = (series / 24 / 60 / 60).astype(np.int64)
        df = df.groupby("day").sum()
        # Variables
        df["Susceptible"] = total_population - df["Confirmed"]
        df = df.loc[:, ["Recovered", "Susceptible"]]
        df = df.rename({"Susceptible": "Actual"}, axis=1)
        # Set to self
        self.all_df = df.copy()
        self.subsets = [df.copy()]
        self.total_population = total_population
        # Name
        if name is None:
            try:
                self.title = f"{kwargs['places'][0][0]}: "
            except Exception:
                self.title = str()
        else:
            self.title = f"{name}: "
        # Initiation
        self.n_points = 0
        self.study = None
        # Debug
        self.n_trials_performed = 0

    def _num2str(self, num):
        """
        Convert numbers to 1st, 2nd etc.
        @num <int>: number
        @return <str>
        """
        q, mod = divmod(num, 10)
        suffix = "th" if q == 1 else self.SUFFIX_DICT[mod]
        return f"{num}{suffix}"

    def curve_fit(self, subset_df, num):
        """
        Peform curve fitting and return the predicted values.
        @subset_df <pd.DataFrame>: subset of data to fit
        @num <int>: the number of subset
        @return <pd.DataFrame>:
            - index: elapsed time [day] from the start date
            - Recovered: actual number of Recovered
            - Actual: actual number of Susceptible
            - {num}th_phase: predicted number of Susceptible
        """
        # Arguments
        df = subset_df.copy()
        title = self._num2str(num)
        # Curve fitting
        # S = a * np.exp(-b * R)
        # dS/dR = - b * S

        def f(x, a, b):
            return a * np.exp(-b * x)
        a_ini = self.total_population
        b_ini = - df["Actual"].diff().reset_index(drop=True)[1] / a_ini
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param, _ = curve_fit(f, df["Recovered"],
                                 df["Actual"], p0=[a_ini, b_ini])
        # Predict Susceptible
        f_partial = functools.partial(f, a=param[0], b=param[1])
        df[f"{title}_phase"] = df["Recovered"].apply(lambda x: f_partial(x))
        return df

    def _show_figure(self, pred_df, day_list):
        """
        SHow trend.
        @pred_df <pd.DataFrame>: predicted data
            -index: timedelta from the first record date
            -columns: Recovered, Actual, __th_phase
        @day_list <list[datetime.date]>: change points
        @return <list[str]>: list of change points in string
        """
        df = pred_df.copy()
        # List of change points in string
        dates = [
            (self.start_date + timedelta(days=day))
            for day in day_list[1:]
        ]
        str_dates = [d.strftime("%d%b%Y") for d in dates]
        # Plotting
        for col in df.columns:
            if col == "Recovered":
                continue
            elif col == "Actual":
                plt.plot(
                    df["Recovered"], df["Actual"],
                    label=col, color="black", marker=".", markeredgewidth=0, linewidth=0
                )
            else:
                plt.plot(df["Recovered"], df[col], label=col)
        # y-axis
        plt.ylabel("Susceptible")
        plt.yscale("log")
        ymin, ymax = df["Actual"].min(), df["Actual"].max()
        ydiff_scale = int(np.log10(ymax - ymin))
        yticks = np.linspace(round(ymin, - ydiff_scale),
                             round(ymax, - ydiff_scale), 5)
        plt.yticks([v.round() for v in yticks])
        # x-axis
        plt.xlabel("Recovered")
        plt.xlim(0, None)
        # Offset in x/y axis
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        plt.gca().xaxis.set_major_formatter(fmt)
        plt.gca().yaxis.set_major_formatter(fmt)
        # Vertical lines with change points
        if len(day_list) > 1:
            for day in day_list[1:]:
                value = df.loc[df.index[day], "Recovered"]
                plt.axvline(x=value, color="black", linestyle=":")
        # Legend
        plt.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
        # Title
        if str_dates:
            plt.title(f"{self.title}S-R trend change at {','.join(str_dates)}")
        else:
            plt.title(f"{self.title}S-R trend without change points")
        # Show figure
        plt.show()
        # Return the list of change points in string
        return str_dates

    def show(self):
        """
        Show trend and return list of change points.
        @return <list[str]>: list of change points, like ['30Jan2020', '31Mar2020']
        """
        fixed_cols = ["Recovered", "Actual"]
        # Predict the future with curve fitting
        df_list = [
            self.curve_fit(df, num).drop(fixed_cols, axis=1)
            for (num, df) in enumerate(self.subsets)
        ]
        pred_df = pd.concat(df_list, axis=1)
        pred_df[fixed_cols] = self.all_df[fixed_cols]
        if "1st_phase" in pred_df.columns:
            phase0_name = "Initial_phase"
        else:
            phase0_name = "Regression"
        pred_df = pred_df.rename({"0th_phase": phase0_name}, axis=1)
        # The list of change points
        day_list = [df.index.min() for df in df_list]
        # Show figure and the list of change points in string
        str_dates = self._show_figure(pred_df, day_list)
        return str_dates

    def analyse(self, n_points=0, n_trials_cycle=10, allowance=0, n_trials_max=500):
        """
        Find change points and return list of change points.
        @n_points <int>: the number of change points
        @n_trials_cycle <int>: the number of trials in one cycle
            - When one cycle will ended,
            - check whether the list of estimated time points is
               equal to that of the last cycle with @allowance
        @allowance <int>: allowance of the check of @n_trial_cycle
        @n_trials_max <int>: max value of the number of trials
        @return: the same as self.show()
        """
        if len(self.all_df) < (n_points + 1) * 3:
            raise Exception("Get more data or reduce n_points!")
        # Without change points
        if n_points <= 0 or not isinstance(n_points, int):
            self.subsets = [self.all_df.copy()]
            return self.show()
        # Find change points using Optuna
        self.n_points = n_points
        last_start_ids = list()
        n_trials_performed = 0
        while True:
            self.run(n_trials=n_trials_cycle)
            n_trials_performed += n_trials_cycle
            # Check the result with allowance
            param_dict = self.study.best_params.copy()
            start_ids = sorted(param_dict.values())
            if last_start_ids:
                are_settled = [
                    abs(this - last) <= allowance
                    for (this, last) in zip(start_ids, last_start_ids)
                ]
                if all(are_settled) or n_trials_performed > n_trials_max:
                    break
            last_start_ids = start_ids[:]
        # Update the list of subsets
        self.subsets = self.create_subsets(start_ids)
        self.n_trials_performed = n_trials_performed
        return self.show()

    def create_subsets(self, start_ids):
        """
        Create the list of subsets using a list of the first row IDs.
        @star_ids <list[int]>: list of the first row IDs
        """
        subsets = list()
        df = self.all_df.copy()
        for sid in start_ids:
            df2 = df.loc[sid:, :]
            subsets.append(df.drop(df2.index, axis=0))
            df = df2.copy()
        subsets.append(df)
        return subsets

    def run(self, n_trials=10):
        """
        Try optimization using Optuna.
        @n_trials <int>: the number of trials
        """
        # Create study object
        if self.study is None:
            self.study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.RandomSampler(seed=123)
            )
        # Run trials
        self.study.optimize(
            lambda x: self.objective(x),
            n_trials=n_trials,
            n_jobs=-1
        )

    def objective(self, trial):
        """
        Objective function for Optuna study.
        @trial <Optuna.trial object>
        """
        # Suggest start row IDs
        selected = [self.all_df.index.max()]
        for i in range(self.n_points):
            id_min = 3 * (self.n_points - len(selected) + 1)
            id_max = selected[-1] - 3
            if id_min + 3 > id_max:
                return np.inf
            new = trial.suggest_int(str(i), id_min, id_max)
            selected.append(new)
        start_ids = sorted(selected)[:-1]
        # Create subsets
        subsets = self.create_subsets(start_ids)
        # Curve fitting for each subset
        df_list = [self.curve_fit(df, num)
                   for (num, df) in enumerate(subsets, start=1)]
        # Calculate the error
        return self.error_f(df_list)

    def error_f(self, df_list):
        """
        Error function of self.objective.
        We need to minimize the difference of actual/predicted Susceptibe.
        """
        diffs = [
            abs(df["Actual"] - df[f"{self._num2str(num)}_phase"]).sum()
            for (num, df) in enumerate(df_list, start=1)
        ]
        return sum(diffs)
