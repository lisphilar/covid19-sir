#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_log_error
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.simulator import ODESimulator


class Estimator(Term):
    """
    Hyperparameter optimization of an ODE model.

    Args:
        record_df (pandas.DataFrame)
            Index
                reset index
            Columns
                - Date (pd.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - any other columns will be ignored
        model (covsirphy.ModelBase): ODE model
        population (int): total population in the place
        tau (int): tau value [min], a divisor of 1440
        kwargs: parameter values of the model and data subseting
    """
    np.seterr(divide="raise")
    optuna.logging.disable_default_handler()
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", SyntaxWarning)

    def __init__(self, record_df, model, population, tau=None, **kwargs):
        # ODE model
        self.model = self._ensure_subclass(model, ModelBase, name="model")
        self.variables = model.VARIABLES[:]
        self.variables_evaluate = [
            v for (v, p) in zip(model.VARIABLES, model.WEIGHTS) if p > 0]
        # Dataset
        if not set(self.NLOC_COLUMNS).issubset(record_df.columns):
            record_df = model.restore(record_df)
        self.record_df = self._ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # Settings for simulation
        self.population = self._ensure_population(population)
        df = model.tau_free(self.record_df, population, tau=None)
        self.y0_dict = {
            k: df.loc[df.index[0], k] for k in model.VARIABLES}
        # Fixed parameter values
        self.fixed_dict = {
            k: v for (k, v) in kwargs.items()
            if k in set(model.PARAMETERS) and v is not None
        }
        # For optimization
        self.study = None
        self.total_trials = 0
        self.runtime = 0
        # Tau value
        self.tau_final = self._ensure_tau(tau)
        self.tau_candidates = self.divisors(1440)
        self.tau = tau
        if tau is None:
            self.step_n = None
            self.taufree_df = pd.DataFrame()
        else:
            self._set_taufree()

    def _init_study(self, seed, pruner, upper, percentile):
        """
        Initialize Optuna study.

        Args:
            seed (int or None): random seed of hyperparameter optimization
            pruner (str): Hyperband, Median, Threshold or Percentile
            upper (float): works for "threshold" pruner,
                intermediate RMSLE score is larger than this value, it prunes
            percentile (float): works for "threshold" pruner,
                the best intermediate value is in the bottom percentile among trials, it prunes
        """
        pruner_dict = {
            "hyperband": optuna.pruners.HyperbandPruner(),
            "median": optuna.pruners.MedianPruner(),
            "threshold": optuna.pruners.ThresholdPruner(upper=upper),
            "percentile": optuna.pruners.PercentilePruner(percentile=percentile),
        }
        try:
            self.study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=pruner_dict[pruner.lower()],
            )
        except KeyError:
            raise KeyError(
                f"@pruner should be selected from {', '.join(pruner_dict.keys())}.")

    def run(self, timeout=180, reset_n_max=3, timeout_iteration=10, tail_n=4, allowance=(0.98, 1.02),
            seed=0, pruner="threshold", upper=0.5, percentile=50, **kwargs):
        """
        Run optimization.
        If the result satisfied the following conditions, optimization ends.
        - RMSLE score did not change in the last @tail_n iterations.
        - Monotonic increasing variables increases monotonically.
        - Predicted values are in the allowance when each actual value shows max value.

        Args:
            timeout (int): timeout of optimization
            reset_n_max (int): if study was reset @reset_n_max times, will not be reset anymore
            timeout_iteration (int): time-out of one iteration
            tail_n (int): the number of iterations to decide whether RMSLE score did not change for the last iterations
            allowance (tuple(float, float)): the allowance of the predicted value
            seed (int or None): random seed of hyperparameter optimization
            pruner (str): hyperband, median, threshold or percentile
            upper (float): works for "threshold" pruner,
                intermediate RMSLE score is larger than this value, it prunes
            percentile (float): works for "Percentile" pruner,
                the best intermediate value is in the bottom percentile among trials, it prunes
            kwargs: other keyword arguments will be ignored

        Note:
            @n_jobs was obsoleted because this is not effective for Optuna.
        """
        # Create a study of optuna
        if self.study is None:
            self._init_study(
                seed=seed, pruner=pruner, upper=upper, percentile=percentile)
        reset_n = 0
        iteration_n = math.ceil(timeout / timeout_iteration)
        increasing_cols = [f"{v}{self.P}" for v in self.model.VARS_INCLEASE]
        stopwatch = StopWatch()
        rmsle_scores = []
        for _ in range(iteration_n):
            # Perform optimization
            self.study.optimize(
                self._objective, n_jobs=1, timeout=timeout_iteration)
            # If RMSLE did not change in the last iterations, stop running
            tau, param_dict = self._param()
            rmsle_scores.append(self._rmsle(tau=tau, param_dict=param_dict))
            if len(rmsle_scores) >= tail_n and len(set(rmsle_scores[-tail_n:])) == 1:
                break
            # Create a table to compare observed/estimated values
            comp_df = self._compare(tau=tau, param_dict=param_dict)
            # Check monotonic variables
            mono_ok_list = [
                comp_df[col].is_monotonic_increasing for col in increasing_cols
            ]
            if not all(mono_ok_list):
                if reset_n == reset_n_max - 1:
                    break
                # Initialize the study
                self._init_study(seed=seed)
                reset_n += 1
                continue
            # Need additional trials when the values are not in allowance
            if self._is_in_allowance(comp_df, allowance):
                break
        # Calculate run-time and the number of trials
        self.runtime += stopwatch.stop()
        self.total_trials = len(self.study.trials)

    def _is_in_allowance(self, comp_df, allowance):
        """
        Return whether all max values of predicted values are in allowance or not.

        Args:
            comp_df (pandas.DataFrame): [description]
            allowance (tuple(float, float)): the allowance of the predicted value

        Returns:
            (bool): True when all max values of predicted values are in allowance
        """
        a_max_values = [comp_df[f"{v}{self.A}"].max() for v in self.variables]
        p_max_values = [comp_df[f"{v}{self.P}"].max() for v in self.variables]
        allowance0, allowance1 = allowance
        ok_list = [
            (a * allowance0 <= p) and (p <= a * allowance1)
            for (a, p) in zip(a_max_values, p_max_values)
        ]
        return all(ok_list)

    def _objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the parameter values using Optuna.

        Args:
            trial (optuna.trial): a trial of the study

        Returns:
            float: score of the error function to minimize
        """
        self.tau = self.tau_final or trial.suggest_categorical(
            self.TAU, self.tau_candidates)
        self._set_taufree()
        # Set parameters of the models
        model_param_dict = self.model.param_range(
            self.taufree_df, self.population)
        param_dict = {
            k: self._suggest(trial, k, *v)
            for (k, v) in model_param_dict.items()
            if k not in self.fixed_dict.keys()
        }
        param_dict.update(self.fixed_dict)
        return self._rmsle(self.tau, param_dict)

    def _suggest(self, trial, name, min_value, max_value):
        """
        Suggest parameter value for the trial.

        Args:
            trial (optuna.trial): a trial of the study
            name (str): parameter name
            min_value (float): minimum value of the parameter
            max_value (float): max value of the parameter

        Returns:
            optuna.trial
        """
        try:
            return trial.suggest_uniform(name, min_value, max_value)
        except (OverflowError, np.AxisError):
            return trial.suggest_uniform(name, 0, 1)

    def _set_taufree(self):
        """
        Divide T by tau in the training dataset and calculate the number of steps.
        """
        self.taufree_df = self.model.tau_free(
            self.record_df, self.population, tau=self.tau)
        self.step_n = int(self.taufree_df[self.TS].max())

    def _rmsle(self, tau, param_dict):
        """
        Calculate RMSLE score.

        Args:
            tau (int): tau value [min]
            param_dict (dict[str, int or float]): dictionary of parameter values

        Returns:
            float: RMSLE score
        """
        comp_df = self._compare(tau, param_dict)
        rec_df = comp_df.loc[:, [
            f"{v}{self.A}" for v in self.variables_evaluate]]
        sim_df = comp_df.loc[:, [
            f"{v}{self.P}" for v in self.variables_evaluate]]
        msle = mean_squared_log_error(sim_df, rec_df)
        return np.sqrt(msle)

    def _simulate(self, step_n, param_dict):
        """
        Simulate the values with the parameters.

        Args:
            step_n (int): number of iteration
            dict[str, int or float]: dictionary of parameter values

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - t (int): Elapsed time divided by tau value [-]
                    - columns with dimensionalized variables
        """
        simulator = ODESimulator()
        simulator.add(
            model=self.model,
            step_n=step_n,
            population=self.population,
            param_dict=param_dict,
            y0_dict=self.y0_dict
        )
        return simulator.taufree()

    def _compare(self, tau, param_dict):
        """
        Return comparison table.

        Args:
            tau (int): tau value [min]
            dict[str, int or float]: dictionary of parameter values

        Returns:
            pandas.DataFrame:
                Index
                    t (int): Elapsed time divided by tau value [-]
                Columns
                    - columns with "_actual"
                    - columns with "_predicted:
                    - columns are defined by self.variables
        """
        self.tau = tau
        self._set_taufree()
        sim_df = self._simulate(self.step_n, param_dict)
        df = self.taufree_df.merge(
            sim_df, on=self.TS, suffixes=(self.A, self.P))
        return df.set_index(self.TS)

    def _param(self):
        """
        Return the estimated parameters as a dictionary.

        Returns:
            tuple(int, dict[str, int or float]): tau value and dictionary of parameter values
        """
        try:
            param_dict = self.study.best_params.copy()
        except ValueError:
            param_dict = {p: 0 for p in self.model.PARAMETERS}
            if self.tau_final is None:
                param_dict[self.TAU] = None
        param_dict.update(self.fixed_dict)
        tau = self.tau_final or param_dict.pop(self.TAU)
        return (tau, param_dict)

    def to_dict(self):
        """
        Summarize the results of optimization.

        Returns:
            dict[str, float or int]:
                - (parameters of the model)
                - tau
                - Rt: basic or phase-dependent reproduction number
                - (dimensional parameters [day])
                - RMSLE: Root Mean Squared Log Error
                - Trials: the number of trials
                - Runtime: run time of estimation
        """
        tau, param_dict = self._param()
        model_instance = self.model(population=self.population, **param_dict)
        return {
            **param_dict,
            self.TAU: tau,
            self.RT: model_instance.calc_r0(),
            **model_instance.calc_days_dict(tau),
            self.RMSLE: self._rmsle(tau, param_dict),
            self.TRIALS: self.total_trials,
            self.RUNTIME: StopWatch.show(self.runtime)
        }

    def _history(self):
        """
        Return dataframe to show the history of optimization.

        Args:
            show_figure (bool): if True, show the history as a pair-plot of parameters.
            filename (str): filename of the figure, or None (show figure)

        Returns:
            pandas.DataFrame: the history
        """
        # Create dataframe of the history
        df = self.study.trials_dataframe()
        series = df["datetime_complete"] - df["datetime_start"]
        df["time[s]"] = series.dt.total_seconds()
        drop_cols = [
            "datetime_complete", "datetime_start", "system_attrs__number"]
        return df.drop(drop_cols, axis=1, errors="ignore")

    def history(self, show_figure=True, filename=None):
        """
        Return dataframe to show the history of optimization.

        Args:
            show_figure (bool): if True, show the history as a pair-plot of parameters.
            filename (str): filename of the figure, or None (show figure)

        Returns:
            pandas.DataFrame: the history
        """
        df = self._history()
        if not show_figure:
            return df
        # Show figure
        fig_df = df.loc[:, df.columns.str.startswith("params_")]
        fig_df.columns = fig_df.columns.str.replace("params_", "")
        warnings.simplefilter("ignore", category=UserWarning)
        sns.pairplot(fig_df, diag_kind="kde", markers="+")
        # Save figure or show figure
        if filename is None:
            plt.show()
            return df
        plt.savefig(filename, bbox_inches="tight", transparent=False, dpi=300)
        plt.clf()
        return df

    def accuracy(self, show_figure=True, filename=None):
        """
        Show accuracy as a figure.

        Args:
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)

        Returns:
            pandas.DataFrame:
                Index
                    t (int): Elapsed time divided by tau value [-]
                Columns
                    - columns with "_actual"
                    - columns with "_predicted:
                    - columns are defined by self.variables
        """
        # Create a table to compare observed/estimated values
        df = self._compare(*self._param())
        if not show_figure:
            return df
        # Variables to show accuracy
        variables = [
            v for (i, (p, v))
            in enumerate(zip(self.model.WEIGHTS, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        # Prepare figure object
        val_len = len(variables) + 1
        fig, axes = plt.subplots(
            ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2))
        # Comparison of each variable
        for (ax, v) in zip(axes.ravel()[1:], variables):
            df[[f"{v}{self.A}", f"{v}{self.P}"]].plot.line(
                ax=ax, ylim=(None, None), sharex=True,
                title=f"{self.model.NAME}: Comparison regarding {v}(t)"
            )
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax.legend(
                bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0
            )
        # Summarize in a figure
        for v in variables:
            df[f"{v}_diff"] = df[f"{v}{self.A}"] - df[f"{v}{self.P}"]
            df[f"{v}_diff"].plot.line(
                ax=axes.ravel()[0], sharex=True,
                title=f"{self.model.NAME}: observed - estimated"
            )
        axes.ravel()[0].axhline(y=0, color="black", linestyle="--")
        axes.ravel()[0].yaxis.set_major_formatter(
            ScalarFormatter(useMathText=True))
        axes.ravel()[0].ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0))
        axes.ravel()[0].legend(
            bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
        fig.tight_layout()
        # Save figure or show figure
        if filename is None:
            plt.show()
            return df
        plt.savefig(filename, bbox_inches="tight", transparent=False, dpi=300)
        plt.clf()
        return df
