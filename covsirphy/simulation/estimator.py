#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import math
from covsirphy.util.stopwatch import StopWatch
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.simulator import ODESimulator


class Estimator(Term):
    """
    Hyperparameter optimization of an ODE model.

    Args:
        record_df (pandas.DataFrame)
            Index:
                reset index
            Columns:
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
        self.model = self.ensure_subclass(model, ModelBase, name="model")
        self.variables = model.VARIABLES[:]
        self.weight_dict = {
            v: p for (v, p) in zip(model.VARIABLES, model.WEIGHTS) if p > 0}
        # Dataset
        if not set(self.NLOC_COLUMNS).issubset(record_df.columns):
            record_df = model.restore(record_df)
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # Settings for simulation
        self.population = self.ensure_population(population)
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
        self.tau = self.ensure_tau(tau)
        self.tau_candidates = self.divisors(1440)
        self.step_n = None
        self.taufree_df = pd.DataFrame() if tau is None else self.divide_minutes(tau)

    def _init_study(self, seed):
        """
        Initialize Optuna study.

        Args:
            seed (int or None): random seed of hyperparameter optimization
        """
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )

    def run(self, timeout=60, reset_n_max=3,
            timeout_iteration=5, allowance=(0.98, 1.02), seed=0, **kwargs):
        """
        Run optimization.
        If the result satisfied the following conditions, optimization ends.
        - all values are not under than 0
        - values of monotonic increasing variables increases monotonically
        - predicted values are in the allowance when each actual value shows max value

        Args:
            timeout (int): timeout of optimization
            reset_n_max (int): if study was reset @reset_n_max times, will not be reset anymore
            timeout_iteration (int): time-out of one iteration
            allowance (tuple(float, float)): the allowance of the predicted value
            seed (int or None): random seed of hyperparameter optimization
            kwargs: other keyword arguments will be ignored

        Notes:
            @n_jobs was obsoleted because this is not effective for Optuna.
        """
        # Create a study of optuna
        if self.study is None:
            self._init_study(seed=seed)
        reset_n = 0
        iteration_n = math.ceil(timeout / timeout_iteration)
        increasing_cols = [f"{v}{self.P}" for v in self.model.VARS_INCLEASE]
        stopwatch = StopWatch()
        for _ in range(iteration_n):
            # Perform optimization
            self.study.optimize(
                self.objective, n_jobs=1, timeout=timeout_iteration)
            # Create a table to compare observed/estimated values
            tau = self.tau or self.param()[self.TAU]
            train_df = self.divide_minutes(tau)
            comp_df = self.compare(train_df, self.predict())
            # Check monotonic variables
            mono_ok_list = [
                comp_df[col].is_monotonic_increasing for col in increasing_cols
            ]
            if not all(mono_ok_list):
                if reset_n == reset_n_max - 1:
                    break
                # Initialize the study
                self._init_study()
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

    def objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the parameter values using Optuna.

        Args:
            trial (optuna.trial): a trial of the study

        Returns:
            (float): score of the error function to minimize
        """
        # Convert T to t using tau
        taufree_df = self.taufree_df.copy()
        if taufree_df.empty:
            tau = trial.suggest_categorical(self.TAU, self.tau_candidates)
            taufree_df = self.divide_minutes(tau)
        # Set parameters of the models
        model_param_dict = self.model.param_range(
            taufree_df, self.population)
        p_dict = {
            k: trial.suggest_uniform(k, *v)
            for (k, v) in model_param_dict.items()
            if k not in self.fixed_dict.keys()
        }
        p_dict.update(self.fixed_dict)
        return self.error_f(p_dict, taufree_df)

    def divide_minutes(self, tau):
        """
        Divide T by tau in the training dataset and calculate the number of steps.

        Args:
            tau (int): tau value [min]

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns:
                    - t (int): Elapsed time divided by tau value [-]
                    - columns with dimensional variables
        """
        df = self.model.tau_free(self.record_df, self.population, tau=tau)
        self.step_n = int(df[self.TS].max())
        return df

    def error_f(self, param_dict, taufree_df):
        """
        Definition of error score to minimize in the study.

        Args:
            param_dict (dict): estimated parameter values
                - key (str): parameter name
                - value (int or float): parameter value
            taufree_df (pandas.DataFrame): training dataset

                Index:
                    reset index
                Columns:
                    - t: time steps [-]
                    - columns with dimensional variables

        Returns:
            float: score of the error function to minimize
        """
        sim_df = self.simulate(self.step_n, param_dict)
        comp_df = self.compare(taufree_df, sim_df)
        # Calculate error score
        return sum(
            self._score(variable, comp_df)
            for variable in self.weight_dict.keys()
        )

    def _score(self, v, comp_df):
        """
        Calculate score of the variable.

        Args:
            v (str): variable name
            com_df (pandas.DataFrame):
                Index:
                    (str): time step
                Columns:
                    - columns with "_actual"
                    - columns with "_predicted"
                    - columns are defined by self.variables

        Returns:
            float: score
        """
        weight = self.weight_dict[v]
        actual = comp_df.loc[:, f"{v}{self.A}"]
        diff = (actual - comp_df.loc[:, f"{v}{self.P}"]).abs() / (actual + 1)
        return weight * diff.mean()

    def simulate(self, step_n, param_dict):
        """
        Simulate the values with the parameters.

        Args:
            step_n (int): number of iteration
            param_dict (dict): estimated parameter values
                - key (str): parameter name
                - value (int or float): parameter value

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns:
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

    def compare(self, actual_df, predicted_df):
        """
        Return comparison table.

        Args:
            actual_df (pandas.DataFrame): actual data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by self.variables

            predicted_df (pandas.DataFrame): predicted data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by self.variables

        Returns:
            pandas.DataFrame:
                Index:
                    (str): time step
                Columns:
                    - columns with "_actual"
                    - columns with "_predicted:
                    - columns are defined by self.variables
        """
        # Data for comparison
        df = actual_df.merge(
            predicted_df, on=self.TS, suffixes=(self.A, self.P))
        return df.set_index(self.TS)

    def param(self):
        """
        Return the estimated parameters as a dictionary.

        Returns:
            dict
                - key (str): parameter name
                - value (int or float): parameter value
        """
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_dict)
        return param_dict

    def result(self, name):
        """
        Return the estimated parameters as a dataframe.
        This method should be overwritten in subclass.

        Args:
            name (str): index of the dataframe

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns:
                    - (estimated parameters)
                    - Trials: the number of trials
                    - Runtime: run time of estimation
        """
        param_dict = self.param()
        # The number of trials
        param_dict["Trials"] = self.total_trials
        # Runtime
        minutes, seconds = divmod(int(self.runtime), 60)
        param_dict["Runtime"] = f"{minutes} min {seconds} sec"
        return param_dict

    def to_dict(self):
        """
        Summarize the results of optimization.

        Args:
            name (str or None): index of the dataframe

        Returns:
            pandas.DataFrame:
                Index:
                    name or reset index (when name is None)
                Columns:
                    - (parameters of the model)
                    - tau
                    - Rt: basic or phase-dependent reproduction number
                    - (dimensional parameters [day])
                    - RMSLE: Root Mean Squared Log Error
                    - Trials: the number of trials
                    - Runtime: run time of estimation
        """
        est_dict = self.param()
        if self.TAU not in est_dict:
            est_dict[self.TAU] = self.tau
        model_instance = self.model(
            population=self.population,
            **{k: v for (k, v) in est_dict.items() if k != self.TAU}
        )
        minutes, seconds = divmod(int(self.runtime), 60)
        return {
            **est_dict,
            self.RT: model_instance.calc_r0(),
            **model_instance.calc_days_dict(est_dict[self.TAU]),
            self.RMSLE: self._rmsle(est_dict[self.TAU]),
            self.TRIALS: self.total_trials,
            self.RUNTIME: StopWatch.show(self.runtime)
        }

    def _rmsle(self, tau):
        """
        Calculate RMSLE score.

        Args:
            tau (int): tau value [min]

        Returns:
            float: RMSLE score
        """
        train_df = self.divide_minutes(tau)
        predicted_df = self.predict()
        df = self.compare(train_df, predicted_df)
        df = (df + 1).astype(np.int64)
        for v in self.variables:
            df = df.loc[df[f"{v}{self.A}"] * df[f"{v}{self.P}"] > 0, :]
        a_list = [np.log10(df[f"{v}{self.A}"]) for v in self.variables]
        p_list = [np.log10(df[f"{v}{self.P}"]) for v in self.variables]
        diffs = [((a - p) ** 2).sum() for (a, p) in zip(a_list, p_list)]
        return np.sqrt(sum(diffs) / len(diffs))

    def predict(self):
        """
        Predict the values with the calculated values.
        This method can be overwritten in subclass.
        """
        param_dict = self.param()
        return self.simulate(self.step_n, param_dict)

    def history(self, show_figure=True, filename=None):
        """
        Show the history of optimization as a figure
            and return it as dataframe.

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
        df = df.drop(
            ["datetime_complete", "datetime_start", "system_attrs__number"],
            axis=1, errors="ignore")
        # Show figure
        if not show_figure:
            return df
        fig_df = df.loc[:, df.columns.str.startswith("params_")]
        fig_df.columns = fig_df.columns.str.replace("params_", "")
        sns.pairplot(fig_df, diag_kind="kde", markers="+")
        # Save figure or show figure
        if filename is None:
            plt.show()
            return df
        plt.savefig(filename, bbox_inches="tight", transparent=False, dpi=300)
        plt.clf()
        return df

    def _accuracy(self, train_df, variables=None, show_figure=True, filename=None):
        """
        Show the accuracy as a figure.
        This method can be overwritten in child class.

        Args:
            train_df (pandas.DataFrame): actual data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by self.variables

            variables (list[str]): variables to compare or None (all variables)
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
        """
        # Create a table to compare observed/estimated values
        predicted_df = self.predict()
        df = self.compare(train_df, predicted_df)
        if not show_figure:
            return df
        # Prepare figure object
        val_len = len(variables) + 1
        fig, axes = plt.subplots(
            ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2)
        )
        # Comparison of each variable
        for (ax, v) in zip(axes.ravel()[1:], variables):
            df[[f"{v}{self.A}", f"{v}{self.P}"]].plot.line(
                ax=ax, ylim=(0, None), sharex=True,
                title=f"{self.model.NAME}: Comparison of observed/estimated {v}(t)"
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
        axes.ravel()[0].legend(bbox_to_anchor=(1.02, 0),
                               loc="lower left", borderaxespad=0)
        fig.tight_layout()
        # Save figure or show figure
        if filename is None:
            plt.show()
            return df
        plt.savefig(filename, bbox_inches="tight", transparent=False, dpi=300)
        plt.clf()
        return df

    def accuracy(self, show_figure=True, filename=None):
        """
        Show the accuracy as a figure.

        Args:
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
        """
        est_dict = self.param()
        if self.TAU not in est_dict:
            est_dict[self.TAU] = self.tau
        train_df = self.divide_minutes(est_dict[self.TAU])
        use_variables = [
            v for (i, (p, v))
            in enumerate(zip(self.model.WEIGHTS, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        return self._accuracy(
            train_df=train_df,
            variables=use_variables,
            show_figure=show_figure,
            filename=filename
        )
