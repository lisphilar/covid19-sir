#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import optuna
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.util.stopwatch import StopWatch
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.optimize import Optimizer
from covsirphy.simulation.simulator import ODESimulator


class Estimator(Optimizer):
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

    def __init__(self, record_df, model, population, tau=None, **kwargs):
        # Arguments
        self.population = self.ensure_population(population)
        self.model = self.ensure_subclass(model, ModelBase, name="model")
        # Dataset
        if isinstance(record_df, JHUData):
            subset_arg_dict = find_args(
                [JHUData.subset, record_df.subset], **kwargs)
            self.record_df = record_df.subset(
                population=population, **subset_arg_dict)
        else:
            if not set(self.NLOC_COLUMNS).issubset(record_df.columns):
                record_df = model.restore(record_df)
            self.record_df = self.ensure_dataframe(
                record_df, name="record_df", columns=self.NLOC_COLUMNS
            )
        # Initial values
        df = model.tau_free(self.record_df, population, tau=None)
        self.y0_dict = {
            k: df.loc[df.index[0], k] for k in model.VARIABLES
        }
        # Fixed parameter values
        self.fixed_dict = {
            k: v for (k, v) in kwargs.items()
            if k in set(model.PARAMETERS) and v is not None
        }
        # For optimization
        optuna.logging.disable_default_handler()
        self.x = self.TS
        self.y_list = model.VARIABLES[:]
        self.weight_dict = {
            v: p for (v, p) in zip(model.VARIABLES, model.WEIGHTS) if p > 0
        }
        self.study = None
        self.total_trials = 0
        self.run_time = 0
        self.tau_candidates = self.divisors(1440)
        # Defined in parent class, but not used
        self.train_df = None
        # step_n will be defined in divide_minutes()
        self.step_n = None
        # tau value
        self.tau = self.ensure_tau(tau)
        self.taufree_df = pd.DataFrame() if tau is None else self.divide_minutes(tau)

    def run(self, timeout=60, reset_n_max=3,
            timeout_iteration=5, allowance=(0.98, 1.02), seed=0, **kwargs):
        """
        Run optimization.
        If the result satisfied the following conditions, optimization ends.
        - all values are not under than 0
        - values of monotonic increasing variables increases monotonically
        - predicted values are in the allowance when each actual value shows max value

        Args:
            timeout (int): time-out of run
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
            tau = self.tau or super().param()[self.TAU]
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
        self.run_time = stopwatch.stop()
        self.run_time_show = stopwatch.show()
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
        a_max_values = [comp_df[f"{v}{self.A}"].max() for v in self.y_list]
        p_max_values = [comp_df[f"{v}{self.P}"].max() for v in self.y_list]
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
            (pandas.DataFrame):
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
            (float): score of the error function to minimize
        """
        sim_df = self.simulate(self.step_n, param_dict)
        comp_df = self.compare(taufree_df, sim_df)
        # Calculate error score
        try:
            return sum(
                self._score(variable, comp_df)
                for variable in self.weight_dict.keys()
            )
        except (ZeroDivisionError, TypeError):
            return np.inf

    def _score(self, v, comp_df):
        """
        Calculate score of the variable.

        Args:
            v (str): variable na,e
            com_df (pandas.DataFrame):
                Index:
                    (str): time step
                Columns:
                    - columns with "_actual"
                    - columns with "_predicted"
                    - columns are defined by self.y_list

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
            (pandas.DataFrame):
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

    def to_dict(self):
        """
        Summarize the results of optimization.

        Args:
            name (str or None): index of the dataframe

        Returns:
            (pandas.DataFrame):
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
        est_dict = super().param()
        if self.TAU not in est_dict:
            est_dict[self.TAU] = self.tau
        model_instance = self.model(
            population=self.population,
            **{k: v for (k, v) in est_dict.items() if k != self.TAU}
        )
        minutes, seconds = divmod(int(self.run_time), 60)
        return {
            **est_dict,
            self.RT: model_instance.calc_r0(),
            **model_instance.calc_days_dict(est_dict[self.TAU]),
            self.RMSLE: self._rmsle(est_dict[self.TAU]),
            self.TRIALS: self.total_trials,
            self.RUNTIME: f"{minutes} min {seconds:>2} sec"
        }

    def summary(self, name=None):
        """
        Summarize the results of optimization.

        Args:
            name (str or None): index of the dataframe

        Returns:
            (pandas.DataFrame):
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
        summary_dict = {name or 0: self.to_dict()}
        df = pd.DataFrame.from_dict(summary_dict, orient="index")
        return df.fillna(self.UNKNOWN)

    def _rmsle(self, tau):
        """
        Return RMSLE score.

        Args:
            tau (int): tau value [min]
        """
        return super().rmsle(
            train_df=self.divide_minutes(tau),
            dim=1
        )

    def accuracy(self, show_figure=True, filename=None):
        """
        Show the accuracy as a figure.

        Args:
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
        """
        est_dict = super().param()
        if self.TAU not in est_dict:
            est_dict[self.TAU] = self.tau
        train_df = self.divide_minutes(est_dict[self.TAU])
        use_variables = [
            v for (i, (p, v))
            in enumerate(zip(self.model.WEIGHTS, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        return super().accuracy(
            train_df=train_df,
            variables=use_variables,
            show_figure=show_figure,
            filename=filename
        )
