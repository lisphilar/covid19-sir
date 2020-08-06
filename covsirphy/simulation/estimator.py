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
        kwargs: parameter values of the model and data subseting
    """
    np.seterr(divide="raise")

    def __init__(self, record_df, model, population, **kwargs):
        # Arguments
        self.population = self.ensure_natural_int(
            population, name="population"
        )
        self.model = self.ensure_subclass(model, ModelBase, name="model")
        # Dataset
        if isinstance(record_df, JHUData):
            subset_arg_dict = find_args(
                [JHUData.subset, record_df.subset], **kwargs)
            record_df = record_df.subset(
                population=population, **subset_arg_dict)
        else:
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
        fixable_set = set(model.PARAMETERS) & set([self.TAU])
        self.fixed_dict = {
            k: v for (k, v) in kwargs.items() if k in fixable_set
        }
        if self.TAU in self.fixed_dict:
            self.ensure_tau(self.fixed_dict[self.TAU])
        # For optimization
        optuna.logging.disable_default_handler()
        self.x = self.TS
        self.y_list = model.VARIABLES[:]
        self.study = None
        self.total_trials = 0
        self.run_time = 0
        self.tau_candidates = self.divisors(1440)
        # Defined in parent class, but not used
        self.train_df = None
        # step_n will be defined in divide_minutes()
        self.step_n = None

    def _run_trial(self, timeout_iteration):
        """
        Run trial.

        Args:
            timeout_iteration (int): time-out of one iteration
        """
        self.study.optimize(
            lambda x: self.objective(x),
            n_jobs=1,
            timeout=timeout_iteration
        )

    def run(self, timeout=60, reset_n_max=3,
            timeout_iteration=5, allowance=(0.98, 1.02),
            seed=0, stdout=True, **kwargs):
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
            stdout (bool): whether show the status of progress or not
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
        if stdout:
            print("\tRunning optimization...")
        stopwatch = StopWatch()
        for _ in range(iteration_n):
            # Perform optimization
            self._run_trial(timeout_iteration=timeout_iteration)
            # Create a table to compare observed/estimated values
            tau = super().param()[self.TAU]
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
        if stdout:
            print(
                f"\tFinished {self.total_trials} trials in {stopwatch.show()}.",
            )

    def _is_in_allowance(self, comp_df, allowance):
        """
        Return whether all max values of predicted values are in allowance or not.

        Args:
            comp_df (pandas.DataFrame): [description]
            allowance (tuple(float, float)): the allowance of the predicted value

        Returns:
            (bool): True when all max values of predicted values are in allowance
        """
        df = self.ensure_dataframe(comp_df, name="comp_df")
        variables = self.model.VARIABLES[:]
        a_max_values = [df[f"{v}{self.A}"].max() for v in variables]
        p_max_values = [df[f"{v}{self.P}"].max() for v in variables]
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
        fixed_dict = self.fixed_dict.copy()
        # Convert T to t using tau
        if self.TAU in fixed_dict.keys():
            tau = fixed_dict.pop(self.TAU)
        else:
            tau = trial.suggest_categorical(self.TAU, self.tau_candidates)
        taufree_df = self.divide_minutes(tau)
        # Set parameters of the models
        model_param_dict = self.model.param_range(
            taufree_df, self.population
        )
        p_dict = {
            k: trial.suggest_uniform(k, *v)
            for (k, v) in model_param_dict.items()
            if k not in self.fixed_dict.keys()
        }
        p_dict.update(fixed_dict)
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
        if self.step_n is None:
            raise ValueError("self.step_n must be defined in advance.")
        sim_df = self.simulate(self.step_n, param_dict)
        df = self.compare(taufree_df, sim_df)
        # Calculate error score
        v_list = [
            v for (p, v)
            in zip(self.model.PRIORITIES, self.model.VARIABLES)
            if p > 0
        ]
        diffs = [df[f"{v}{self.A}"] - df[f"{v}{self.P}"] for v in v_list]
        numerators = [df[f"{v}{self.A}"] + 1 for v in v_list]
        try:
            score = sum(
                p * np.average(diff.abs() / numerator, weights=df.index)
                for (p, diff, numerator)
                in zip(self.model.PRIORITIES, diffs, numerators)
            )
        except (ZeroDivisionError, TypeError):
            return np.inf
        return score

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
        simulator.run()
        return simulator.taufree()

    def summary(self, name=None):
        """
        Summarize the results of optimization.
        This function should be overwritten in subclass.

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
        param_dict = super().param()
        model_params = param_dict.copy()
        tau = model_params.pop(self.TAU)
        model_instance = self.model(
            population=self.population, **model_params
        )
        # Rt
        param_dict["Rt"] = model_instance.calc_r0()
        # dimensional parameters [day]
        param_dict.update(model_instance.calc_days_dict(tau))
        # RMSLE
        param_dict["RMSLE"] = self.rmsle(tau)
        # The number of trials
        param_dict["Trials"] = self.total_trials
        # Runtime
        minutes, seconds = divmod(int(self.run_time), 60)
        param_dict["Runtime"] = f"{minutes} min {seconds} sec"
        # Convert to dataframe
        df = pd.DataFrame.from_dict({str(name): param_dict}, orient="index")
        if name is None:
            df = df.reset_index(drop=True)
        return df.fillna(self.UNKNOWN)

    def rmsle(self, tau):
        """
        Return RMSLE score.

        Args:
            tau (int): tau value
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
        tau = super().param()[self.TAU]
        train_df = self.divide_minutes(tau)
        use_variables = [
            v for (i, (p, v))
            in enumerate(zip(self.model.PRIORITIES, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        return super().accuracy(
            train_df=train_df,
            variables=use_variables,
            show_figure=show_figure,
            filename=filename
        )
