#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pandas as pd
from covsirphy.util.stopwatch import StopWatch
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.estimation_study import EstimationStudy


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

    Notes:
        If some columns are not included, they may be calculated with the model.
    """

    def __init__(self, record_df, model, population, tau=None, **kwargs):
        # Arguments
        self.population = self.ensure_population(population)
        self.model = self.ensure_subclass(model, ModelBase, name="model")
        # Dataset
        if not set(self.NLOC_COLUMNS).issubset(record_df.columns):
            record_df = model.restore(record_df)
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # For optimization
        self.y_list = model.VARIABLES[:]
        self.total_trials = 0
        self.run_time = 0
        # tau value
        self.tau = self.ensure_tau(tau)
        self.divided_df = pd.DataFrame()
        self.increasing_cols = [
            f"{v}{self.P}" for v in self.model.VARS_INCLEASE]

    def _create_study(self, record_df, tau, seed, **kwargs):
        return EstimationStudy(
            record_df=record_df, model=self.model, population=self.population,
            seed=seed, tau=tau, **kwargs
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
            timeout (int): time-out of run
            reset_n_max (int): if study was reset @reset_n_max times, will not be reset anymore
            timeout_iteration (int): time-out of one iteration
            allowance (tuple(float, float)): the allowance of the predicted value
            seed (int or None): random seed of hyperparameter optimization
            kwargs: other keyword arguments will be ignored

        Notes:
            @n_jobs was obsoleted because this is not effective for Optuna.
        """
        reset_n = 0
        iteration_n = math.ceil(timeout / timeout_iteration)
        stopwatch = StopWatch()
        study = self._create_study(
            self.record_df, self.tau, seed=seed, **kwargs)
        for _ in range(iteration_n):
            self.divided_df, comp_df, n_trials = study.run(
                timeout_iteration=timeout_iteration, seed=seed)
            self.est_dict = study.estimated()
            self.total_trials += n_trials
            # Check monotonic variables
            if not self._is_monotonic(comp_df):
                if reset_n == reset_n_max - 1:
                    break
                # Initialize the study
                study = self._create_study(
                    self.record_df, self.tau, seed=seed, **kwargs)
                reset_n += 1
                continue
            # Need additional trials when the values are not in allowance
            if self._is_in_allowance(comp_df, allowance):
                break
        # Calculate runtime
        self.run_time = stopwatch.stop()

    def _is_monotonic(self, comp_df):
        # Check monotonic variables
        mono_ok_list = [
            comp_df[col].is_monotonic_increasing for col in self.increasing_cols
        ]
        return all(mono_ok_list)

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
        est_dict = self.est_dict.copy()
        model_instance = self.model(
            population=self.population,
            **{k: v for (k, v) in est_dict.items() if k != self.TAU}
        )
        return {
            **est_dict,
            self.RT: model_instance.calc_r0(),
            **model_instance.calc_days_dict(est_dict[self.TAU]),
            self.RMSLE: self.rmsle(),
            self.TRIALS: self.total_trials,
            self.RUNTIME: StopWatch.show_time(self.run_time)
        }

    def rmsle(self):
        """
        Calculate RMSLE score.

        Returns:
            float: RMSLE score
        """
        return super().rmsle(train_df=self.divided_df, dim=1)

    def accuracy(self, show_figure=True, filename=None):
        """
        Show the accuracy as a figure.

        Args:
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
        """
        use_variables = [
            v for (i, (p, v))
            in enumerate(zip(self.model.WEIGHTS, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        return super().accuracy(
            train_df=self.divided_df,
            variables=use_variables,
            show_figure=show_figure,
            filename=filename
        )
