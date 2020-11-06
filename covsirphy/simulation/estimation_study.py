#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import optuna
import pandas as pd
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.optimize import Optimizer
from covsirphy.simulation.simulator import ODESimulator


class EstimationStudy(Optimizer):
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

    def __init__(self, record_df, model, population, seed, tau=None, **kwargs):
        # Dataset
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # Arguments
        self.population = self.ensure_population(population)
        self.model = self.ensure_subclass(model, ModelBase, name="model")
        # Initial values
        df = model.tau_free(self.record_df, population, tau=None)
        self.y0_dict = {
            k: df.loc[df.index[0], k] for k in model.VARIABLES}
        # Fixed parameter values
        self.fixed_dict = {
            k: v for (k, v) in kwargs.items()
            if k in set(model.PARAMETERS) and v is not None
        }
        # For optimization
        seed = self.ensure_natural_int(seed, include_zero=True)
        self._init_study(seed=seed)
        optuna.logging.disable_default_handler()
        self.x = self.TS
        self.y_list = model.VARIABLES[:]
        self.weight_dict = {
            v: p for (v, p) in zip(model.VARIABLES, model.WEIGHTS) if p > 0}
        self.study = None
        self.total_trials = 0
        self.run_time = 0
        self.tau_candidates = self.divisors(1440)
        self._n_trials = None
        # Defined in parent class, but not used
        self.train_df = None
        # step_n will be defined in divide_minutes()
        self.step_n = None
        # tau value
        self.tau = self.ensure_tau(tau)
        self.taufree_df = pd.DataFrame() if tau is None else self.divide_minutes(tau)

    def run(self, timeout_iteration, seed=0):
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
        # Perform optimization
        self.study.optimize(
            self.objective, n_jobs=1, timeout=timeout_iteration)
        self._n_trials = len(self.study.trials)
        # Create a table to compare observed/estimated values
        self.tau = self.tau or super().param()[self.TAU]
        train_df = self.divide_minutes(self.tau)
        comp_df = self.compare(train_df, self.predict())
        return (train_df, comp_df)

    @property
    def n_trials(self):
        return self._n_trials

    def estimated(self):
        _dict = {self.TAU: self.tau}
        _dict.update(super().param())
        return _dict

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
