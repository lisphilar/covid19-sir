#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import optuna
import pandas as pd
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.simulator import ODESimulator


class EstimationStudy(Term):
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
        seed (int or None): random seed of hyperparameter optimization
    """
    optuna.logging.disable_default_handler()
    np.seterr(divide="raise")

    def __init__(self, record_df, model, population, seed=0):
        # Dataset
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # ODE model
        self.model = self.ensure_subclass(model, ModelBase, name="model")
        self.variables = model.VARIABLES[:]
        self.weight_dict = {
            v: p for (v, p) in zip(model.VARIABLES, model.WEIGHTS) if p > 0}
        self.fixed_dict = {}
        # Settings for simulation
        self.population = self.ensure_population(population)
        df = model.tau_free(record_df, population, tau=None)
        self.y0_dict = {
            k: df.loc[df.index[0], k] for k in model.VARIABLES}
        # Settings for optimization
        seed = self.ensure_natural_int(seed, include_zero=True)
        self.study = self._create_study(seed=seed)
        # What to estimate
        self.pram_dict = {param: None for param in model.PARAMETERS}
        self.tau = None
        # Tau-free dataframe
        self.tau_candidates = self.divisors(1440)
        self.taufree_df = pd.DataFrame()
        self.step_n = None

    @property
    def n_trials(self):
        """
        int: the number of trials
        """
        return len(self.study.trials)

    def estimated(self):
        """
        Return the dictionary of estimated parameters and tau value.

        Returns:
            dict[str, float/int]: keys are "tau", "rho" etc.
        """
        return {self.TAU: self.tau, **self.param_dict}

    def _create_study(self, seed=None):
        """
        Initialize Optuna study.

        Args:
            seed (int or None): random seed of hyperparameter optimization

        Notes:
            @seed will effective when the number of CPUs is 1
        """
        return optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))

    def _set_taufree(self):
        """
        Set tau-free dataframe created with the records and tau value.
        """
        if self.tau is None:
            return
        self.taufree_df = self._create_taufree(self.tau)
        self.step_n = int(self.taufree_df[self.TS].max())

    def _create_taufree(self, tau):
        """
        Create tau-free dataframe using the records and tau value.

        Returns:
            pandas.DataFrame: tau-free dataframe
                Index:
                    reset index
                Columns:
                    - t: time steps [-]
                    - columns with dimensional variables
        """
        return self.model.tau_free(self.record_df, self.population, tau=tau)

    def run(self, timeout, tau=None, **kwargs):
        """
        Run trials of hyperparameter optimization.

        Args:
            timeout_iteration (int): time-out of one iteration
            tau (int or None): tau value [min], a divisor of 1440
            kwargs: parameter values of the model

        Returns:
            covsirphy.EstimationStudy: self

        Notes:
            If @tau is None, tau value will be estimated.
            This method can be called many times to run many trials.
        """
        # Tau value
        if tau is not None:
            self.tau = self.ensure_tau(tau)
            self._set_taufree()
        # Fixed parameter values
        self.fixed_dict = {
            k: v for (k, v) in kwargs.items()
            if k in set(self.variables) and v is not None}
        # Perform optimization
        self.study.optimize(self._objective, n_jobs=1, timeout=timeout)
        # Register the estimated values
        self.param_dict = self.study.best_params.copy()
        if self.TAU in self.param_dict:
            self.tau = self.param_dict.pop(self.TAU)
        self.param_dict.update(self.fixed_dict)
        return self

    def _objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the parameter values using Optuna.

        Args:
            trial (optuna.trial): a trial of the study

        Returns:
            float: score of the error function to minimize
        """
        # Set tau value and taufree dataframe
        if self.tau is None:
            self.tau = trial.suggest_categorical(self.TAU, self.tau_candidates)
            self.set_taufree()
        # Set parameters of the models
        range_dict = self.model.param_range(
            self.taufree_df, self.population)
        param_dict = {
            k: trial.suggest_uniform(k, *v)
            for (k, v) in range_dict.items() if k not in self.fixed_dict.keys()}
        param_dict.update(self.fixed_dict)
        return self._score_total(param_dict)

    def _score_total(self, param_dict):
        """
        Definition of error score to minimize in the study.

        Args:
            param_dict (dict): estimated parameter values
                - key (str): parameter name
                - value (int or float): parameter value

        Returns:
            float: score of the error function to minimize
        """
        sim_df = self._simulate(param_dict)
        comp_df = self._compare(sim_df)
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
                    t (int): time step, 0, 1, 2,...
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

    def _simulate(self, param_dict):
        """
        Simulate the number of cases with applied parameter values.

        Args:
            param_dict (dict): estimated parameter values
                - key (str): parameter name
                - value (int or float): parameter value

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns:
                    - t (int): time step, 0, 1, 2,...
                    - columns with dimensionalized variables
        """
        simulator = ODESimulator()
        simulator.add(
            model=self.model,
            step_n=self.step_n,
            population=self.population,
            param_dict=param_dict,
            y0_dict=self.y0_dict
        )
        return simulator.taufree()

    def _compare(self, sim_df):
        """
        Return comparison table.

        Args:
            sim_df (pandas.DataFrame): simulated number of cases
                Index:
                    reset index
                Columns:
                    - t (int): time step, 0, 1, 2,...
                    - includes columns defined by self.variables

        Returns:
            pandas.DataFrame:
                Index:
                    t (int): time step, 0, 1, 2,...
                Columns:
                    - columns with suffix "_actual"
                    - columns with suffix "_predicted"
                    - columns are defined by self.variables
        """
        # Data for comparison
        df = self.taufree_df.merge(
            sim_df, on=self.TS, suffixes=(self.A, self.P))
        return df.set_index(self.TS)

    def compare(self):
        """
        Create comparison table with the estimated parameter values.

        Returns:
            pandas.DataFrame:
                Index:
                    t (int): time step, 0, 1, 2,...
                Columns:
                    - columns with suffix "_actual"
                    - columns with suffix "_predicted"
                    - columns are defined by self.variables
        """
        if None in self.param_dict.values():
            raise ValueError(
                "EstimationStudy.run() must be performed in advance.")
        sim_df = self._simulate(self.param_dict)
        return self._compare(sim_df)

    def rmsle(self):
        """
        Calculate RMSLE score.

        Returns:
            float: RMSLE score
        """
        df = (self.compare() + 1).astype(np.int64)
        for v in self.variables:
            df = df.loc[df[f"{v}{self.A}"] * df[f"{v}{self.P}"] > 0, :]
        a_list = [np.log10(df[f"{v}{self.A}"]) for v in self.variables]
        p_list = [np.log10(df[f"{v}{self.P}"]) for v in self.variables]
        diffs = [((a - p) ** 2).sum() for (a, p) in zip(a_list, p_list)]
        return np.sqrt(sum(diffs) / len(diffs))

    def history(self):
        """
        Return the dataframe which show the details of hyperparameter optimization.

        Returns:
            pandas.DataFrame: the details of optimization
                Index:
                    reset index
                Columns:
                    - columns with suffix "params_" (float): estimated values
                    - time[s] (int): runtime [sec]
        """
        # Create dataframe of the history
        df = self.study.trials_dataframe()
        series = df["datetime_complete"] - df["datetime_start"]
        df["time[s]"] = series.dt.total_seconds()
        return df.drop(
            ["datetime_complete", "datetime_start", "system_attrs__number"],
            axis=1, errors="ignore")
