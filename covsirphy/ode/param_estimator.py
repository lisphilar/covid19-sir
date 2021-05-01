#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import optuna
from covsirphy.util.argument import find_args
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.ode_solver import _ODESolver


class _ParamEstimator(Term):
    """
    Estimate ODE parameter values with records.

    Args:
        model (covsirphy.ModelBase): ODE model
        data (pandas.DataFrame):
            Index
                reset index
            Columns
                - Date (pd.Timestamp): Observation date
                - Susceptible(int): the number of susceptible cases
                - Infected (int): the number of currently infected cases
                - Fatal(int): the number of fatal cases
                - Recovered (int): the number of recovered cases
        tau (int): tau value [min]
        metric (str): metric to minimize
        quantiles (tuple(int, int)): quantiles to cut parameter range, like confidence interval
    """

    def __init__(self, model, data, tau, metric, quantiles):
        self._model = self._ensure_subclass(model, ModelBase, name="model")
        self._ensure_dataframe(data, name="data", columns=self.DSIFR_COLUMNS)
        self._tau = self._ensure_tau(tau, accept_none=False)
        self._metric = self._ensure_selectable(metric, Evaluator.metrics(), name="metric")
        # time steps (index), variables of the model
        df = model.convert(data, tau)
        self._taufree_df = df.copy()
        # Initial values
        self._y0_dict = df.iloc[0].to_dict()
        # Total population
        self._population = df.iloc[0].sum()
        # Step numbers
        self._step_n = df.index.max()
        # Parameter range
        self._range_dict = model.guess(data, tau, q=quantiles)
        # Max values of the variables
        self._max_dict = {v: df[v].max() for v in model.VARIABLES}

    def run(self, check_dict, study_dict):
        """
        Perform parameter estimation of the ODE model, not including tau.

        Args:
            check_dict (dict[str, object]): setting of validation
                - timeout (int): timeout of optimization
                - timeout_iteration (int): timeout of one iteration
                - tail_n (int): the number of iterations to decide whether score did not change for the last iterations
                - allowance (tuple(float, float)): the allowance of the max predicted values
            study_dict (dict[str, object]): setting of optimization study
                - pruner (str): kind of pruner (hyperband, median, threshold or percentile)
                - upper (float): works for "threshold" pruner, intermediate score is larger than this value, it prunes
                - percentile (float): works for "Percentile" pruner, the best intermediate value is in the bottom percentile among trials, it prunes

        Returns:
            dict(str, object):
                - Rt (float): phase-dependent reproduction number
                - (dict(str, float)): estimated parameter values
                - (dict(str, int or float)): day parameters, including 1/beta [days]
                - {metric}: score with the estimated parameter values
                - Trials (int): the number of trials
                - Runtime (str): runtime of optimization

        Note:
            Please refer to covsirphy.Evaluator.score() for metric names.
        """
        timeout = check_dict.get("timeout", 180)
        timeout_iteration = check_dict.get("timeout_iteration", 5)
        tail_n = check_dict.get("tail_n", 4)
        allowance = check_dict.get("allowance", (0.99, 1.01))
        # Initialize optimization
        study_kwargs = {"pruner": "threshold", "upper": 0.5, "percentile": 50, "seed": 0}
        study_kwargs.update(study_dict)
        study = self._init_study(**find_args(self._init_study, **study_kwargs))
        # The number of iterations
        iteration_n = math.ceil(timeout / timeout_iteration)
        stopmatch = StopWatch()
        # Optimization
        scores = []
        param_dict = {}
        for _ in range(iteration_n):
            # Run iteration
            study.optimize(self._objective, n_jobs=1, timeout=timeout_iteration)
            param_dict = study.best_params.copy()
            # If score did not change in the last iterations, stop running
            scores.append(self._score(**param_dict))
            if len(scores) >= tail_n and len(set(scores[-tail_n:])) == 1:
                break
            # Check max values are in the allowance
            if self._is_in_allowance(allowance, **param_dict):
                break
        model_instance = self._model(self._population, **param_dict)
        return {
            self.RT: model_instance.calc_r0(),
            **param_dict.copy(),
            **model_instance.calc_days_dict(self._tau),
            self._metric: self._score(**param_dict),
            self.TRIALS: len(study.trials),
            self.RUNTIME: stopmatch.stop_show(),
        }

    def _init_study(self, pruner, upper, percentile, seed):
        """
        Initialize Optuna study.

        Args:
            pruner (str): Hyperband, Median, Threshold or Percentile
            upper (float): works for "threshold" pruner,
                intermediate score is larger than this value, it prunes
            percentile (float): works for "threshold" pruner,
                the best intermediate value is in the bottom percentile among trials, it prunes
            seed (int or None): random seed of hyperparameter optimization

        Returns:
            optuna.study.Study
        """
        pruner_dict = {
            "hyperband": optuna.pruners.HyperbandPruner(),
            "median": optuna.pruners.MedianPruner(),
            "threshold": optuna.pruners.ThresholdPruner(upper=upper),
            "percentile": optuna.pruners.PercentilePruner(percentile=percentile),
        }
        return optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=pruner_dict[pruner.lower()],
        )

    def _objective(self, trial):
        """
        Objective function to minimize.
        Score will be calculated with the data and metric.

        Args:
            trial (optuna.trial): a trial of the study

        Returns:
            float: score
        """
        param_dict = {}
        for (k, v) in self._range_dict.items():
            try:
                param_dict[k] = trial.suggest_uniform(k, *v)
            except OverflowError:
                param_dict[k] = trial.suggest_uniform(k, 0, 1)
        return self._score(**param_dict)

    def _score(self, **kwargs):
        """
        Objective function to minimize.
        Score will be calculated the data and metric.

        Args:
            kwargs: values of non-dimensional model parameters, including rho and sigma

        Returns:
            float: score
        """
        # Simulate with applied parameter values
        solver = _ODESolver(model=self._model, **kwargs)
        sim_df = solver.run(step_n=self._step_n, **self._y0_dict)
        # The first variable (Susceptible) will be ignored in score calculation
        taufree_df = self._taufree_df.loc[:, self._taufree_df.columns[1:]]
        sim_df = sim_df.loc[:, sim_df.columns[1:]]
        # Calculate score
        evaluator = Evaluator(taufree_df, sim_df, how="inner", on=None)
        return evaluator.score(metric=self._metric)

    def _is_in_allowance(self, allowance, **kwargs):
        """
        Return whether all max values of estimated values are in allowance or not.

        Args:
            allowance (tuple(float, float)): the allowance of the predicted value
            kwargs: values of non-dimensional model parameters, including rho and sigma

        Returns:
            (bool): True when all max values of predicted values are in allowance
        """
        # Get max values with estimated parameter values
        solver = _ODESolver(model=self._model, **kwargs)
        sim_df = solver.run(step_n=self._step_n, **self._y0_dict)
        sim_max_dict = {v: sim_df[v].max() for v in self._model.VARIABLES}
        # Check all max values are in allowance
        allowance0, allowance1 = allowance
        ok_list = [
            (a * allowance0 <= p) and (p <= a * allowance1)
            for (a, p) in zip(self._max_dict.values(), sim_max_dict.values())]
        return all(ok_list)
