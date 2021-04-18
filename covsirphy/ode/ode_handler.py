#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import functools
from multiprocessing import cpu_count, Pool
from covsirphy.util.evaluator import Evaluator
import itertools
import pandas as pd
from covsirphy.util.error import UnExecutedError
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.ode_solver_multi import _MultiPhaseODESolver
from covsirphy.ode.param_estimator import _ParamEstimator


class ODEHandler(Term):
    """
    Perform simulation and parameter estimation with a multi-phased ODE model.

    Args:
        model (covsirphy.ModelBase): ODE model
        start_date (str): start date of simulation, like 14Apr2021
        tau (int or None): tau value [min] or None (to be determined)
        metric (str): metric name for estimation
        n_jobs (int): the number of parallel jobs or -1 (CPU count)
    """

    def __init__(self, model, start_date, tau=None, metric="RMSLE", n_jobs=-1):
        self._model = self._ensure_subclass(model, ModelBase, name="model")
        self._start = pd.to_datetime(start_date)
        self._metric = self._ensure_selectable(metric, Evaluator.metrics(), name="metric")
        self._n_jobs = cpu_count() if n_jobs == -1 else self._ensure_natural_int(n_jobs, name="n_jobs")
        # Tau value [min] or None
        self._tau = self._ensure_tau(tau, accept_none=True)
        # {"0th": output of self.add()}
        self._info_dict = {}

    def add(self, end_date, param_dict=None, y0_dict=None):
        """
        Add a new phase.

        Args:
            end_date (str): end date of the phase
            param_dict (dict[str, float] or None): parameter values or None (not set)
            y0_dict (dict[str, int] or None): initial values or None (not set)

        Returns:
            dict(str, object): setting of the phase (key: phase name)
                - start (pandas.Timestamp): start date
                - end (pandas.Timestamp): end date
                - y0 (dict[str, int]): initial values of model-specialized variables or empty dict
                - param (dict[str, float]): parameter values or empty dict
        """
        if not self._info_dict and y0_dict is None:
            raise ValueError("@y0_dict must be specified for the 0th phase, but None was applied.")
        phase = self.num2str(len(self._info_dict))
        if self._info_dict:
            start = list(self._info_dict.values())[-1]["end"] + timedelta(days=1)
        else:
            start = self._start
        end = pd.to_datetime(end_date)
        self._info_dict[phase] = {
            "start": start, "end": end, "y0": y0_dict or {}, "param": param_dict or {}}
        return self._info_dict[phase]

    def simulate(self):
        """
        Perform simulation with the multi-phased ODE model.

        Raises:
            covsirphy.UnExecutedError: either tau value or phase information was not set

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        if self._tau is None:
            raise UnExecutedError(
                "ODEHandler.estimate_tau()",
                message="or specify tau when creating an instance of ODEHandler")
        if not self._info_dict:
            raise UnExecutedError("ODEHandler.add()")
        combs = itertools.product(self._model.PARAMETERS, self._info_dict.items())
        for (param, (phase, phase_dict)) in combs:
            if param not in phase_dict["param"]:
                raise ValueError(f"{param.capitalize()} is not registered for the {phase} phase.")
        solver = _MultiPhaseODESolver(self._model, self._start, self._tau)
        return solver.simulate(*self._info_dict.values())

    def _score_tau(self, tau, data):
        """
        Calculate score for the tau value.

        Args:
            tau (int): tau value [min]
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        info_dict = self._info_dict.copy()
        for (phase, phase_dict) in info_dict.items():
            start, end = phase_dict["start"], phase_dict["end"]
            df = data.loc[(start <= data[self.DATE]) & (data[self.DATE] <= end)]
            info_dict[phase]["param"] = self._model.guess(df, tau)
        solver = _MultiPhaseODESolver(self._model, self._start, tau)
        sim_df = solver.simulate(*info_dict.values())
        evaluator = Evaluator(data.set_index(self.DATE), sim_df.set_index(self.DATE))
        return evaluator.score(metric=self._metric)

    def estimate_tau(self, data):
        """
        Select tau value [min] which minimize the score of the metric.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Returns:
            int: estimated tau value [min]

        Raises:
            covsirphy.UnExecutedError: phase information was not set

        Note:
            ODE parameter for each tau value will be guessed by .guess() classmethod of the model.
            Tau value will be selected from the divisors of 1440 [min] and set to self.
        """
        self._ensure_dataframe(data, name="data", columns=self.DSIFR_COLUMNS)
        if not self._info_dict:
            raise UnExecutedError("ODEHandler.add()")
        # Calculate scores of tau candidates
        calc_f = functools.partial(self._score_tau, data=data)
        divisors = self.divisors(1440)
        if self._n_jobs == 1:
            scores = [calc_f(candidate) for candidate in divisors]
        else:
            with Pool(self._n_jobs) as p:
                scores = p.map(calc_f, divisors)
        score_dict = {k: v for (k, v) in zip(divisors, scores)}
        # Return the best tau value
        comp_f = {True: min, False: max}[Evaluator.smaller_is_better(metric=self._metric)]
        self._tau = comp_f(score_dict.items(), key=lambda x: x[1])[0]
        return self._tau

    def _estimate_params(self, phase, data, quantiles, check_dict, study_dict):
        """
        Perform parameter estimation for one phase.

        Args:
            phase (str): phase name
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            quantiles (tuple(int, int)): quantiles to cut parameter range, like confidence interval
            check_dict (dict[str, object]): setting of validation
            study_dict (dict[str, object]): setting of optimization study

        Returns:
            dict(str, object):
                param (dict(str, float)): dictionary of estimated parameter values
                {metric}: score with the estimated parameter values
                Runtime (str): runtime of optimization
                Trials (int): the number of trials
        """
        phase_dict = self._info_dict[phase].copy()
        start, end = phase_dict["start"], phase_dict["end"]
        df = data.loc[(start <= data[self.DATE]) & (data[self.DATE] <= end)]
        estimator = _ParamEstimator(self._model, df, self._tau, self._metric, quantiles)
        return estimator.run(check_dict, study_dict)

    def estimate_params(self, data, quantiles=(0.1, 0.9), check_dict=None, study_dict=None, **kwargs):
        """
        Estimate ODE parameter values of the all phases to minimize the score of the metric.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            quantiles (tuple(int, int)): quantiles to cut parameter range, like confidence interval
            check_dict (dict[str, object] or None): setting of validation
                - None means {"timeout": 180, "timeout_interation": 5, "tail_n": 4, "allowance": (0.99, 1.01)}
                - timeout (int): timeout of optimization
                - timeout_iteration (int): timeout of one iteration
                - tail_n (int): the number of iterations to decide whether score did not change for the last iterations
                - allowance (tuple(float, float)): the allowance of the max predicted values
            study_dict (dict[str, object] or None): setting of optimization study
                - None means {"pruner": "threshold", "upper": 0.5, "percentile": 50, "seed": 0}
                - pruner (str): kind of pruner (hyperband, median, threshold or percentile)
                - upper (float): works for "threshold" pruner, intermediate score is larger than this value, it prunes
                - percentile (float): works for "Percentile" pruner, the best intermediate value is in the bottom percentile among trials, it prunes
            kwargs: we can set arguments directly. E.g. timeout=180 for check_dict={"timeout": 180,...}

        Raises:
            covsirphy.UnExecutedError: either tau value or phase information was not set

        Returns:
            dict(str, object): setting of the phase (key: phase name)
                - start (pandas.Timestamp): start date
                - end (pandas.Timestamp): end date
                - param (dict(str, float)): dictionary of estimated parameter values
                - y0 (dict[str, int]): initial values of model-specialized variables or empty dict
                - {metric}: score with the estimated parameter values
                - Runtime (str): runtime of optimization
                - Trials (int): the number of trials
        """
        # Arguments
        self._ensure_dataframe(data, name="data", columns=self.DSIFR_COLUMNS)
        if not self._info_dict:
            raise UnExecutedError("ODEHandler.add()")
        if self._tau is None:
            raise UnExecutedError(
                "ODEHandler.estimate_tau()",
                message="or specify tau when creating an instance of ODEHandler")
        # Arguments used in the old Estimator
        check_dict = check_dict or {
            "timeout": 180, "timeout_interation": 5, "tail_n": 4, "allowance": (0.99, 1.01)}
        check_dict.update(kwargs)
        study_dict = study_dict or {"pruner": "threshold", "upper": 0.5, "percentile": 50, "seed": 0}
        study_dict.update(kwargs)
        # ODE parameter estimation
        est_f = functools.partial(
            self._estimate_params, data=data, quantiles=quantiles,
            check_dict=check_dict, study_dict=study_dict)
        phases = list(self._info_dict.keys())
        if self._n_jobs == 1:
            est_dict_list = [est_f(ph) for ph in phases]
        else:
            with Pool(self._n_jobs) as p:
                est_dict_list = p.map(est_f, phases)
        for (phase, est_dict) in zip(phases, est_dict_list):
            self._info_dict[phase].update(est_dict)
        return self._info_dict

    def estimate(self, data, **kwargs):
        """
        Estimate tau value [min] and ODE parameter values.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            kwargs: keyword arguments of ODEHander.estimate_param()

        Raises:
            covsirphy.UnExecutedError: either tau value or phase information was not set

        Returns:
            tuple(int, dict(str, dict[str, object]))
                - int: tau value [min]
                - dict(str, object): setting of the phase (key: phase name)
                    - start (pandas.Timestamp): start date
                    - end (pandas.Timestamp): end date
                    - param (dict(str, float)): dictionary of estimated parameter values
                    - y0 (dict[str, int]): initial values of model-specialized variables or empty dict
                    - {metric}: score with the estimated parameter values
                    - Runtime (str): runtime of optimization
                    - Trials (int): the number of trials
        """
        tau = self.estimate_tau(data)
        info_dict = self.estimate_params(data, **kwargs)
        return (tau, info_dict)
