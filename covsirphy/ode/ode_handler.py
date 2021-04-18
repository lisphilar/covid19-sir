#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
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
    """

    def __init__(self, model, start_date, tau=None, metric="RMSLE"):
        self._model = self._ensure_subclass(model, ModelBase, name="model")
        self._start = pd.to_datetime(start_date)
        self._metric = self._ensure_selectable(metric, Evaluator.metrics(), name="metric")
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

    def estimate_tau(self, data):
        """
        Estimate tau value [min] to minimize the score of the metric.

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
        score_dict = {}
        for tau in self.divisors(1440):
            info_dict = self._info_dict.copy()
            for (phase, phase_dict) in info_dict.items():
                start, end = phase_dict["start"], phase_dict["end"]
                df = data.loc[(start <= data[self.DATE]) & (data[self.DATE] <= end)]
                info_dict[phase]["param"] = self._model.guess(df, tau)
            solver = _MultiPhaseODESolver(self._model, self._start, tau)
            sim_df = solver.simulate(*info_dict.values())
            evaluator = Evaluator(data.set_index(self.DATE), sim_df.set_index(self.DATE))
            score_dict[tau] = evaluator.score(metric=self._metric)
        # Return the best tau value
        score_f = {True: min, False: max}[Evaluator.smaller_is_better(metric=self._metric)]
        self._tau = score_f(score_dict.items(), key=lambda x: x[1])[0]
        return self._tau

    def estimate_params(self, data, quantiles=(0.1, 0.9),
                        check_dict={"timeout": 180, "timeout_interation": 5, "tail_n": 4, "allowance": (0.99, 1.01)},
                        study_dict={"pruner": "threshold", "upper": 0.5, "percentile": 50, "seed": 0}, **kwargs):
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
            check_dict (dict[str, object]): setting of validation
                - timeout (int): timeout of optimization
                - timeout_iteration (int): timeout of one iteration
                - tail_n (int): the number of iterations to decide whether score did not change for the last iterations
                - allowance (tuple(float, float)): the allowance of the max predicted values
            study_dict (dict[str, object]): setting of optimization study
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
        check_dict.update(kwargs)
        study_dict.update(kwargs)
        # ODE parameter estimation
        for (phase, phase_dict) in self._info_dict.items():
            start, end = phase_dict["start"], phase_dict["end"]
            df = data.loc[(start <= data[self.DATE]) & (data[self.DATE] <= end)]
            estimator = _ParamEstimator(self._model, df, self._tau, self._metric, quantiles)
            self._info_dict[phase].update(estimator.run(check_dict, study_dict))
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
