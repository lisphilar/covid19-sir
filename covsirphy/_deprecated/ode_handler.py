#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.stopwatch import StopWatch
from datetime import timedelta
import functools
from multiprocessing import cpu_count, Pool
import itertools
from covsirphy.util.error import UnExecutedError, deprecate
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.ode_solver_multi import _MultiPhaseODESolver
from covsirphy._deprecated.param_estimator import _ParamEstimator


class ODEHandler(Term):
    """
    Perform simulation and parameter estimation with a multi-phased ODE model.

    Args:
        model (covsirphy.ModelBase): ODE model
        first_date (str or pandas.Timestamp): the first date of simulation, like 14Apr2021
        tau (int or None): tau value [min] or None (to be estimated)
        metric (str): metric name for estimation
        n_jobs (int): the number of parallel jobs or -1 (CPU count)
    """

    @deprecate(old="ODEHandler", new="Dynamics", version="2.24.0-xi")
    def __init__(self, model, first_date, tau=None, metric="RMSLE", n_jobs=-1):
        self._model = Validator(model, "model").subclass(ModelBase)
        self._first = Validator(first_date, "first_date").date()
        self._metric = Validator([metric], "metric").sequence(candidates=Evaluator.metrics())[0]
        self._n_jobs = cpu_count() if n_jobs == -1 else Validator(n_jobs, "n_jobs").int()
        # Tau value [min] or None
        self._tau = Validator(tau, "tau").tau(default=None)
        # {"0th": output of self.add()}
        self._info_dict = {}

    def add(self, end_date, param_dict=None, y0_dict=None):
        """
        Add a new phase.

        Args:
            end_date (str or pandas.Timestamp): end date of the phase
            param_dict (dict[str, float] or None): parameter values or None (not set)
            y0_dict (dict[str, int] or None): initial values or None (not set)

        Returns:
            dict(str, object): setting of the phase (key: phase name)
                - Start (pandas.Timestamp): start date
                - End (pandas.Timestamp): end date
                - y0 (dict[str, int]): initial values of model-specialized variables or empty dict
                - param (dict[str, float]): parameter values or empty dict
        """
        if not self._info_dict and y0_dict is None:
            raise ValueError("@y0_dict must be specified for the 0th phase, but None was applied.")
        phase = self.num2str(len(self._info_dict))
        if self._info_dict:
            start = list(self._info_dict.values())[-1][self.END] + timedelta(days=1)
        else:
            start = self._first
        end = Validator(end_date, "end_date").date()
        self._info_dict[phase] = {
            self.START: start, self.END: end, "y0": y0_dict or {}, "param": param_dict or {}}
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
                "ODEHandler.estimate_tau()", details="Or specify tau when creating an instance of ODEHandler")
        if not self._info_dict:
            raise UnExecutedError("ODEHandler.add()")
        combs = itertools.product(self._model.PARAMETERS, self._info_dict.items())
        for (param, (phase, phase_dict)) in combs:
            if param not in phase_dict["param"]:
                raise ValueError(f"{param.capitalize()} is not registered for the {phase} phase.")
        solver = _MultiPhaseODESolver(self._model, self._first, self._tau)
        return solver.simulate(*self._info_dict.values())

    def _score_tau(self, tau, data, quantile):
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
            quantile (float): quantile to guess ODE parameter values for the candidates of tau
        """
        info_dict = self._info_dict.copy()
        for (phase, phase_dict) in info_dict.items():
            start, end = phase_dict[self.START], phase_dict[self.END]
            df = data.loc[(start <= data[self.DATE]) & (data[self.DATE] <= end)]
            info_dict[phase]["param"] = self._model.guess(df, tau, q=quantile)
        solver = _MultiPhaseODESolver(self._model, self._first, tau)
        sim_df = solver.simulate(*info_dict.values())
        evaluator = Evaluator(data.set_index(self.DATE), sim_df.set_index(self.DATE))
        return evaluator.score(metric=self._metric)

    def estimate_tau(self, data, guess_quantile=0.5):
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
            guess_quantile (float): quantile to guess ODE parameter values for the candidates of tau

        Returns:
            int: estimated tau value [min]

        Raises:
            covsirphy.UnExecutedError: phase information was not set

        Note:
            ODE parameter for each tau value will be guessed by .guess() classmethod of the model.
            Tau value will be selected from the divisors of 1440 [min] and set to self.
        """
        Validator(data, "data").dataframe(columns=self.DSIFR_COLUMNS)
        df = data.loc[:, self.DSIFR_COLUMNS]
        if not self._info_dict:
            raise UnExecutedError("ODEHandler.add()")
        # Calculate scores of tau candidates
        Validator(guess_quantile, "quantile").float(value_range=(0, 1))
        calc_f = functools.partial(self._score_tau, data=df, quantile=guess_quantile)
        divisors = [i for i in range(1, 1441) if 1440 % i == 0]
        if self._n_jobs == 1:
            scores = [calc_f(candidate) for candidate in divisors]
        else:
            with Pool(self._n_jobs) as p:
                scores = p.map(calc_f, divisors)
        score_dict = dict(zip(divisors, scores))
        # Return the best tau value
        comp_f = {True: min, False: max}[Evaluator.smaller_is_better(metric=self._metric)]
        self._tau = comp_f(score_dict.items(), key=lambda x: x[1])[0]
        return self._tau

    def _estimate_params(self, phase, data, quantiles, check_dict, study_dict, show_phase):
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
            show_phase (bool): whether show phase name or not (stdout)

        Returns:
            dict(str, object):
                - Rt (float): phase-dependent reproduction number
                - (str, float): estimated parameter values, including rho
                - (int or float): day parameters, including 1/beta [days]
                - {metric}: score with the estimated parameter values
                - Trials (int): the number of trials
                - Runtime (str): runtime of optimization
        """
        phase_dict = self._info_dict[phase].copy()
        start, end = phase_dict[self.START], phase_dict[self.END]
        df = data.loc[(start <= data[self.DATE]) & (data[self.DATE] <= end)]
        estimator = _ParamEstimator(self._model, df, self._tau, self._metric, quantiles)
        est_dict = estimator.run(check_dict, study_dict)
        n_trials, runtime = est_dict[self.TRIALS], est_dict[self.RUNTIME]
        start_date = start.strftime(self.DATE_FORMAT)
        end_date = end.strftime(self.DATE_FORMAT)
        if show_phase:
            ph_statement = f"{phase:>4} phase ({start_date} - {end_date})"
        else:
            ph_statement = f"{start_date} - {end_date}"
        print(f"\t{ph_statement}: finished {n_trials:>4} trials in {runtime}")
        return est_dict

    def estimate_params(self, data, quantiles=(0.1, 0.9), check_dict=None, study_dict=None, **kwargs):
        """
        Estimate ODE parameter values of the all phases to minimize the score of the metric.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            quantiles (tuple(int, int)): quantiles to cut parameter range, like confidence interval
            check_dict (dict[str, object] or None): setting of validation
                - None means {"timeout": 180, "timeout_interaction": 1, "tail_n": 4, "allowance": (0.99, 1.01)}
                - timeout (int): timeout of optimization
                - timeout_iteration (int): timeout of one iteration
                - tail_n (int): the number of iterations to decide whether score did not change for the last iterations
                - allowance (tuple(float, float)): the allowance of the max predicted values
            study_dict (dict[str, object] or None): setting of optimization study
                - None means {"pruner": "threshold", "upper": 0.5, "percentile": 50, "seed": 0, "constant_liar": False}
                - pruner (str): kind of pruner (hyperband, median, threshold or percentile)
                - upper (float): works for "threshold" pruner, intermediate score is larger than this value, it prunes
                - percentile (float): works for "Percentile" pruner, the best intermediate value is in the bottom percentile among trials, it prunes
                - constant_liar (bool): whether use constant liar to reduce search effort or not
            kwargs: we can set arguments directly. E.g. timeout=180 for check_dict={"timeout": 180,...}

        Raises:
            covsirphy.UnExecutedError: either tau value or phase information was not set

        Returns:
            dict(str, object): setting of the phase (key: phase name)
                - Start (pandas.Timestamp): start date
                - End (pandas.Timestamp): end date
                - Rt (float): phase-dependent reproduction number
                - (str, float): estimated parameter values, including rho
                - (int or float): day parameters, including 1/beta [days]
                - {metric}: score with the estimated parameter values
                - Trials (int): the number of trials
                - Runtime (str): runtime of optimization
        """
        print(f"\n<{self._model.NAME} model: parameter estimation>")
        print(f"Running optimization with {self._n_jobs} CPUs...")
        stopwatch = StopWatch()
        # Arguments
        Validator(data, "data").dataframe(columns=self.DSIFR_COLUMNS)
        df = data.loc[:, self.DSIFR_COLUMNS]
        if not self._info_dict:
            raise UnExecutedError("ODEHandler.add()")
        if self._tau is None:
            raise UnExecutedError(
                "ODEHandler.estimate_tau()",
                details="Or specify tau when creating an instance of ODEHandler")
        # Arguments of _ParamEstimator
        check_kwargs = {"timeout": 180, "timeout_iteration": 1, "tail_n": 4, "allowance": (0.99, 1.01)}
        check_kwargs.update(check_dict or {})
        check_kwargs.update(kwargs)
        study_kwargs = {
            "pruner": "threshold", "upper": 0.5, "percentile": 50, "seed": 0, "constant_liar": False}
        study_kwargs.update(study_dict or {})
        study_kwargs.update(kwargs)
        # ODE parameter estimation
        phases = list(self._info_dict.keys())
        est_f = functools.partial(
            self._estimate_params, data=df, quantiles=quantiles,
            check_dict=check_kwargs, study_dict=study_kwargs, show_phase=(len(phases) > 1))
        if self._n_jobs == 1:
            est_dict_list = [est_f(ph) for ph in phases]
        else:
            with Pool(self._n_jobs) as p:
                est_dict_list = p.map(est_f, phases)
        for (phase, est_dict) in zip(phases, est_dict_list):
            self._info_dict[phase]["param"] = {
                param: est_dict[param] for param in self._model.PARAMETERS}
        print(f"Completed optimization. Total: {stopwatch.stop_show()}")
        return {
            k: {self.START: self._info_dict[k][self.START], self.END: self._info_dict[k][self.END], **v}
            for (k, v) in zip(phases, est_dict_list)}

    def estimate(self, data, **kwargs):
        """
        Estimate tau value [min] and ODE parameter values.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            kwargs: keyword arguments of  ODEHandler.estimate_tau() and ODEHandler.estimate_param()

        Raises:
            covsirphy.UnExecutedError: phase information was not set

        Returns:
            tuple(int, dict(str, dict[str, object]))
                - int: tau value [min]
                - dict(str, object): setting of the phase (key: phase name)
                    - Start (pandas.Timestamp): start date
                    - End (pandas.Timestamp): end date
                    - Rt (float): phase-dependent reproduction number
                    - (str, float): estimated parameter values, including rho
                    - (int or float): day parameters, including 1/beta [days]
                    - {metric}: score with the estimated parameter values
                    - Trials (int): the number of trials
                    - Runtime (str): runtime of optimization
        """
        tau = self.estimate_tau(data, **Validator(kwargs, "keyword arguments").kwargs(functions=self.estimate_tau))
        info_dict = self.estimate_params(data, **kwargs)
        return (tau, info_dict)
