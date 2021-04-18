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


class ODEHandler(Term):
    """
    Perform simulation and parameter estimation with a multi-phased ODE model.

    Args:
        model (covsirphy.ModelBase): ODE model
        start_date (str): start date of simulation, like 14Apr2021
        tau (int or None): tau value [min] or None (to be determined)
    """

    def __init__(self, model, start_date, tau=None):
        self._model = self._ensure_subclass(model, ModelBase, name="model")
        self._start = pd.to_datetime(start_date)
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
            dict(str, object): setting of the phase
                - param (dict[str, float]): parameter values or empty dict
                - y0 (dict[str, int]): initial values of model-specialized variables or empty dict
                - start (pandas.Timestamp): start date
                - end (pandas.Timestamp): end date
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
            "param": param_dict or {}, "y0": y0_dict or {}, "start": start, "end": end}
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

    def estimate_tau(self, data, metric="RMSLE"):
        """
        Estimate tau value [min] to minimize the score of the metric with ODE parameters.

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
            metric (str): metric name

        Returns:
            int: estimated tau value [min]

        Raises:
            covsirphy.UnExecutedError: phase information was not set

        Note:
            ODE parameter for each tau value will be guessed by .guess() classmethod of the model.
            Tau value will be selected from the divisors of 1440 [min] and set to self
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
            score_dict[tau] = evaluator.score(metric=metric)
        # Return the best tau value
        score_f = {True: min, False: max}[Evaluator.smaller_is_better(metric=metric)]
        self._tau = score_f(score_dict.items(), key=lambda x: x[1])[0]
        return self._tau
