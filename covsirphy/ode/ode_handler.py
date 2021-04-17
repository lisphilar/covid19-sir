#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.ode_solver import _ODESolver
from covsirphy.ode.param_estimator import _ParamEstimator


class ODEHandler(Term):
    """
    Perform simulation and parameter estimation with multi-phased ODE models.

    Args:
        model (covsirphy.ModelBase): ODE model
        start_date (str): start date of simulation, like 14Apr2021
        tau (int): tau value [min]
    """

    def __init__(self, model, start_date, tau):
        self._model = self._ensure_subclass(model, ModelBase, name="model")
        self._start = pd.to_datetime(start_date)
        self._tau = self._ensure_tau(tau)
        # {"0th": {"y0": {initial values}, "param": {parameters}, "step_n": int}}
        self._info_dict = {}

    def add(self, end_date, y0_dict=None, param_dict=None):
        """
        Add a new phase.

        Args:
            end_date (str): end date of the phase
            y0_dict (dict[str, int] or None): initial values or None (not set)
            param_dict (dict[str, int] or None): parameter values or None (not set)

        Returns:
            ODEHandler: self

        Note:
            Internal variable "step_n" means from the start date to the next date of the end date.
        """
        phase = self.num2str(len(self._info_dict))
        all_step_n = self.steps(
            self._start.strftime(self.DATE_FORMAT),
            pd.to_datetime(end_date).strftime(self.DATE_FORMAT), tau=self._tau)
        step_n = all_step_n - sum(phase_dict["step_n"] for phase_dict in self._info_dict.values())
        self._info_dict[phase] = {"y0": y0_dict or {}, "param": param_dict or {}, "step_n": step_n, }
        return self

    def simulate(self):
        """
        Perform simulation with the multi-phased ODE model.

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
        dataframes = []
        for (_, info_dict) in self._info_dict.items():
            # Step numbers
            step_n = info_dict["step_n"]
            # Initial values: registered information (with priority) or the last values
            y0_dict = dataframes[-1].iloc[-1].to_dict() if dataframes else {}
            y0_dict.update(info_dict["y0"])
            # parameter values
            param_dict = info_dict["param"].copy()
            # Solve the initial value problem with the ODE model
            solver = _ODESolver(self._model, **param_dict)
            solved_df = solver.run(step_n=step_n, **y0_dict)
            dataframes += [solved_df.iloc[1:]] if dataframes else [solved_df]
        # Combine the simulation results
        df = pd.concat(dataframes, ignore_index=True, sort=True)
        return self._model.convert_reverse(df, start=self._start, tau=self._tau)

    def estimate_parameters(self, data, metric="RMSLE"):
        """
        Estimate ODE parameters with data.

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
        """
        self._ensure_dataframe(data, name="data", columns=self.DSIFR_COLUMNS)
        param_estimator = _ParamEstimator(model=self._model, data=data, tau=self._tau)
        return param_estimator.run(metric=metric)
