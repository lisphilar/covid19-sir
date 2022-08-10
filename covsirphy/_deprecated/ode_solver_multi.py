#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.error import NAFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.ode_solver import _ODESolver


class _MultiPhaseODESolver(Term):
    """
    Perform simulation with a multi-phased ODE model.

    Args:
        model (covsirphy.ModelBase): ODE model
        first (pandas.Timestamp): first date of simulation, like 14Apr2021
        tau (int): tau value [min]
    """

    def __init__(self, model, first, tau):
        self._model = Validator(model, "model").subclass(ModelBase)
        self._first = Validator(first, "first").instance(pd.Timestamp)
        self._tau = Validator(tau, "tau").tau(default=None)
        if self._tau is None:
            NAFoundError("tau", None)
        # {"0th": output of self.add()}
        self._info_dict = {}

    def _add(self, end, param_dict, y0_dict):
        """
        Add a new phase.

        Args:
            end (pandas.Timestamp): end date of the phase
            param_dict (dict[str, float]): parameter values
            y0_dict (dict[str, int] or None): initial values or None (not set)

        Returns:
            dict(str, object): setting of the phase
                - param (dict[str, float]): parameter values
                - y0 (dict[str, int]): initial values of model-specialized variables or empty dict
                - step_n (int): the number of steps

        Note:
            Internal variable "step_n" means from the first date to the next date of the end date.
        """
        Validator(end, end).instance(pd.Timestamp)
        phase = self.num2str(len(self._info_dict))
        all_step_n = self.steps(
            self._first.strftime(self.DATE_FORMAT), end.strftime(self.DATE_FORMAT), tau=self._tau)
        step_n = all_step_n - sum(phase_dict["step_n"] for phase_dict in self._info_dict.values())
        self._info_dict[phase] = {"param": param_dict, "y0": y0_dict or {}, "step_n": step_n}
        return self._info_dict[phase]

    def simulate(self, *args):
        """
        Perform simulation with the multi-phased ODE model.

        Args:
            args (dict(str, object)): list of phase settings.
                - End (pandas.Timestamp): end date of the phase
                - param_dict (dict[str, float]): parameter values
                - y0_dict (dict[str, int] or None): initial values or None (not set)

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
        # Settings
        for phase_dict in args:
            self._add(phase_dict[self.END], phase_dict["param"], phase_dict["y0"])
        # Multi-phased simulation
        dataframes = []
        for (_, phase_dict) in self._info_dict.items():
            # Step numbers
            step_n = phase_dict["step_n"]
            # Initial values: registered information (with priority) or the last values
            y0_dict = dataframes[-1].iloc[-1].to_dict() if dataframes else {}
            y0_dict.update(phase_dict["y0"])
            if set(self._model.VARIABLES) - y0_dict.keys():
                un_vars = list(set(self._model.VARIABLES) - y0_dict.keys())
                s, be = ("s", "were") if len(un_vars) > 1 else ("", "was")
                raise KeyError(f"Initial value{s} of <{'>, <'.join(un_vars)}> {be} not specified.")
            # parameter values
            param_dict = phase_dict["param"].copy()
            # Solve the initial value problem with the ODE model
            solver = _ODESolver(self._model, **param_dict)
            solved_df = solver.run(step_n=step_n, **y0_dict)
            dataframes += [solved_df.iloc[1:]] if dataframes else [solved_df]
        # Combine the simulation results
        df = pd.concat(dataframes, ignore_index=True, sort=True)
        return self._model.convert_reverse(df, start=self._first, tau=self._tau)
