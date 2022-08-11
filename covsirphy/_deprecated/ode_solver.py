#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated._mbase import ModelBase


class _ODESolver(Term):
    """
    Solve initial value problems for a SIR-derived ODE model.

    Args:
        model (covsirphy.ModelBase): SIR-derived ODE model
        kwargs: values of non-dimensional model parameters, including rho and sigma

    Note:
        We can check non-dimensional model parameters with model.PARAMETERS class variable.
        All non-dimensional parameters must be specified with keyword arguments.
    """

    def __init__(self, model, **kwargs):
        self._model = Validator(model, "model").subclass(ModelBase)
        param_dict = {k: float(v) for (k, v) in kwargs.items() if isinstance(v, (float, int))}
        self._param_dict = Validator(param_dict, "kwargs").dict(required_keys=model.PARAMETERS, errors="raise")

    def run(self, step_n, **kwargs):
        """
        Solve an initial value problem.

        Args:
            step_n (int): the number of steps
            kwargs: initial values of dimensional variables, including Susceptible

        Returns:
            pandas.DataFrame: numerical solution
                Index
                    reset index: time steps
                Columns
                    (int): dimensional variables of the model

        Note:
            We can check dimensional variables with model.VARIABLES class variable.
            All dimensional variables must be specified with keyword arguments.
            Total value of initial values will be regarded as total population.
        """
        # Check arguments
        step_n = Validator(step_n, "number").int(value_range=(1, None))
        kwargs = {param: int(value) for (param, value) in kwargs.items()}
        y0_dict = Validator(kwargs, "kwargs").dict(required_keys=self._model.VARIABLES, errors="raise")
        # Calculate population
        population = sum(y0_dict.values())
        # Solve problem
        return self._run(step_n=step_n, y0_dict=y0_dict, population=population)

    def _run(self, step_n, y0_dict, population):
        """
        Solve an initial value problem for a SIR-derived ODE model.

        Args:
            step_n (int): the number of steps
            y0_dict (dict[str, int]): initial values of dimensional variables, including Susceptible
            population (int): total population

        Returns:
            pandas.DataFrame: numerical solution
                Index
                    reset index: time steps
                Columns
                    (int): dimensional variables of the model
        """
        tstart, dt, tend = 0, 1, step_n
        variables = self._model.VARIABLES[:]
        initials = [y0_dict[var] for var in variables]
        sol = solve_ivp(
            fun=self._model(population=population, **self._param_dict),
            t_span=[tstart, tend],
            y0=np.array(initials, dtype=np.int64),
            t_eval=np.arange(tstart, tend + dt, dt),
            dense_output=False
        )
        y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=variables)
        return y_df.round().astype(np.int64)
