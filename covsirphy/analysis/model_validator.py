#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import swifter
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.cleaning.population import PopulationData
from covsirphy.analysis.example_data import ExampleData
from covsirphy.analysis.scenario import Scenario


class ModelValidator(Term):
    """
    Validate a ODE model as follows.
    - Create a model parameter set, phase length and initial values randomly.
    - Simulate the number of cases with an user-defined parameter set.
    - Perform parameter estimation with the theoretical data (the dataset created with step 1).
    - Compare the user-defined parameter set and estimated parameter set.
    - Repeat with anather set of parameters.
    - Small difference of the two parameter sets means that the model can be used for parameter estimation.

    Args:
        n_trials (int): the number of trials
        seed (int): random seed

    Notes:
        Population value and initial values are defined by model.EXAMPLE,
        tau value will be fixed as 1440 min.
    """

    def __init__(self, n_trials=10, seed=0):
        self.n_trials = self.ensure_natural_int(n_trials, name="n_trials")
        self.seed = self.ensure_natural_int(
            seed, name="seed", include_zero=True)
        self.tau = 1440
        # To avoid "imported but unused"
        self.__swifter = swifter

    def run(self, model):
        """
        Execute model validation.

        Args:
            model (covsirphy.ModelBase): ODE model
        """
        model = self.ensure_subclass(model, ModelBase, name="model")
        # Setup: parameter set and phase length
        df = self._setup(model, n_trials=self.n_trials, seed=self.seed)
        # Simulation and parameter estimation
        df["result"] = df.swifter.progress_bar(True).apply(
            lambda x: self._simulate_estimate(model, **x.to_dict()),
            axis=1
        )
        df = df.join(df["result"].apply(pd.Series), how="left", rsuffix="est")
        return df.drop("result", axis=1)

    def _setup(self, model, n_trials, seed):
        """
        Create a model parameter set, phase length randomly.

        Args:
            model (covsirphy.ModelBase): ODE model
            n_trials (int): the number of trials
            seed (int): random seed

        Returns:
            pandas.DataFrame:
                Index: reset index
                Columns:
                    - (float): parameter values from 0 to 1.0
                    - Rt (float): reproduction number
                    - step_n (int): step number of simulation
        """
        np.random.seed(seed)
        population = model.EXAMPLE[self.N.lower()]
        # Parameter set
        parameters = model.PARAMETERS[:]
        df = pd.DataFrame(
            np.random.rand(n_trials, len(parameters)), columns=parameters)
        # Reproduction number
        df[self.RT] = df.swifter.progress_bar(False).apply(
            lambda x: model(population, **x.to_dict()).calc_r0(),
            axis=1)
        # Tau value
        df[self.TAU] = self.tau
        # Step number
        df[self.STEP_N] = np.random.randint(5, 60, n_trials)
        # Return the setting
        return df

    def _simulate_estimate(self, model, step_n, **kwargs):
        """
        Perform simulation and parameter estimation.

        Args:
            model (covsirphy.ModelBase): ODE model
            tau (int): tau value [min]
            step_n (int): step number of simulation
            kwargs: keyword arguments of parameters

        Returns:
            dict(str, float): values of parameters and "Rt" (reproduction number)
        """
        name = model.NAME
        population = model.EXAMPLE[self.N.lower()]
        # Simulation
        param_dict = {
            k: v for (k, v) in kwargs.items() if k in set(model.PARAMETERS)}
        example_data = ExampleData(tau=self.tau, start_date="01Jan2020")
        example_data.add(
            model, step_n=step_n, country=name, param_dict=param_dict)
        # Create PopulationData instance
        population_data = PopulationData(filename=None)
        population_data.update(population, country=name)
        # Parameter estimation
        snl = Scenario(example_data, population_data, country=name)
        snl.add()
        snl.estimate(model, n_jobs=1)
        # Get estimated parameters
        columns = [self.RT, *model.PARAMETERS]
        df = snl.summary(columns=columns)
        return df.iloc[0].to_dict()
