#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
import numpy as np
import pandas as pd
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.cleaning.population import PopulationData
from covsirphy.phase.phase_estimator import MPEstimator
from covsirphy.analysis.scenario import Scenario
from covsirphy.analysis.example_data import ExampleData


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
        tau (int): tau value [min]
        n_trials (int): the number of trials
        seed (int): random seed

    Note:
        Population value and initial values are defined by model.EXAMPLE.
        Estimators know tau values before parameter estimation.
    """

    def __init__(self, tau=1440, n_trials=8, seed=0):
        self.n_trials = self._ensure_natural_int(n_trials, name="n_trials")
        self.seed = self._ensure_natural_int(
            seed, name="seed", include_zero=True)
        self.tau = self._ensure_tau(tau)
        # Validated models
        self.model_names = []
        # Dataframes of results created by ._get_result()
        self._results = []

    def run(self, model, timeout=180, allowance=(0.98, 1.02), n_jobs=-1):
        """
        Execute model validation.

        Args:
            model (covsirphy.ModelBase): ODE model
            timeout (int): time-out of run
            allowance (tuple(float, float)): the allowance of the predicted value
            n_jobs (int): the number of parallel jobs or -1 (CPU count)

        Returns:
            covsirphy.ModelValidator: self
        """
        model = self._ensure_subclass(model, ModelBase, name="model")
        if model.NAME in self.model_names:
            raise ValueError(f"{model.NAME} has been validated.")
        self.model_names.append(model.NAME)
        # Setup: create parameter set, phase length and processor
        df = self._setup(model, n_trials=self.n_trials, seed=self.seed)
        processor = self._processor(model, df)
        # Parameter estimation
        units_estimated = processor.run(
            timeout=timeout, allowance=allowance, n_jobs=n_jobs)
        # Get estimated parameters
        self._results.append(self._get_result(model, df, units_estimated))
        return self

    def _setup(self, model, n_trials, seed):
        """
        Create a model parameter set, phase length randomly.

        Args:
            model (covsirphy.ModelBase): ODE model
            n_trials (int): the number of trials
            seed (int): random seed

        Returns:
            pandas.DataFrame:
                Index reset index
                Columns
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
        df[self.RT] = df.apply(
            lambda x: model(population, **x.to_dict()).calc_r0(),
            axis=1)
        # Tau value
        df[self.TAU] = self.tau
        # Step number
        df[self.STEP_N] = np.random.randint(5, 60, n_trials)
        # Return the setting
        return df

    def _processor(self, model, setting_df):
        """
        Generate multi-processor for parameter estimation,
        registering theoretical data and phase units.

        Args:
            model (covsirphy.ModelBase): ODE model
            setting_df (pandas.DataFrame):
                Index reset index
                Columns
                    - (float): parameter values from 0 to 1.0
                    - Rt (float): reproduction number
                    - step_n (int): step number of simulation

        Returns:
            covsirphy.MPEstimator: multi-processor for parameter estimation
        """
        units = []
        # Instance to save theoretical data
        example_data = ExampleData(tau=self.tau, start_date="01Jan2020")
        # Population values
        population = model.EXAMPLE[self.N.lower()]
        population_data = PopulationData(filename=None)
        # Register data for each setting
        for (i, setting_dict) in enumerate(setting_df.to_dict(orient="records")):
            name = f"{model.NAME}_{i}"
            step_n = setting_dict[self.STEP_N]
            param_dict = {
                k: v for (k, v) in setting_dict.items() if k in model.PARAMETERS}
            # Add theoretical data
            example_data.add(
                model, step_n=step_n, country=name, param_dict=param_dict)
            # Population
            population_data.update(population, country=name)
            # Phase unit
            snl = Scenario(
                example_data, population_data, country=name, auto_complement=False)
            snl.add()
            unit = snl[self.MAIN].unit("last").del_id().set_id(country=name)
            units.append(unit)
        # Multi-processor for parameter estimation
        processor = MPEstimator(
            model, jhu_data=example_data, population_data=population_data, tau=self.tau)
        return processor.add(units)

    def _get_result(self, model, setting_df, units_estimated):
        """
        Show the result as a dataframe.

        Args:
            model (covsirphy.ModelBase): ODE model
            setting_df (pandas.DataFrame):
                Index reset index
                Columns
                    - (float): parameter values from 0 to 1.0
                    - Rt (float): reproduction number
                    - step_n (int): step number of simulation
            units_estimated (list[covsirphy.PhaseUnit]): phase units with estimated parameters

        Returns:
            pandas.DataFrame:
                Index reset index
                Columns
                    - ID (str): ID, like SIR_0
                    - ODE (str): model name
                    - Rt (float): reproduction number set by ._setup() method
                    - Rt_est (float): estimated reproduction number
                    - rho etc. (float): parameter values set by ._setup() method
                    - rho_est etc. (float): estimated parameter values
                    - step_n (int): step number of simulation
                    - RMSLE (float): RMSLE score of parameter estimation
                    - Trials (int): the number of trials in parameter estimation
                    - Runtime (str): runtime of parameter estimation
        """
        df = setting_df.copy()
        df["results"] = [unit.to_dict() for unit in units_estimated]
        df = df.join(
            df["results"].apply(pd.Series), how="left", rsuffix="_est")
        df = df.drop(["results", *model.DAY_PARAMETERS], axis=1)
        cols_to_alternate = [self.RT, *model.PARAMETERS]
        cols_alternated = chain.from_iterable(
            zip(cols_to_alternate, [f"{col}_est" for col in cols_to_alternate]))
        df["ID"] = df[self.ODE].str.cat(df.index.astype("str"), sep="_")
        columns = [
            "ID", self.ODE, *cols_alternated, self.STEP_N, *self.EST_COLS]
        return df.loc[:, columns]

    def summary(self):
        """
        Show the summary of validation.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - ID (str): ID, like SIR_0
                    - ODE (str): model name
                    - Rt (float): reproduction number set by ._setup() method
                    - Rt_est (float): estimated reproduction number
                    - rho etc. (float): parameter values set by ._setup() method
                    - rho_est etc. (float): estimated parameter values
                    - step_n (int): step number of simulation
                    - RMSLE (float): RMSLE score of parameter estimation
                    - Trials (int): the number of trials in parameter estimation
                    - Runtime (str): runtime of parameter estimation
        """
        df = pd.concat(self._results, ignore_index=True, sort=True)
        pre_cols = ["ID", self.ODE, self.RT, f"{self.RT}_est"]
        post_cols = [self.STEP_N, *self.EST_COLS]
        centers = list(set(df.columns) - set(pre_cols) - set(post_cols))
        centers_sorted = sorted(
            centers, key=lambda x: df.columns.tolist().index(x))
        return df[[*pre_cols, *centers_sorted, *post_cols]]
