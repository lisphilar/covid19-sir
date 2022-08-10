#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.scenario import Scenario
from covsirphy._deprecated.example_data import ExampleData


class ModelValidator(Term):
    """
    Deprecated. Evaluate ODE models performance as follows.
    1. Select model parameter sets randomly
    2. Set user-defined/random phase duration
    3. Perform simulation with a specified ODE model
    4. Perform parameter estimation
    5. Compare the estimated parameters and the parameters produced with th 1st step
    6. Repeat trials (1 trial = from the 1st step to the 5th step)
    Small difference is expected in the 6th step.

    Args:
        tau (int): tau value [min]
        n_trials (int): the number of trials
        step_n (int or None): the number of steps in simulation (over 2) or None (randomly selected)
        seed (int): random seed

    Note:
        Population value and initial values are defined by model.EXAMPLE.
        Estimators know tau values before parameter estimation.
    """

    @deprecate(old="ModelValidator()", version="2.24.0-kappa")
    def __init__(self, tau=1440, n_trials=8, step_n=None, seed=0):
        self._tau = Validator(tau, "tau").tau(default=None)
        self._n_trials = Validator(n_trials, "n_trials").int(value_range=(1, None))
        self._seed = Validator(seed, "seed").int(value_range=(0, None))
        # list[int]: the number of steps for each trials
        if step_n is None:
            np.random.seed(seed)
            self._step_n_list = np.random.randint(5, 60, n_trials)
        else:
            Validator(step_n, "step_n").int(value_range=(3, None))
            self._step_n_list = [step_n for _ in range(n_trials)]
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
        model = Validator(model, "model").subclass(ModelBase)
        if model.NAME in self.model_names:
            raise ValueError(f"{model.NAME} has been validated.")
        self.model_names.append(model.NAME)
        # Setup: create parameter set, phase length and processor
        df = self._setup(model)
        # Parameter estimation
        scenarios = self._processor(model, df, timeout=timeout, allowance=allowance)
        # Get estimated parameters
        self._results.append(self._get_result(model, df, scenarios))
        return self

    def _setup(self, model):
        """
        Create a model parameter set randomly.

        Args:
            model (covsirphy.ModelBase): ODE model

        Returns:
            pandas.DataFrame:
                Index reset index
                Columns
                    - (float): parameter values from 0 to 1.0
                    - Rt (float): reproduction number
                    - step_n (int): step number of simulation
        """
        np.random.seed(self._seed)
        population = model.EXAMPLE[self.N.lower()]
        # Parameter set
        parameters = model.PARAMETERS[:]
        df = pd.DataFrame(np.random.rand(self._n_trials, len(parameters)), columns=parameters)
        # Reproduction number
        df[self.RT] = df.apply(lambda x: model(population, **x.to_dict()).calc_r0(), axis=1)
        # Tau value
        df[self.TAU] = self._tau
        # Step number
        df[self.STEP_N] = self._step_n_list
        # Return the setting
        return df

    def _processor(self, model, setting_df, timeout, allowance):
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
            timeout (int): time-out of run
            allowance (tuple(float, float)): the allowance of the predicted value

        Returns:
            list[covsirphy.Scenario]: list of Scenario instances
        """
        scenarios = []
        # Instance to save theoretical data
        example_data = ExampleData(tau=self._tau, start_date="01Jan2020")
        # Population values
        population = model.EXAMPLE[self.N.lower()]
        # Register data for each setting
        for (i, setting_dict) in enumerate(setting_df.to_dict(orient="records")):
            name = f"{model.NAME}_{i}"
            step_n = setting_dict[self.STEP_N]
            param_dict = {k: v for (k, v) in setting_dict.items() if k in model.PARAMETERS}
            # Add theoretical data
            example_data.add(
                model, step_n=step_n, country=name, param_dict=param_dict, population=population)
            # Phase unit
            snl = Scenario(country=name, auto_complement=False)
            snl.register(example_data)
            snl.add()
            snl.estimate(model, timeout=timeout, allowance=allowance)
            scenarios.append(snl)
        return scenarios

    def _get_result(self, model, setting_df, scenarios):
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
            list[covsirphy.Scenario]: list of Scenario instances

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
        results_df = pd.concat([snl.summary() for snl in scenarios], axis=0)
        df = df.join(results_df.reset_index(drop=True), how="left", rsuffix="_est")
        df = df.drop(model.DAY_PARAMETERS, axis=1, errors="ignore")
        cols_to_alternate = [self.RT, *model.PARAMETERS]
        cols_alternated = chain.from_iterable(
            zip(cols_to_alternate, [f"{col}_est" for col in cols_to_alternate]))
        df["ID"] = df[self.ODE].str.cat(df.index.astype("str"), sep="_")
        columns = ["ID", self.ODE, *cols_alternated, self.STEP_N, "RMSLE", self.TRIALS, self.RUNTIME]
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
        post_cols = [self.STEP_N, "RMSLE", self.TRIALS, self.RUNTIME]
        centers = list(set(df.columns) - set(pre_cols) - set(post_cols))
        centers_sorted = sorted(centers, key=lambda x: df.columns.tolist().index(x))
        return df[[*pre_cols, *centers_sorted, *post_cols]]
