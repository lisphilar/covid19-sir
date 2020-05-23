#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import optuna
import pandas as pd
from covsirphy.analysis.simulator import Simulator
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.nondim import NondimData
from covsirphy.phase.optimize import Optimizer


class Estimator(Optimizer):
    """
    Hyperparameter optimization of an ODE model.
    """

    def __init__(self, clean_df, model, population,
                 country, province=None, **kwargs):
        """
        @clean_df <pd.DataFrame>: cleaned data
            - index <int>: reseted index
            - Date <pd.TimeStamp>: Observation date
            - Country <str>: country/region name
            - Province <str>: province/prefecture/sstate name
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        @model <subclass of cs.ModelBase>: ODE model
        @population <int>: total population in the place
        @country <str>: country name
        @province <str>: province name
        @kwargs: the other keyword arguments of NondimData.make()
            - @start_date <str>: start date, like 22Jan2020
            - @end_date <str>: end date, like 01Feb2020
        """
        optuna.logging.disable_default_handler()
        if not issubclass(model, ModelBase):
            raise TypeError(
                "@model must be an ODE model <sub-class of cs.ModelBase>."
            )
        self.model = model
        self.population = population
        self.country = country
        self.province = province
        nondim_data = NondimData(
            clean_df, country=country, province=province
        )
        self.min_train_df = nondim_data.make(
            model=model, population=population, **kwargs
        )
        self.y0_dict = self.min_train_df.iloc[0, :].to_dict()
        self.train_df = None
        self.x = self.TS
        self.y_list = model.VARIABLES[:]
        self.fixed_dict = kwargs.copy()
        self.study = None
        self.total_trials = 0
        self.run_time = 0
        # step_n will be defined in divide_minutes()
        self.step_n = None

    def run(self, timeout=120, n_jobs=-1,
            timeout_iteration=30, allowance=(0.8, 1.2)):
        """
        Run optimization.
        Maximum values:
            - @timeout <int>: time-out of run
        @n_jobs <int>: the number of parallel jobs or -1 (CPU count)
        @timeout <int>: time-out of one iteration
            - if the result satisfied all conditions, optimization ends
                - values of monotonic increasing variables increases monotonically
                - the last value of predicted values are in the allowance
                    for the variables that are increasing monotonically
        @allowance <tuple(float, float)>:
            - the allowance of the predicted value
        @return self
        """
        if self.study is None:
            self.study = optuna.create_study(direction="minimize")
        print("\tRunning optimization...")
        start_time = datetime.now()
        while True:
            # Perform optimization
            self.study.optimize(
                lambda x: self.objective(x),
                n_jobs=n_jobs,
                timeout=timeout_iteration
            )
            end_time = datetime.now()
            self.run_time = (end_time - start_time).total_seconds()
            self.total_trials = len(self.study.trials)
            # Show run-time
            minutes, seconds = divmod(int(self.run_time), 60)
            print(
                f"\r\tPerformed {self.total_trials} trials in {minutes} min {seconds} sec",
                end=str()
            )
            # Time-out
            if self.run_time >= timeout:
                break
            # Create a table to compare observed/estimated values
            tau = super().param()["tau"]
            train_df = self.divide_minutes(tau)
            comp_df = self.compare(train_df, self.predict())
            # Check monotonic variables
            mono_ok_list = [
                comp_df[f"{v}{self.P}"].is_monotonic_increasing
                for v in self.model.MONOTONIC_INCREASE
            ]
            if not all(mono_ok_list):
                continue
            # Check the last value
            last_value_nest = [
                comp_df.iloc[-1, :][[f"{v}{self.A}", f"{v}{self.P}"]].tolist()
                for v in self.model.MONOTONIC_INCREASE
            ]
            last_ok_list = [
                (a * allowance[0] <= p) and (p <= a * allowance[1])
                for (a, p) in last_value_nest
            ]
            if not all(last_ok_list):
                continue
            break
        minutes, seconds = divmod(int(self.run_time), 60)
        print(
            f"\r\tFinished {self.total_trials} trials in {minutes} min {seconds} sec.",
            end="\n"
        )
        return self

    def objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the parameter values using Optuna.
        @trial <optuna.trial>: a trial of the study
        @return <float>: score of the error function to minimize
        """
        # Convert T to t using tau
        if "tau" in self.fixed_dict.keys():
            tau = self.fixed_dict["tau"]
        else:
            tau = trial.suggest_int("tau", 1, 1440)
        if not isinstance(tau, int) or tau <= 0:
            raise TypeError(
                f"@tau must be a non-negative integer, but {tau} was applied."
            )
        train_df = self.divide_minutes(tau)
        # Set parameters of the models
        p_dict = {"tau": None}
        model_param_dict = self.model.param(train_df_divided=train_df)
        p_dict.update(
            {
                k: trial.suggest_uniform(k, *v)
                for (k, v) in model_param_dict.items()
            }
        )
        p_dict.update(self.fixed_dict)
        p_dict.pop("tau")
        return self.error_f(p_dict, train_df)

    def divide_minutes(self, tau):
        """
        Devide T by tau in the training dataset.
        @tau <int>: tau value [min]
        """
        if not isinstance(tau, int) or tau <= 0:
            raise TypeError(
                f"@tau must be a non-negative integer, but {tau} was applied."
            )
        df = self.min_train_df.copy()
        df[self.TS] = (df[self.T] / tau).astype(np.int64)
        train_df = df.drop(self.T, axis=1).reset_index(drop=True)
        self.step_n = int(train_df[self.TS].max())
        return train_df

    def error_f(self, param_dict, train_df):
        """
        Definition of error score to minimize in the study.
        @param_dict <dict[str]=int/float>:
            - estimated parameter values
        @train_df <pd.DataFrame>: training dataset
            - index: reseted index
            - t: time steps
            - x, y, z, w etc.
        @return <float>: score of the error function to minimize
        """
        if self.step_n is None:
            raise ValueError("self.step_n must be done.")
        sim_df = self.simulate(self.step_n, param_dict)
        df = self.compare(train_df, sim_df)
        df = (df * self.population + 1).astype(np.int64)
        # Calculate error score
        v_list = self.model.VARIABLES[:]
        diffs = [df[f"{v}{self.A}"] - df[f"{v}{self.P}"] for v in v_list]
        numerators = [df[f"{v}{self.A}"] for v in v_list]
        np.seterr(divide="raise")
        try:
            scores = [
                p * np.average(diff.abs() / numerator, weights=df.index)
                for (p, diff, numerator)
                in zip(self.model.PRIORITIES, diffs, numerators)
                if p != 0
            ]
        except ZeroDivisionError:
            return np.inf
        return sum(scores)

    def simulate(self, step_n, param_dict):
        """
        Simulate the values with the parameters.
        @step_n <int>: number of iteration
        @param_dict <dict[str]=int/float>:
            - estimated parameter values
        @return <pd.DataFrame>:
            - index: reseted index
            - t: time steps, 0, 1, 2, 3...
            - x, y, z, w etc.
        """
        simulator = Simulator(country=self.country, province=self.province)
        simulator.add(
            model=self.model,
            step_n=step_n,
            population=self.population,
            param_dict=param_dict,
            y0_dict=self.y0_dict
        )
        simulator.run()
        df = simulator.non_dim()
        return df

    def summary(self, name):
        """
        Summarize the results of optimization.
        This function should be overwritten in subclass.
        @name <str>: index of the dataframe
        @return <pd.DataFrame>:
            - index (@name)
            - (parameters of the model)
            - tau
            - Rt: basic or phase-dependent reproduction number
            - (dimensional parameters [day])
            - RMSLE: Root Mean Squared Log Error
            - Trials: the number of trials
            - Runtime: run time of estimation
        """
        param_dict = super().param()
        model_params = param_dict.copy()
        tau = model_params.pop("tau")
        model_instance = self.model(**model_params)
        # Rt
        param_dict["Rt"] = model_instance.calc_r0()
        # dimensional parameters [day]
        param_dict.update(model_instance.calc_days_dict(tau))
        # RMSLE
        param_dict["RMSLE"] = self.rmsle(tau)
        # The number of trials
        param_dict["Trials"] = self.total_trials
        # Runtime
        minutes, seconds = divmod(int(self.run_time), 60)
        param_dict["Runtime"] = f"{minutes} min {seconds} sec"
        # Convert to dataframe
        df = pd.DataFrame.from_dict({name: param_dict}, orient="index")
        return df.fillna("-")

    def rmsle(self, tau):
        """
        Return RMSLE score.
        @tau <int>: tau value
        """
        score = super().rmsle(
            train_df=self.divide_minutes(tau),
            dim=self.population
        )
        return score

    def accuracy(self, filename=None):
        """
        Show the accuracy as a figure.
        @filename <str>: filename of the figure, or None (show figure)
        """
        tau = super().param()["tau"]
        train_df = self.divide_minutes(tau)
        use_variables = [
            v for (i, (p, v))
            in enumerate(zip(self.model.PRIORITIES, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        df = super().accuracy(
            train_df=train_df,
            variables=use_variables,
            filename=filename
        )
        return df
