#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import optuna
import pandas as pd
from covsirphy.cleaning.word import Word


class Optimizer(Word):
    """
    Hyperparameter optimization with Optuna package.
    """
    A = "_actual"
    P = "_predicted"

    def __init__(self, train_df, x="t", **params):
        """
        @train_df <pd.DataFrame>: training dataset
            - index: reseted index
            - Explanatory variable defined by @x
            - Response variables which is not @x
        @param (keyword arguments): fixed parameter values
        """
        optuna.logging.disable_default_handler()
        self.x = x
        self.y_list = [v for v in train_df.columns if v != x]
        self.train_df = train_df.copy()
        self.y0_dict = train_df.iloc[0, :].to_dict()
        self.step_n = len(train_df)
        self.fixed_dict = params.copy()
        self.study = None
        self.total_trials = 0
        self.run_time = 0

    def run(self, n_trials, n_jobs=-1):
        """
        Run optimization.
        @n_trials <int>: the number of trials.
        @n_jobs <int>: the number of parallel jobs or -1 (CPU count)
        """
        self.total_trials += n_trials
        start_time = datetime.now()
        if self.study is None:
            self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda x: self.objective(x),
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        end_time = datetime.now()
        self.run_time += (end_time - start_time).total_seconds()

    def objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the parameter values using Optuna.
        This function should be overwritten in subclass.
        @trial <optuna.trial>: a trial of the study
        @return <float>: score of the error function to minimize
        """
        param_dict = dict()
        return self.error_f(param_dict, self.train_df)

    def error_f(self, param_dict, train_df):
        """
        Definition of error score to minimize in the study.
        This function should be overwritten in subclass.
        @param_dict <dict[str]=int/float>:
            - estimated parameter values
        @train_df <pd.DataFrame>: actual data
            - index: reseted index
            - t: time step, 0, 1, 2,...
            - includes columns defined by @variables
        @return <float>: score of the error function to minimize
        """
        sim_df = self.simulate(self.step_n, param_dict)
        comp_df = self.compare(self.train_df, sim_df)
        _ = (sim_df, comp_df)
        return None

    def simulate(self, param_dict):
        """
        Simulate the values with the parameters.
        This function should be overwritten in subclass.
        @param_dict <dict[str]=int/float>:
            - estimated parameter values
        @return <pd.DataFrame>:
            - index: rested index
            - Explanatory variable defined by self.x
            - Response variables which defined by self.y_list
        """
        _ = param_dict.copy()
        df = pd.DataFrame(columns=[self.x, *self.y_list])
        return df

    def compare(self, actual_df, predicted_df):
        """
        Return comparison table.
        @actual_df <pd.DataFrame>: actual data
            - index: reseted index
            - t: time step, 0, 1, 2,...
            - includes columns defined by self.y_list
        @predicted_df <pd.DataFrame>: predicted data
            - index: reseted index
            - t: time step, 0, 1, 2,...
            - includes columns defined by self.y_list
        @return <pd.DataFrame>:
            - index: time step
            - columns with "_actual"
            - columns with "_predicted:
                - columns are defined by self.y_list
        """
        # Check the arguments
        if not set(self.y_list).issubset(set(predicted_df.columns)):
            y_str = ", ".join(self.y_list)
            raise KeyError(f"@predicted_df must have {y_str} columns.")
        # Data for comparison
        df = pd.merge(
            actual_df, predicted_df, on=self.x,
            suffixes=(self.A, self.P)
        )
        df = df.set_index(self.x)
        return df

    def param(self):
        """
        Return the estimated parameters as a dictionary.
        @return <dict[str]=float/int>
        """
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_dict)
        return param_dict

    def result(self, name):
        """
        Return the estimated parameters as a dataframe.
        This function should be overwritten in subclass.
        @name <str>: index of the dataframe
        @return <pd.DataFrame>:
            - (estimated parameters)
            - Trials: the number of trials
            - Runtime: run time of estimation
        """
        param_dict = self.param()
        # The number of trials
        param_dict["Trials"] = self.total_trials
        # Runtime
        minutes, seconds = divmod(int(self.run_time), 60)
        param_dict["Runtime"] = f"{minutes} min {seconds} sec"
        return param_dict

    def rmsle(self, train_df, dim=1):
        """
        Calculate RMSLE score.
        @train_df <pd.DataFrame>: actual data
            - index: reseted index
            - t: time step, 0, 1, 2,...
            - includes columns defined by self.y_list
        @dim <int/float>: dimension where comparison will be performed
        @return <float>
        """
        predicted_df = self.predict()
        df = self.compare(train_df, predicted_df)
        df = df * dim + 1
        a_list = [np.log10(df[f"{v}{self.A}"]) for v in self.y_list]
        p_list = [np.log10(df[f"{v}{self.P}"]) for v in self.y_list]
        diffs = [((a - p) ** 2).sum() for (a, p) in zip(a_list, p_list)]
        score = np.sqrt(sum(diffs) / len(diffs))
        return score

    def predict(self):
        """
        Predict the values with the calculated values.
        This function can be overwritten in subclass.
        """
        param_dict = self.param()
        param_dict.pop("tau")
        return self.simulate(self.step_n, param_dict)
