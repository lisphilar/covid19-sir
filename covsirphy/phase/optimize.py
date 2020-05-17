#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optuna
from covsirphy.cleaning.word import Word


class Optimizer(Word):
    """
    Hyperparameter optimization with Optuna package.
    """

    def __init__(self, train_df, **params):
        """
        @train_df <pd.DataFrame>: training dataset
            - index: reseted index
            - (Explanatory variable)
            - (Response variables)
        @param (keyword arguments): fixed parameters
        """
        self.train_df = train_df.copy()
        self.fixed_dict = params.copy()
        self.study = None

    def run(self, n_trials, n_jobs=-1):
        """
        Run optimization.
        @n_trials <int>: the number of trials.
        @n_jobs <int>: the number of parallel jobs or -1 (CPU count)
        """
        if self.study is None:
            self.study = optuna.create_study(direction="minimize")
        self.study.optimze(
            lambda x: self.objective(x),
            n_trials=n_trials,
            n_jobs=n_jobs
        )

    def objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the the range of parameters.
        This function should be overwritten in subclass.
        @trial <optuna.trial>: a trial of the study
        """
        return self.error_f()

    def error_f(self):
        """
        Definition of error score to minimize in the study.
        This function should be overwritten in subclass.
        """
        return None

    def result(self):
        """
        Return the estimated parameters.
        This function should be overwritten in subclass.
        @return <dict[str]=float/int>
        """
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_dict)
        return param_dict
