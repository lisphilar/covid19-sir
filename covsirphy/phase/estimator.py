#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.nondim_data import NondimData
from covsirphy.phase.optimize import Optimizer


class Estimator(Optimizer):
    """
    Hyperparameter optimization of an ODE model.
    """

    def __init__(self, clean_df, model, country=None, province=None, **kwargs):
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
        @country <str>: country name
        @province <str>: province name
        @kwargs: the other keyword arguments of NondimData.make()
            - @population <int>: total population in the place
            - @start_date <str>: start date, like 22Jan2020
            - @end_date <str>: end date, like 01Feb2020
        """
        super().__init__()
        if not issubclass(model, ModelBase):
            raise TypeError(
                "@model must be an ODE model <sub-class of cs.ModelBase>."
            )
        self.model = model
        nondim_data = NondimData(
            clean_df, country=country, province=province
        )
        self._train_df = nondim_data.make(model=model, **kwargs)

    @property
    def train_df(self):
        """
        Return the train data.
        @return <pd.DataFrame>:
            - index (Date) <pd.TimeStamp>: Observation date
            - Elapsed <int>: Elapsed time from the start date [min]
            - x, y, z, w etc.
                - calculated in child classes.
                - non-dimensionalized variables of Susceptible etc.
        """
        return self._train_df

    def objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the the range of parameters.
        @trial <optuna.trial>: a trial of the study
        """
        # Convert T to t using tau
        if "tau" in self.fixed_dict.keys():
            tau = self.fixed_dict["tau"]
        else:
            tau = trial.suggest_int("tau", 1, 1440)
        df = self.train_df.copy()
        df["t"] = (df["T"] / tau).astype(np.int64)
        df = df.drop("T", axis=1)
        # Set parameters of the models
        p_dict = {"tau": None}
        model_param_dict = self.model.param_dict(train_df_divided=df)
        p_dict.update(
            {
                k: trial.suggest_uniform(k, *v)
                for (k, v) in model_param_dict.items()
            }
        )
        p_dict.update(self.fixed_dict)
        p_dict.pop("tau")
        return self.error_f(p_dict, df)

    def error_f(self, param_dict):
        """
        Definition of error score to minimize in the study.
        @param_dict <dict[str]=float/int>: dictionary of parameters
        """
        # TODO
        return None

    def result(self):
        """
        Return the estimated parameters.
        This function should be overwritten in subclass.
        @return <dict[str]=float/int>
        """
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_dict)
        # TODO
        return param_dict
