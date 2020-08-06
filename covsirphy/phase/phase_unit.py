#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.estimator import Estimator


class PhaseUnit(Term):
    """
    Save information of  a phase.

    Args:
        start_date (str): start date of the phase
        end_date (str): end date of the phase
        population (int): population value
    """

    def __init__(self, start_date, end_date, population):
        start = self.date_obj(start_date)
        end = self.date_obj(end_date)
        if start >= end:
            raise ValueError(
                f"@end_date ({end_date}) must be over @start_date ({start_date}).")
        self.start_date = start_date
        self.end_date = end_date
        self.population = self.ensure_natural_int(
            population, name="population")
        self.ode_dict = {}

    def summary(self):
        """
        Summarize information.

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns:
                    - Start: start date of the phase
                    - End: end date of the phase
                    - Population: population value of the start date
                    - if available:
                        - ODE: model name
                        - parameter values if available
                        - tau: tau value [min]
                        - Rt: (basic) reproduction number
                        - day parameter values if available
                        - RMSLE: RMSLE value of estimation
                        - Trials: the number of trials in estimation
                        - Runtime: runtime of estimation
        """
        summary_dict = {
            self.START: [self.start_date],
            self.END: [self.end_date],
            self.N: [self.population]
        }
        summary_dict.update(self.ode_dict)
        return pd.DataFrame(summary_dict, orient="index")

    def estimate(self, model, record_df, **kwargs):
        """
        Perform parameter estimation.

        Args:
            model (covsirphy.ModelBase): ODE model
            record_df (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases, if calculated
                    - any other columns will be ignored
            kwargs: keyword arguments of model parameters, tau and covsirphy.Estimator.run()
        """
        # Arguments
        self.ensure_subclass(model, ModelBase, name="model")
        self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # Parameter estimation of ODE model
        estimator = Estimator(record_df, model, self.population, **kwargs)
        estimator.run(**kwargs)
        est_dict = estimator.summary().to_dict()
        self.ode_dict.update(est_dict)
