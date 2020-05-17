#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.cleaning.word import Word
from covsirphy.cleaning.jhu_data import JHUData


class EstimatorNew(Word):
    """
    Hyperparameter estimation of ODE models.
    """
    # TODO: Rename to Estimator

    def __init__(self, jhu_data, country, province=None):
        """
        @jhu_data <cs.JHUData>: dataset of the number of cases
        """
        if not isinstance(jhu_data, JHUData):
            raise TypeError("jhu_data must be <covsirphy.JHUData>.")
        subset_df = jhu_data.subset(country=country, province=province)
        self.record_df = self.nondim_data(subset_df)

    def nondim_data(self, subset_df):
        """
        Return the subset dataframe to an nondim_data dataframe.
        @subset_df <pd.DataFrme>: subset dataframe of JHU data
        @return:
            - <pd.DataFrame>
            - 
        """
        df = subset_df.copy()
        # Elapsed time [min]
        start_date = df[self.DATE].min()
        df[self.T] = (df[self.DATE] - start_date).dt.total_seconds() / 60
        df[self.T] = df[self.T].astype(np.int64)
        # TODO: Calculate Susceptible with population value
        # This must be for each model
        # Total population (stable in a phase) must be applied as an integer
        # TODO: Return start_date
        return df
