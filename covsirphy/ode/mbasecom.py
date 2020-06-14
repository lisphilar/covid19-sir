#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.cleaning.word import Word


class ModelBaseCommon(Word):
    # Quartile range of the parametes when setting initial values
    QUANTILE_RANGE = [0.3, 0.7]
    # Model name
    NAME = "ModelBaseCommon"

    def __init__(self):
        # Dictionary of non-dim parameters: {name: value}
        self.non_param_dict = dict()

    def __str__(self):
        return self.NAME

    def __repr__(self):
        if not self.non_param_dict:
            return self.NAME
        param_str = ", ".join(
            [f"{p}={v}" for (p, v) in self.non_param_dict.items()]
        )
        return f"{self.NAME} model with {param_str}"

    def __getitem__(self, key):
        """
        @key <str>: parameter name
        """
        if key not in self.non_param_dict.keys():
            raise KeyError(f"key must be in {', '.join(self.PARAMETERS)}")
        return self.non_param_dict[key]

    @classmethod
    def calc_elapsed(cls, cleaned_df):
        """
        Calculate elapsed time from the first date.
        @cleaned_df <pd.DataFrame>: cleaned data
            - index (Date) <pd.TimeStamp>: Observation date
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        @return <pd.DataFrame>
            - index (Date) <pd.TimeStamp>: Observation date
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
            - Elapsed <int>: Elapsed time from the first date [min]
        """
        df = cleaned_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index of @cleaned_df must be <pd.DatetimeIndex>")
        if set(df.columns) != set(cls.VALUE_COLUMNS):
            cols_str = ", ".join(cls.VALUE_COLUMNS)
            raise KeyError(f"@cleaned_df must has {cols_str} columns.")
        # Calculate elapsed time from the first date [min]
        df[cls.T] = (df.index - df.index.min()).total_seconds()
        df[cls.T] = (df[cls.T] // 60).astype(np.int64)
        return df

    @classmethod
    def validate_tau_free(cls, tau_free_df):
        """
        Validate tau-free dataset and return itself.
        @tau_free_df <pd.DataFrame>:
            - columns: t and dimensional variables
            - dimensional variables are defined by model.VARIABLES
        @return <pd.DataFrame>
        """
        df = tau_free_df.copy()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("@tau_free_df must be a instance of <pd.DataFrame>")
        if not set(cls.VARIABLES).issubset(set(cls.TS) + set(df.columns)):
            cols_str = ', '.join(list(df.columns))
            raise KeyError(f"@tau_free_df must have {cols_str} columns.")
        return df
