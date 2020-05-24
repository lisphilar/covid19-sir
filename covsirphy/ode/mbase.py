#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.cleaning.word import Word


class ModelBase(Word):
    # Quantile range of the parametes when setting initial values
    QUANTILE_RANGE = [0.3, 0.7]
    # Model name
    NAME = "Model"
    # Variables in non-dimensional ODEs
    VARIABLES = ["x"]
    # Priorities of the varialbles when optimization
    PRIORITIES = np.array([1])
    # Variables that increases monotonically
    VARS_INCLEASE = ["x"]

    @classmethod
    def param(cls, train_df_divided=None, q_range=None):
        """
        Define parameters without tau.
        This function should be overwritten in subclass.
        @train_df_divided <pd.DataFrame>:
            - column: t and non-dimensional variables
        @q_range <list[float, float]>:
            quantile rage of the parameters calculated by the data
        @return <dict[name]=(min, max):
            @min <float>: min value
            @max <float>: max value
        """
        # TODO: revise
        param_dict = dict()
        return param_dict

    @classmethod
    def calc_variables(cls, cleaned_df, population):
        """
        Calculate the variables of the model.
        This function should be overwritten in subclass.
        @cleaned_df <pd.DataFrame>: cleaned data
            - index (Date) <pd.TimeStamp>: Observation date
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        @population <int>: total population in the place
        @return <pd.DataFrame>
            - index (Date) <pd.TimeStamp>: Observation date
            - Elapsed <int>: Elapsed time from the start date [min]
            - x, y, z, w etc.
                - calculated in child classes.
                - non-dimensionalized variables of Susceptible etc.
        """
        df = cleaned_df.copy()
        if set(df.columns) != set(cls.VALUE_COLUMNS):
            cols_str = ", ".join(cls.VALUE_COLUMNS)
            raise KeyError(f"@cleaned_df must has {cols_str} columns.")
        # Calculate Susceptible
        df[cls.S] = population - df[cls.C]
        # Calculate elapsed time from the start date [min]
        df[cls.T] = (df.index - df.index.min()).total_seconds()
        df[cls.T] = (df[cls.T] // 60).astype(np.int64)
        return df

    @classmethod
    def nondim_cols(cls, target_df, columns, population):
        """
        Non-dimentionalize the columns with population value.
        @target_df <pd.DataFrame>: dataframe with dimention
            - Elapsed: elpased time from the start date [min]
            - X, Y etc. (defined by @column, upper cases)
        @columns <list[str]>: list of column (with upper strings)
        @population <int>: total population in the place
        """
        df = target_df.copy()
        cols_lower = [col.lower() for col in columns]
        df[cols_lower] = df[columns] / population
        df = df.loc[:, [cls.T, *cols_lower]]
        return df

    @classmethod
    def calc_variables_reverse(cls, df, total_population):
        """
        Calculate measurable variables using the variables of the model.
        This function should be overwritten in subclass.
        @df <pd.DataFrame>:
            - x, y, z, w etc.
                - calculated in child classes.
                - non-dimensionalized variables of Susceptible etc.
        @total_population <int>: total population
        @return <pd.DataFrame>:
            - The number of cases
        """
        return df

    def calc_r0(self):
        """
        This function should be overwritten in subclass.
        """
        return None

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        This function should be overwritten in subclass.
        @param tau <int>: tau value [hour]
        """
        return dict()
