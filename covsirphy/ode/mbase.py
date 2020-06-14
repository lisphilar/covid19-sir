#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbasecom import ModelBaseCommon


class ModelBase(ModelBaseCommon):
    """
    Base class of ODE models.
    """
    # Model name
    NAME = "ModelBase"
    # names of parameters
    PARAMETERS = list()
    # Variable names in dimensional ODEs
    VARIABLES = list()
    # Priorities of the variables when optimization
    PRIORITIES = np.array(list())
    # Variables that increases monotonically
    VARS_INCLEASE = list()

    def __init__(self, population):
        """
        This method should be overwritten in subclass.
        @population <int>: total population
        """
        # Total population
        if not isinstance(population, int):
            raise TypeError("@population must be an integer.")
        self.population = population
        # Non-dim parameters

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.
        This method should be overwritten in subclass.
        @return <np.array>
        """
        return np.array(list())

    @classmethod
    def param_range(cls, ode_df):
        """
        Define the range of parameters (not including tau value).
        This function should be overwritten in subclass.
        @ode_df <pd.DataFrame>:
            - columns: t and dimensional variables
            - dimensional variables are defined by model.VARIABLES
        @return <dict[name]=(min, max)>:
            - min <float>: min value
            - max <float>: max value
        """
        _ = cls.validate_ode(ode_df)
        _dict = dict()
        return _dict

    @classmethod
    def specialize(cls, data_df, population):
        """
        Specialize the dataset for this model.
        This method should be overwritten in subclass.
        @data_df <pd.DataFrame>:
            - index: reset index
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
            - any columns
        @population <int>: total population in the place
        @return <pd.DataFrame>:
            - index: reset index
            - any columns @data_df has
            - columns with dimensional variables
        """
        df = cls.validate_dataframe(
            data_df, name="data_df", columns=cls.VALUE_COLUMNS
        )
        return df

    def calc_r0(self):
        """
        This method should be overwritten in subclass.
        """
        return None

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.
        This method should be overwritten in subclass.
        @param tau <int>: tau value [min]
        """
        return dict()
