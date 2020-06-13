#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from covsirphy.phase.phase_data import PhaseData
from covsirphy.ode.mbase import ModelBase


class NondimData(PhaseData):
    """
    Create dataset for ODE analysis.
    """
    # TODO: not use non-dimensional data

    def __init__(self, clean_df, country=None, province=None):
        super().__init__(clean_df, country=country, province=province)

    def _make(self, grouped_df, model, population):
        """
        Make non-dimensional dataset for an ODE model.
        @grouped_df <pd.DataFrame>: cleaned data grouped by Date
            - index (Date) <pd.TimeStamp>: Observation date
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        @model <sub-class of cs.ModelBase>: ODE model
        @population <int>: total population in the place
        @return <pd.DataFrame>
            - index (Date) <pd.TimeStamp>: Observation date
            - Elapsed <int>: Elapsed time from the start date [min]
            - x, y, z, w etc.
                - calculated in child classes.
                - non-dimensionalized variables of Susceptible etc.
        """
        df = grouped_df.copy()
        if set(df.columns) != set(self.VALUE_COLUMNS):
            cols_str = ", ".join(self.VALUE_COLUMNS)
            raise KeyError(f"@cleaned_df must has {cols_str} columns.")
        if not issubclass(model, ModelBase):
            raise TypeError(
                "@model must be an ODE model <sub-class of cs.ModelBase>."
            )
        return model.calc_variables(df, population)

    def make(self, model, population, start_date=None, end_date=None):
        """
        Make non-dimensional dataset for an ODE model.
        @model <covsirphy.ModelBase>: ODE model
        @population <int>: total population in the place
        @start_date <str>: start date, like 22Jan2020
            - if None, the first date of the records will be used
        @end_date <str>: end date, like 01Feb2020
            - if None, the last date of the records will be used
        @return <pd.DataFrame>
            - index (Date) <pd.TimeStamp>: Observation date
            - Elapsed <int>: Elapsed time from the start date [min]
            - x, y, z, w etc.
                - calculated in child classes.
                - non-dimensionalized variables of Susceptible etc.
        """
        df = self.all_df.copy()
        series = df.index.copy()
        # Start date
        if start_date is None:
            start_obj = series.min()
        else:
            start_obj = datetime.strptime(start_date, self.DATE_FORMAT)
        # End date
        if end_date is None:
            end_obj = series.max()
        else:
            end_obj = datetime.strptime(end_date, self.DATE_FORMAT)
        # Subset
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        return self._make(df, model, population)
