#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from covsirphy.cleaning.word import Word
from covsirphy.ode.mbase import ModelBase


class NondimData(Word):
    """
    Create dataset for ODE analysis.
    """

    def __init__(self, clean_df, country=None, province=None):
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
        @country <str>: country name
        @province <str>: province name
        """
        df = self._set_place(
            clean_df, country=country, province=province
        )
        self.all_df = self._groupby_date(df)

    def _set_place(self, clean_df, country=None, province=None):
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
        @country <str>: country name
        @province <str>: province name
        @return <pd.DataFrme>:
            - index <int>: reseted index
            - Date <pd.TimeStamp>: Observation date
            - Country <str>: country/region name
            - Province <str>: province/prefecture/sstate name
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        """
        df = clean_df.copy()
        c, p = country, province
        if c is None:
            if p is not None:
                return df
            s = "@country must be defined when @province is not None."
            raise Exception(s)
        if p is None:
            return df.loc[df[self.COUNTRY] == c, :]
        return df.loc[(df[self.COUNTRY] == c) & (df[self.PROVINCE] == p), :]

    def _groupby_date(self, cleaned_df):
        """
        Grouping by date.
        @clean_df <pd.DataFrame>: cleaned data
            - index <int>: reseted index
            - Date <pd.TimeStamp>: Observation date
            - Country <str>: country/region name
            - Province <str>: province/prefecture/sstate name
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        @return <pd.DataFrme>:
            - index (Date) <pd.TimeStamp>: Observation date
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        """
        df = cleaned_df.copy()
        df = df.drop([self.COUNTRY, self.PROVINCE], axis=1)
        df = df.groupby(self.DATE).sum()
        return df

    def _make(self, grouped_df, model, population):
        """
        Make nondim_data dataset for an ODE model using the cleaned dataset.
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
        Make nondim_data datssrt for an ODE model.
        @model <cs.ModelBase>: ODE model
        @population <int>: total population in the place
        @start_date <str>: start date, like 22Jan2020
        @end_date <str>: end date, like 01Feb2020
        @return <pd.DataFrame>
            - index (Date) <pd.TimeStamp>: Observation date
            - T <int>: Elapsed time from the start date [min]
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
        # subset
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        return self._make(df, model, population)
