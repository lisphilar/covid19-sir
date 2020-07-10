#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.phase.phase_data import PhaseData
from covsirphy.ode.mbase import ModelBase


class ODEData(PhaseData):
    """
    Dataset for ODE analysis.

    Args:
        clean_df (pandas.DataFrame): cleaned data

            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Country (str): country/region name
                - Province (str): province/prefecture/sstate name
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
        country (str): country name
        province (str): province name
    """

    def __init__(self, clean_df, country=None, province=None):
        super().__init__(clean_df, country=country, province=province)

    def _calc_elapsed(self, grouped_df):
        """
        Calculate elapsed time from the first date.

        Args:
            grouped_df (pandas.DataFrame): cleaned data grouped by Date

                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Elapsed (int): Elapsed time from the first date [min]
        """
        df = grouped_df.copy()
        df[self.COUNTRY] = self.country
        df[self.PROVINCE] = self.province
        df = self.validate_dataframe(
            df, name="grouped_df", time_index=True, columns=self.VALUE_COLUMNS
        )
        # Calculate elapsed time from the first date [min]
        df[self.T] = (df.index - df.index.min()).total_seconds()
        df[self.T] = (df[self.T] // 60).astype(np.int64)
        df = df.reset_index(drop=True)
        return df

    def _tau_free(self, elapsed_df, tau):
        """
        Create tau-free dataset.

        Args:
            elapsed_df (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Elapsed (int): Elapsed time from the first date [min]
        tau (int): tau value [min]

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - t (int): Elapsed time divided by tau value [-]
        """
        df = self.validate_dataframe(
            elapsed_df,
            name="elapsed_df", columns=[*self.VALUE_COLUMNS, self.T]
        )
        if not isinstance(tau, int) or tau < 1 or tau > 1440:
            raise TypeError("@tau must be an integer and 1 <= @tau <= 1440.")
        df[self.TS] = (df[self.T] / tau).astype(np.int64)
        df = df.drop(self.T, axis=1)
        return df

    def _specialize(self, data_df, model, population):
        """
        Specialize the dataset for the model.

        Args:
            data_df (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): total population in the place

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Elapsed (int): if included in @data_df
                    - t (int): if included in @data_df
                    - columns with dimensional variables
        """
        model = self.validate_subclass(model, ModelBase, name="model")
        df = model.specialize(data_df, population)
        time_cols = list()
        if self.T in data_df.columns:
            time_cols.append(self.T)
        if self.TS in data_df.columns:
            time_cols.append(self.TS)
        return df.loc[:, [*time_cols, *model.VARIABLES]]

    def _make(self, grouped_df, model, population, tau):
        """
        Create tau-free dataset for ODE analysis.

        Args:
            grouped_df (pandas.DataFrame): cleaned data grouped by Date

                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            model (covsirphy.ModelBase): ODE model
            population (int): total population
            tau (int or None): tau value [min] or None (un-set)

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Elapsed (int): if tau is None
                    - t (int): if tau is not None
                    - columns with dimensional variables
        """
        df = self._calc_elapsed(grouped_df)
        if tau is not None:
            df = self._tau_free(df, tau)
        df = self._specialize(df, model, population)
        return df

    def make(self, model, population, start_date=None, end_date=None, tau=None):
        """
        Make a dataset for ODE analysis.

        Args:
            model (covsirphy.ModelBase): ODE model
            population (int): total population
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            tau (int or None): tau value [min] or None (un-set)

        Notes:
            - When @start_date is None, the first date of the records will be used
            - When @end_date is None, the last date of the records will be used

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Elapsed (int): if tau is None
                    - t (int): if tau is not None
                    - columns with dimensional variables
        """
        df = self.subset(start_date=start_date, end_date=end_date)
        return self._make(df, model, population, tau)

    def y0(self, model, population, start_date=None):
        """
        Return the initial values of the model.

        Args:
            model (covsirphy.ModelBase): ODE model
            population (int): total population
            start_date (str): start date, like 22Jan2020

        Notes:
            - When @start_date is None, the first date of the records will be used

        Returns:
            (dict)
                - key (str): dimensional variables
                - value (int): the number of cases
        """
        subset_df = self.subset(start_date=start_date, end_date=None)
        df = self._make(subset_df, model, population, tau=None)
        y0_dict = {
            k: df.loc[df.index[0], k] for k in model.VARIABLES
        }
        return y0_dict
