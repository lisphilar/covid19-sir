#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.simulator import ODESimulator


class ExampleData(JHUData):
    """
    Example dataset as a child class of JHUData.

    Args:
        clean_df (pandas.DataFrame): cleaned data

            Index:
                - reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Country (str): country/region name
                - Province (str): province/prefecture/sstate name
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases

        tau (int): tau value [min]
        start_date (str): start date, like 22Jan2020
    """

    def __init__(self, clean_df=None, tau=1440, start_date="22Jan2020"):
        if clean_df is None:
            clean_df = pd.DataFrame(columns=self.COLUMNS)
        clean_df = self.ensure_dataframe(
            clean_df, name="clean_df", columns=self.COLUMNS)
        self._raw = clean_df.copy()
        self._cleaned_df = clean_df.copy()
        self._citation = str()
        self.tau = self.ensure_tau(tau)
        self.start_date = self.ensure_date(start_date, name="start_date")
        self._specialized_dict = {}
        self.nondim_dict = {}

    def _model_to_area(self, model=None, country=None, province=None):
        """
        If country is None and model is not None, model name will returned as country name.

        Args:
            model (cs.ModelBase or None): the first ODE model
            country (str or None): country name
            province (str or None): province name

        Raises:
            ValueError: both of country and model are None

        Returns:
            tuple(str, str): country name and province name
        """
        province = province or self.UNKNOWN
        if country is not None:
            return (country, province)
        if model is None:
            raise ValueError("@model or @country must be specified.")
        model = self.ensure_subclass(model, ModelBase, name="model")
        return (model.NAME, province)

    def add(self, model, country=None, province=None, **kwargs):
        """
        Add example data.
        If the country has been registered,
        the start date will be the next data of the registered records.

        Args:
            model (cs.ModelBase): the first ODE model
            country (str or None): country name
            province (str or None): province name
            kwargs: the other keyword arguments of ODESimulator.add()

        Notes:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        # Arguments
        model = self.ensure_subclass(model, ModelBase, name="model")
        arg_dict = model.EXAMPLE.copy()
        arg_dict.update(kwargs)
        population = arg_dict["population"]
        # Area
        country, province = self._model_to_area(
            model=model, country=country, province=province)
        # Start date and y0 values
        df = self._cleaned_df.copy()
        df = df.loc[
            (df[self.COUNTRY] == country) & (df[self.PROVINCE] == province), :
        ]
        if df.empty:
            start_date = self.start_date
        else:
            start_date = df.loc[
                df.index[-1], self.DATE].strftime(self.DATE_FORMAT)
            df = model.tau_free(df, population, tau=None)
            arg_dict["y0_dict"] = {
                k: df.loc[df.index[0], k] for k in model.VARIABLES
            }
        # Simulation
        simulator = ODESimulator(country=country, province=province)
        simulator.add(model=model, **arg_dict)
        # Specialized records
        dim_df = simulator.dim(tau=self.tau, start_date=start_date)
        if country not in self._specialized_dict:
            self._specialized_dict[country] = {}
        self._specialized_dict[country][province] = dim_df.copy()
        # JHU-type records
        restored_df = model.restore(dim_df)
        restored_df[self.COUNTRY] = country
        restored_df[self.PROVINCE] = province
        selected_df = restored_df.loc[:, self.COLUMNS]
        self._cleaned_df = pd.concat(
            [self._cleaned_df, selected_df], axis=0, ignore_index=True
        )
        # Set non-dimensional data
        if country not in self.nondim_dict:
            self.nondim_dict[country] = {}
        nondim_df = simulator.non_dim()
        if province in self.nondim_dict[country]:
            nondim_df_old = self.nondim_dict[country][province].copy()
            nondim_df = pd.concat([nondim_df_old, nondim_df], axis=0)
        self.nondim_dict[country][province] = nondim_df.copy()

    def specialized(self, model=None, country=None, province=None):
        """
        Return dimensional records with model variables.

        Args:
            model (cs.ModelBase or None): the first ODE model
            country (str or None): country name
            province (str or None): province name

        Notes:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        country, province = self._model_to_area(
            model=model, country=country, province=province)
        try:
            return self._specialized_dict[country][province]
        except KeyError:
            raise KeyError(
                f"Records of {country} - {province} were not registered."
            )

    def non_dim(self, model=None, country=None, province=None):
        """
        Return non-dimensional data.

        Args:
            model (cs.ModelBase or None): the first ODE model
            country (str or None): country name
            province (str or None): province name

        Notes:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        country, province = self._model_to_area(
            model=model, country=country, province=province)
        try:
            return self.nondim_dict[country][province]
        except KeyError:
            raise KeyError(
                f"Records of {country} - {province} were not registered."
            )

    def subset(self, model=None, country=None, province=None, **kwargs):
        """
        Return the subset of dataset.

        Args:
            model (cs.ModelBase or None): the first ODE model
            country (str or None): country name
            province (str or None): province name
            kwargs: other keyword arguments of JHUData.subset()

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Notes:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
            If @population is not None, the number of susceptible cases will be calculated.
            Records with Recovered > 0 will be selected.
        """
        country, _ = self._model_to_area(
            model=model, country=country, province=province)
        return super().subset(country=country, province=province, **kwargs)
