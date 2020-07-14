#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.analysis.simulator import ODESimulator
from covsirphy.ode.mbase import ModelBase


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
        self._raw = clean_df.copy()
        self._cleaned_df = clean_df.copy()
        self._citation = str()
        self.tau = self.validate_natural_int(tau, name="tau")
        self.start_date = self.validate_date(start_date, name="start_date")
        self.nondim_dict = dict()

    def add(self, model, country=None, province=None, **kwargs):
        """
        Add example data.
        If the country has been registered,
        the start date will be the next data of the registered records.

        Args:
            model (subclass of cs.ModelBase): the first ODE model
            country (str): country name
            province (str): province name
            kwargs: the other keyword arguments of ODESimulator.add()

        Notes:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        # Arguments
        model = self.validate_subclass(model, ModelBase, name="model")
        arg_dict = model.EXAMPLE.copy()
        arg_dict.update(kwargs)
        country = country or model.NAME
        province = province or self.UNKNOWN
        try:
            population = arg_dict["population"]
        except KeyError:
            raise KeyError("@population must be specified.")
        # Start date and y0 values
        df = self._cleaned_df.copy()
        df = df.loc[
            (df[self.COUNTRY] == country) & (df[self.PROVINCE] == province)
        ]
        if df.empty:
            start_date = self.start_date
        else:
            start_date = df.loc[df.index[-1], self.DATE]
            df = model.tau_free(df, population, tau=None)
            arg_dict["y0_dict"] = {
                k: df.loc[df.index[0], k] for k in model.VARIABLES
            }
        # Simulation
        simulator = ODESimulator(country=country, province=province)
        simulator.add(model=model, **arg_dict)
        simulator.run()
        # Add the simulated records to self
        dim_df = simulator.dim(tau=self.tau, start_date=start_date)
        restored_df = model.restore(dim_df)
        restored_df[self.COUNTRY] = country
        restored_df[self.PROVINCE] = province
        selected_df = restored_df.loc[:, self.COLUMNS]
        self._cleaned_df = pd.concat(
            [self._cleaned_df, selected_df], axis=0, ignore_index=True
        )
        # Set non-dimensional data
        if country not in self.nondim_dict.keys():
            self.nondim_dict[country] = dict()
        nondim_df = simulator.non_dim()
        if province in self.nondim_dict[country].keys():
            nondim_df_old = self.nondim_dict[country][province].copy()
            nondim_df = pd.concat([nondim_df_old, nondim_df], axis=0)
        self.nondim_dict[country][province] = nondim_df.copy()

    def non_dim(self, model=None, country=None, province=None):
        """
        Return non-dimensional data.

        Args:
            model (subclass of cs.ModelBase or None): the first ODE model
            country (str): country name
            province (str): province name

        Notes:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        if model is not None:
            model = self.validate_subclass(model, ModelBase, name="model")
            country = country or model.NAME
            province = province or self.UNKNOWN
        if country is None:
            raise KeyError("@model or @country must be applied.")
        try:
            return self.nondim_dict[country][province]
        except KeyError:
            raise KeyError(
                f"Records of {country} - {province} were not registered."
            )

    def subset(self, model=None, **kwargs):
        """
        Return the subset of dataset.

        Args:
            model (subclass of cs.ModelBase or None): the first ODE model
            kwargs: keyword arguments of JHUData.subset()

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
        if "country" not in kwargs.keys():
            if model is None:
                raise KeyError("@model or @country must be applied.")
            model = self.validate_subclass(model, ModelBase, name="model")
            kwargs["country"] = model.NAME
        return super().subset(**kwargs)
