#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import numpy as np
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.ode_handler import ODEHandler


class ExampleData(JHUData):
    """
    Example dataset as a child class of JHUData.

    Args:
        clean_df (pandas.DataFrame or None): cleaned data

            Index
                - reset index
            Columns
                - Date (pd.Timestamp): Observation date
                - Country (pandas.Category): country/region name
                - Province (pandas.Category): province/prefecture/sstate name
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
        clean_df = self._ensure_dataframe(clean_df, name="clean_df", columns=self.COLUMNS)
        self._raw = clean_df.copy()
        self._cleaned_df = clean_df.copy()
        self._citation = str()
        self._tau = self._ensure_tau(tau)
        self._start = self._ensure_date(start_date, name="start_date")
        self._population = None
        self._recovery_period = None

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
        model = self._ensure_subclass(model, ModelBase, name="model")
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

        Note:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        # Arguments
        model = self._ensure_subclass(model, ModelBase, name="model")
        arg_dict = model.EXAMPLE.copy()
        arg_dict.update(kwargs)
        self._population = arg_dict["population"]
        # Area
        country, province = self._model_to_area(model=model, country=country, province=province)
        # Start date and y0 values
        df = self._cleaned_df.copy()
        df = df.loc[(df[self.COUNTRY] == country) & (df[self.PROVINCE] == province), :]
        if df.empty:
            start = self._start
        else:
            start = df.loc[df.index[-1], self.DATE]
            df[self.S] = self._population - df[self.C]
            df = model.convert(df, tau=None)
            arg_dict[self.Y0_DICT] = {k: df.loc[df.index[0], k] for k in model.VARIABLES}
        # Simulation
        end = start + timedelta(days=int(arg_dict[self.STEP_N] * self._tau / 1440))
        handler = ODEHandler(model, start, tau=self._tau)
        handler.add(end, param_dict=arg_dict[self.PARAM_DICT], y0_dict=arg_dict[self.Y0_DICT])
        restored_df = handler.simulate()
        # JHU-type records
        restored_df[self.COUNTRY] = country
        restored_df[self.PROVINCE] = province
        restored_df[self.C] = restored_df[[self.CI, self.F, self.R]].sum(axis=1)
        selected_df = restored_df.loc[:, self.COLUMNS]
        cleaned_df = pd.concat([self._cleaned_df, selected_df], axis=0, ignore_index=True)
        for col in self.AREA_COLUMNS:
            cleaned_df[col] = cleaned_df[col].astype("category")
        for col in self.VALUE_COLUMNS:
            cleaned_df[col] = cleaned_df[col].astype(np.int64)
        self._cleaned_df = cleaned_df.copy()

    def specialized(self, model=None, country=None, province=None):
        """
        Return dimensional records with model variables.

        Args:
            model (cs.ModelBase or None): the first ODE model
            country (str or None): country name
            province (str or None): province name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - (int) variables of the model

        Note:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        restored_df = self.subset(model=model, country=country, province=province)
        restored_df[self.S] = self._population - restored_df[self.C]
        return model.convert(restored_df, tau=None).reset_index()

    def non_dim(self, model=None, country=None, province=None):
        """
        Return non-dimensional data.

        Args:
            model (cs.ModelBase or None): the first ODE model
            country (str or None): country name
            province (str or None): province name

        Returns:
            pandas.DataFrame:
                Index
                    t: Dates divided by tau value (time steps)
                Columns
                    - (int) variables of the model

        Note:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        restored_df = self.subset(model=model, country=country, province=province)
        restored_df[self.S] = self._population - restored_df[self.C]
        df = model.convert(restored_df, tau=self._tau)
        return df.apply(lambda x: x / x.sum(), axis=1)

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
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Note:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
            If @population is not None, the number of susceptible cases will be calculated.
            Records with Recovered > 0 will be selected.
        """
        country, _ = self._model_to_area(model=model, country=country, province=province)
        kwargs["population"] = self._population
        kwargs = find_args([super().subset], **kwargs)
        return super().subset(country=country, province=province, **kwargs)

    def subset_complement(self, **kwargs):
        """
        This is the same as ExampleData.subset().
        Complement will not be done.
        """
        return (self.subset(**kwargs), False)

    def records(self, **kwargs):
        """
        This is the same as ExampleData.subset().
        Complement will not be done.
        """
        return (self.subset(**kwargs), False)
