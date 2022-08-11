#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import numpy as np
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy.util.validator import Validator
from covsirphy._deprecated.jhu_data import JHUData
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.ode_handler import ODEHandler


class ExampleData(JHUData):
    """
    Deprecated. Example dataset as a child class of JHUData.

    Args:
        clean_df (pandas.DataFrame or None): cleaned data

            Index
                - reset index
            Columns
                - Date (pd.Timestamp): Observation date
                - Country (pandas.Category): country/region name
                - Province (pandas.Category): province/prefecture/state name
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases

        tau (int): tau value [min]
        start_date (str): start date, like 22Jan2020
    """

    @deprecate(old="ExampleData()", new="Dynamics.from_sample()", version="2.24.0-kappa")
    def __init__(self, clean_df=None, tau=1440, start_date="22Jan2020"):
        if clean_df is None:
            clean_df = pd.DataFrame(columns=self.COLUMNS)
        clean_df = self._ensure_dataframe(clean_df, name="clean_df", columns=self.COLUMNS)
        variables = [self.C, self.CI, self.F, self.R, self.N]
        self._raw_cols = [self.DATE, self.ISO3, self.COUNTRY, self.PROVINCE, *variables]
        self._subset_cols = [self.DATE, *variables]
        self._raw = clean_df.copy()
        self._cleaned_df = clean_df.copy()
        self._citation = str()
        self._tau = Validator(tau, "tau").tau(default=None)
        self._start = Validator(start_date, "start_date").date()
        self._population = None
        self._recovery_period = None

    @staticmethod
    def _ensure_dataframe(target, name="df", time_index=False, columns=None, empty_ok=True):
        """
        Ensure the dataframe has the columns.

        Args:
            target (pandas.DataFrame): the dataframe to ensure
            name (str): argument name of the dataframe
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have
            empty_ok (bool): whether give permission to empty dataframe or not

        Returns:
            pandas.DataFrame:
                Index
                    as-is
                Columns:
                    columns specified with @columns or all columns of @target (when @columns is None)
        """
        if not isinstance(target, pd.DataFrame):
            raise TypeError(f"@{name} must be a instance of (pandas.DataFrame).")
        df = target.copy()
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if not empty_ok and target.empty:
            raise ValueError(f"@{name} must not be a empty dataframe.")
        if columns is None:
            return df
        if not set(columns).issubset(df.columns):
            cols_str = ", ".join(col for col in columns if col not in df.columns)
            included = ", ".join(df.columns.tolist())
            s1 = f"Expected columns were not included in {name} with {included}."
            raise KeyError(f"{s1} {cols_str} must be included.")
        return df.loc[:, columns]

    def ensure_country_name(self, country, errors="raise"):
        """
        Ensure that the country name is correct.

        Args:
            country (str): country name
            errors (str): 'raise' or 'coerce'

        Returns:
            str: country name

        Raises:
            SubsetNotFoundError: no records were found for the country and @errors is 'raise'
        """
        df = self._cleaned_df.copy()
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.COUNTRY])
        selectable_set = set(df[self.COUNTRY].unique())
        if country in selectable_set:
            return country
        if errors == "raise":
            raise SubsetNotFoundError(country=country)

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
        province = province or self.NA
        if country is not None:
            return (country, province)
        if model is None:
            raise ValueError("@model or @country must be specified.")
        model = Validator(model, "model").subclass(ModelBase)
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
            kwargs: the other keyword arguments of model.EXAMPLE

        Note:
            If country is None, the name of the model will be used.
            If province is None, '-' will be used.
        """
        # Arguments
        model = Validator(model, "model").subclass(ModelBase)
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
        restored_df[self.ISO3] = country
        restored_df[self.COUNTRY] = country
        restored_df[self.PROVINCE] = province
        restored_df[self.C] = restored_df[[self.CI, self.F, self.R]].sum(axis=1)
        restored_df[self.N] = self._population
        selected_df = restored_df.loc[:, self._raw_cols]
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
