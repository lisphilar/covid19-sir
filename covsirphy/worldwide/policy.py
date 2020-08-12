#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import groupby
from operator import itemgetter
import pandas as pd
from covsirphy.util.plotting import line_plot
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.analysis.scenario import Scenario


class PolicyMeasures(Term):
    """
    Analyse the relationship of policy measures and parameters of ODE models.
    This analysis will be done at country level because OxCGRT tracks policies at country level.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        tau (int): tau value [min]
    """

    def __init__(self, jhu_data, population_data, oxcgrt_data, tau=1440):
        # Records
        self.jhu_data = self.ensure_instance(
            jhu_data, JHUData, name="jhu_data")
        # Population
        self.population_data = self.ensure_instance(
            population_data, PopulationData, name="population_data")
        # OxCGRT
        self.oxcgrt_data = self.ensure_instance(
            oxcgrt_data, OxCGRTData, name="oxcgrt_data")
        # tau value must be shared
        self.tau = self.ensure_tau(tau)
        # Init
        self._countries = self._all_countries()
        self._init_scenario()
        self.model = None

    def _all_countries(self):
        """
        Return names of countries where records are registered.

        Returns:
            (list[str]): list of country names
        """
        j_list = self.jhu_data.countries()
        p_list = self.population_data.countries()
        o_list = self.oxcgrt_data.countries()
        return list(set(j_list) & set(p_list) & set(o_list))

    def _init_scenario(self):
        """
        Initialize the scenario classes of registered countries.
        """
        self.scenario_dict = {
            country: Scenario(
                self.jhu_data, self.population_data, country=country, tau=self.tau)
            for country in self._countries
        }

    def scenario(self, country):
        """
        Return Scenario instance of the country.

        Args:
            country (str): country name

        Raises:
            KeyError: the country is not registered

        Returns:
            covsirphy.Scenario: Scenario instance
        """
        if country not in self.scenario_dict.keys():
            raise KeyError(f"{country} is not registered.")
        return self.scenario_dict[country]

    @property
    def countries(self):
        """
        list[str]: countries to analyse
        """
        return self._countries

    @countries.setter
    def countries(self, country_list):
        selected_set = set(country_list)
        all_set = set(self._all_countries())
        if not selected_set.issubset(all_set):
            un_selectable_set = selected_set - all_set
            un_selectable = ", ".join(list(un_selectable_set))
            raise KeyError(
                f"{un_selectable} cannot be selected because records not registered.")
        self._countries = country_list[:]

    def trend(self, min_len=2):
        """
        Perform S-R trend analysis for all registered countries.

        Args:
            min_len (int): minimum length of phases to have

        Returns:
            covsirphy.PolicyMeasures: self

        Notes:
            Countries which do not have @min_len phases will be un-registered.
        """
        min_len = self.ensure_natural_int(min_len, name="min_len")
        for country in self._countries:
            self.scenario_dict[country].trend(
                set_phases=True, show_figure=False)
        countries = [
            country for country in self._countries
            if len(self.scenario_dict[country][self.MAIN]) >= min_len
        ]
        self.countries = countries
        return self

    def summary(self, columns=None, countries=None):
        """
        Summarize of scenarios.

        Args:
            columns (list[str] or None): columns to show
            countries (list[str] or None): countries to show

        Returns:
            pandas.DataFrame

        Notes:
            If @columns is None, all columns will be shown.
        """
        countries = countries or self._countries
        if not isinstance(countries, (list, set)):
            raise TypeError("@countries must be a list or set.")
        dataframes = []
        for country in countries:
            df = self.scenario_dict[country].summary(columns=columns)
            df[self.PHASE] = df.index
            df[self.COUNTRY] = country
            dataframes.append(df)
        summary_df = pd.concat(dataframes, axis=0, ignore_index=True)
        return summary_df.set_index([self.COUNTRY, self.PHASE])

    def phase_len(self):
        """
        Make groups of countries with the length of phases.

        Returns:
            dict(int, list[str]): list of countries with the length of phases
        """
        len_nest = [
            (country, len(self.scenario_dict[country][self.MAIN]))
            for country in self._countries
        ]
        sorted_nest = sorted(len_nest, key=itemgetter(1), reverse=True)
        return {
            length: [country for (country, _) in records]
            for (length, records) in groupby(sorted_nest, key=itemgetter(1))
        }

    def estimate(self, model):
        """
        Estimate the parameter values of phases in the registered countries.

        Args:
            model (covsirphy.ModelBase): ODE model
        """
        model = self.ensure_subclass(model, ModelBase)
        for country in self._countries:
            print(f"\n{'-' * 20}{country}{'-' * 20}")
            self.scenario_dict[country].estimate(model)
        self.model = model

    def param_history(self, param, roll_window=None, show_figure=True, filename=None, **kwargs):
        """
        Return subset of summary and show a figure to show the history in each country.

        Args:
            param (str): parameter to show
            roll_window (int or None): rolling average window if necessary
            show_figure (bool): If True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keword arguments of pd.DataFrame.plot or line_plot()

        Returns:
            pandas.DataFrame:
                Index: (int) phase number
                Columns: (str) country names
                Values: parameter values
        """
        if self.model is None:
            raise TypeError(
                "PolicyMeasures.estimate(model) must be done in advance.")
        selectable_params = [
            *self.model.PARAMETERS, *self.model.DAY_PARAMETERS, self.RT]
        if param not in selectable_params:
            sel_param_str = ', '.join(selectable_params)
            raise KeyError(
                f"@param must be selected from {sel_param_str}, but {param} was applied.")
        # Get the parameter value of each date
        df = self.summary().reset_index()
        df[self.START] = pd.to_datetime(
            df[self.START], format=self.DATE_FORMAT)
        df[self.END] = pd.to_datetime(df[self.END], format=self.DATE_FORMAT)
        df[self.DATE] = df[[self.START, self.END]].apply(
            lambda x: pd.date_range(x[0], x[1]).tolist(), axis=1)
        df = df.explode(self.DATE)
        df = df.pivot_table(
            values=param, index=self.DATE, columns=self.COUNTRY)
        # Rolling mean
        if roll_window is not None:
            roll_window = self.ensure_natural_int(
                roll_window, name="roll_window")
            df = df.rolling(window=roll_window).mean()
        # Show figure
        if not show_figure:
            return df
        line_plot(
            df, title=f"History of {param} in each country",
            ylabel=param,
            h=1 if param == self.RT else None,
            filename=filename
        )
        return df
