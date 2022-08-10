#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from itertools import groupby
from operator import itemgetter
import pandas as pd
from covsirphy.util.error import deprecate, UnExecutedError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated.jhu_data import JHUData
from covsirphy._deprecated.population import PopulationData
from covsirphy._deprecated.oxcgrt import OxCGRTData
from covsirphy.visualization.line_plot import line_plot
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.phase_estimator import MPEstimator
from covsirphy._deprecated.scenario import Scenario


class PolicyMeasures(Term):
    """
    Deprecated.
    Analyze the relationship of policy measures and parameters of ODE models.
    This analysis will be done at country level because OxCGRT tracks policies at country level.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        tau (int or None): tau value [min]
    """

    @deprecate("PolicyMeasures", version="2.19.1-zeta-fu1")
    def __init__(self, jhu_data, population_data, oxcgrt_data, tau=None):
        # Records
        self.jhu_data = self._ensure_instance(
            jhu_data, JHUData, name="jhu_data")
        # Population
        self.population_data = self._ensure_instance(
            population_data, PopulationData, name="population_data")
        # OxCGRT
        self.oxcgrt_data = self._ensure_instance(
            oxcgrt_data, OxCGRTData, name="oxcgrt_data")
        # tau value must be shared
        self.tau = Validator(tau, "tau").tau(default=None)
        # Init
        self._countries = self._all_countries()
        self._init_scenario()
        self.model = None

    @staticmethod
    def _ensure_instance(target, class_obj, name="target"):
        """
        Ensure the target is an instance of the class object.

        Args:
            target (instance): target to ensure
            parent (class): class object
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        s = f"@{name} must be an instance of {class_obj}, but {type(target)} was applied."
        if not isinstance(target, class_obj):
            raise TypeError(s)
        return target

    def _all_countries(self):
        """
        Return names of countries where records are registered.

        Returns:
            (list[str]): list of country names
        """
        j_list = self.jhu_data.countries(complement=True)
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
        list[str]: countries to analyze
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
                f"{un_selectable} cannot be selected because records are not registered.")
        self._countries = country_list

    def trend(self, min_len=2):
        """
        Perform S-R trend analysis for all registered countries.

        Args:
            min_len (int): minimum length of phases to have

        Returns:
            covsirphy.PolicyMeasures: self

        Note:
            Countries which do not have @min_len phases will be un-registered.
        """
        min_len = self._ensure_natural_int(min_len, name="min_len")
        for country in self._countries:
            with contextlib.suppress(ValueError):
                self.scenario_dict[country].trend(set_phases=True, show_figure=False)
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

        Note:
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
        summary_df = summary_df.set_index([self.COUNTRY, self.PHASE])
        return summary_df.fillna(self.NA)

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

    def estimate(self, model, n_jobs=-1, **kwargs):
        """
        Estimate the parameter values of phases in the registered countries.

        Args:
            model (covsirphy.ModelBase): ODE model
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()
        """
        model = Validator(model, "model").subclass(ModelBase)
        unit_nest = [
            [
                unit.set_id(
                    country=self.jhu_data.country_to_iso3(country),
                    phase=f"{self.num2str(num):>4}")
                for (num, unit) in enumerate(self.scenario_dict[country][self.MAIN]) if unit]
            for country in self._countries
        ]
        units = self.flatten(unit_nest)
        # Parameter estimation
        mp_estimator = MPEstimator(
            jhu_data=self.jhu_data, population_data=self.population_data,
            model=model, tau=self.tau, **kwargs)
        mp_estimator.add(units)
        results = mp_estimator.run(
            n_jobs=n_jobs, auto_complement=True, **kwargs)
        # Register the results
        for country in self._countries:
            new_units = [
                unit for unit in results
                if unit.id_dict["country"] == self.jhu_data.country_to_iso3(country)]
            self.scenario_dict[country][self.MAIN].replaces(
                phase=None, new_list=new_units, keep_old=True)
        self.model = model
        self.tau = mp_estimator.tau

    @deprecate(
        old="PolicyMeasures.param_history(param: str)",
        new="PolicyMeasures.history(param: str)",
        version="2.9.0")
    def param_history(self, param, roll_window=None, show_figure=True, filename=None, **kwargs):
        return self.history(param, roll_window=None, show_figure=True, filename=None, **kwargs)

    def history(self, param, roll_window=None, show_figure=True, filename=None, **kwargs):
        """
        Return subset of summary and show a figure to show the history of all countries.

        Args:
            param (str): parameter/day parameter/Rt/OxCGRT score to show
            roll_window (int or None): rolling average window if necessary
            show_figure (bool): If True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of line_plot()

        Returns:
            pandas.DataFrame:
                Index
                    Date (pd.Timestamp) date
                Columns
                    (str) country names
                Values:
                    parameter values
        """
        # Get the parameter value of each date
        df = self.track()
        # Select the param
        if param not in df.columns:
            sel_param_str = ', '.join(df.columns.tolist())
            raise KeyError(
                f"@param must be selected from {sel_param_str}, but {param} was applied.")
        df = df.pivot_table(
            values=param, index=self.DATE, columns=self.COUNTRY, aggfunc="last")
        # Rolling mean
        if roll_window is not None:
            roll_window = self._ensure_natural_int(
                roll_window, name="roll_window")
            df = df.rolling(window=roll_window).mean()
        # Show figure
        if not show_figure:
            return df
        line_plot(
            df, title=f"History of {param} in each country",
            ylabel=param,
            h=1 if param == self.RT else None,
            filename=filename,
            **kwargs
        )
        return df

    @staticmethod
    def _ensure_natural_int(target, name="number", include_zero=False, none_ok=False):
        """
        Ensure a natural (non-negative) number.

        Args:
            target (int or float or str or None): value to ensure
            name (str): argument name of the value
            include_zero (bool): include 0 or not
            none_ok (bool): None value can be applied or not.

        Returns:
            int: as-is the target

        Note:
            When @target is None and @none_ok is True, None will be returned.
            If the value is a natural number and the type was float or string,
            it will be converted to an integer.
        """
        if target is None and none_ok:
            return None
        s = f"@{name} must be a natural number, but {target} was applied"
        try:
            number = int(target)
        except TypeError as e:
            raise TypeError(f"{s} and not converted to integer.") from e
        if number != target:
            raise ValueError(f"{s}. |{target} - {number}| > 0")
        min_value = 0 if include_zero else 1
        if number < min_value:
            raise ValueError(f"{s}. This value is under {min_value}")
        return number

    def track(self):
        """
        Return subset of summary and show a figure to show the history in each country.

        Args:
            param (str): parameter to show
            roll_window (int or None): rolling average window if necessary
            show_figure (bool): If True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of pd.DataFrame.plot or line_plot()

        Returns:
            pandas.DataFrame: parameter values
                Index
                    reset index
                Columns
                    - Country (str): country name
                    - Date (pd.Timestamp): date
                    - (float): model parameters
                    - (float): model day parameters
                    - Rt (float): reproduction number
                    - (float): OxCGRT values
        """
        if self.model is None:
            raise UnExecutedError("PolicyMeasures.estimate(model)")
        # Get parameter/Rt/data parameter value of each date
        df = self.summary().reset_index().replace(self.NA, None)
        df[self.START] = pd.to_datetime(
            df[self.START], format=self.DATE_FORMAT)
        df[self.END] = pd.to_datetime(df[self.END], format=self.DATE_FORMAT)
        df[self.DATE] = df[[self.START, self.END]].apply(
            lambda x: pd.date_range(x[0], x[1]).tolist(), axis=1)
        df = df.explode(self.DATE)
        cols = [
            self.DATE, self.COUNTRY, *self.model.PARAMETERS, *self.model.DAY_PARAMETERS, self.RT]
        param_df = df.reindex(cols, axis=1)
        # OxCGRT
        oxcgrt_df = self.oxcgrt_data.cleaned()
        sel = oxcgrt_df[self.COUNTRY].isin(self._countries)
        oxcgrt_df = oxcgrt_df.loc[
            sel, [self.DATE, self.COUNTRY, *OxCGRTData.OXCGRT_VARS]]
        # Combine data
        return pd.merge(
            param_df, oxcgrt_df, how="inner", on=[self.COUNTRY, self.DATE])
