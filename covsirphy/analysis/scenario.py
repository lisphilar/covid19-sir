#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from datetime import timedelta
from inspect import signature
import sys
import warnings
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import numpy as np
import pandas as pd
from covsirphy.ode import ModelBase
from covsirphy.cleaning import JHUData, PopulationData, Word
from covsirphy.phase import Estimator, SRData, ODEData
from covsirphy.util import line_plot, box_plot
from covsirphy.analysis.phase_series import PhaseSeries
from covsirphy.analysis.simulator import ODESimulator
from covsirphy.analysis.sr_change import ChangeFinder


class Scenario(Word):
    """
    Scenario analysis.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        country (str): country name
        province (str): province name
    """

    def __init__(self, jhu_data, population_data, country, province=None):
        # Population
        population_data = self.validate_instance(
            population_data, PopulationData, name="population_data")
        self.population = population_data.value(country, province=province)
        # Records
        jhu_data = self.validate_instance(jhu_data, JHUData, name="jhu_data")
        self.jhu_data = jhu_data
        self.clean_df = jhu_data.cleaned(population=self.population)
        # Area name
        self.country = country
        self.province = province
        if province is None:
            self.area = country
        else:
            self.area = f"{country}{self.SEP}{province}"
        # First/last date of the area
        sr_data = SRData(self.clean_df, country=country, province=province)
        df = sr_data.make(self.population)
        self.first_date = df.index.min().strftime(self.DATE_FORMAT)
        self.last_date = df.index.max().strftime(self.DATE_FORMAT)
        # Init
        self.tau = None
        # {model_name: model_class}
        self.model_dict = dict()
        # {scenario_name: PhaseSeries}
        self.series_dict = dict()
        self.series_dict[self.MAIN] = PhaseSeries(
            self.first_date, self.last_date, self.population
        )
        # {scenario: {phase: Estimator}}
        self.estimator_dict = dict()

    def delete(self, name):
        """
        Delete a PhaseSeries.

        Args:
            name (str): PhaseSeries name
        """
        if name == self.MAIN:
            raise ValueError(f"@name {name} cannot be deleted.")
        self.series_dict.pop(name)

    def records(self, show_figure=True, filename=None):
        """
        Return the records as a dataframe.

        Args:
            show_figure (bool): if True, show the records as a line-plot.
            filename (str): filename of the figure, or None (show figure)
        """
        df = self.jhu_data.subset(self.country, province=self.province)
        if not show_figure:
            return df
        line_plot(
            df.set_index(self.DATE).drop(self.C, axis=1),
            f"{self.area}: Cases over time",
            y_integer=True,
            filename=filename
        )
        return df

    def add_phase(self, name="Main", end_date=None, days=None,
                  population=None, model=None, **kwargs):
        """
        Add a new phase.
        The start date is the next date of the last registered phase.

        Args:
            name (str): phase series name, 'Main' or user-defined name
            end_date (str): end date of the new phase
            days (int): the number of days to add
            population (int): population value of the start date
                - if None, the same as initial value
            model (covsirphy.ModelBase): ODE model
                - if None, the model of the last phase will be used
            kwargs: keyword arguments of ODE model parameters
                - un-included parameters will be the same as the last phase
                    - if model is not the same, None
                    - tau is fixed as the last phase's value or 1440

        Notes:
            If the phases series has not been registered, new phase series will be created.
            @end_date or @days must be specified.

        Returns:
            None
        """
        # Parse arguments
        if not isinstance(name, str):
            raise TypeError(
                f"@name must be a string, but {type(name)} was applied.")
        if name == "Main":
            name = self.MAIN
        if name not in self.series_dict.keys():
            self.series_dict[name] = copy.deepcopy(self.series_dict[self.MAIN])
            self.series_dict[name].clear()
        start_date = self.series_dict[name].next_date()
        if end_date is None:
            if days is None:
                raise NameError("@end_date or @days must be specified.")
            if not isinstance(days, int):
                raise TypeError("@days must be an integer.")
            end_obj = self.date_obj(start_date) + timedelta(days=days)
            end_date = end_obj.strftime(self.DATE_FORMAT)
        population = population or self.population
        summary_df = self.series_dict[name].summary()
        if model is None:
            if self.ODE not in summary_df.columns:
                self.series_dict[name].add(start_date, end_date, population)
                return None
            last_model_name = summary_df.loc[summary_df.index[-1], self.ODE]
            model = self.model_dict[last_model_name]
        model = self.validate_subclass(model, ModelBase, name="model")
        # Phase information
        param_dict = {self.TAU: self.tau, self.ODE: model.NAME}
        model_param_dict = {
            param: summary_df.loc[summary_df.index[-1], param]
            for param in model.PARAMETERS
        }
        model_param_dict.update(kwargs)
        param_dict.update(model_param_dict)
        model_instance = model(
            population=population, **model_param_dict
        )
        param_dict[self.RT] = model_instance.calc_r0()
        param_dict.update(model_instance.calc_days_dict(self.tau))
        self.series_dict[name].add(
            start_date, end_date, population, **param_dict)

    def clear(self, name="Main", include_past=False):
        """
        Clear phase information.

        Args:
            name (str): phase series name
                - if 'Main', main phase series will be used
                - if not registered, new phaseseries will be created
            include_past (bool):
                - if True, include past phases.
                - future phase are always included
        """
        if name == "Main":
            name = self.MAIN
        if name not in self.series_dict.keys():
            self.series_dict[name] = copy.deepcopy(self.series_dict[self.MAIN])
        self.series_dict[name].clear(include_past=include_past)

    def summary(self, name=None):
        """
        Summarize the series of phases and return a dataframe.

        Args:
            name (str): phase series name
                - name of alternative phase series registered by self.add_phase()
                - if None, all phase series will be shown

        Returns:
            (pandas.DataFrame):
            - if @name not None, as the same as PhaseSeries().summary()
            - if @name is None, index will be phase series name and phase name

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        if name is None and len(self.series_dict.keys()) > 1:
            dataframes = list()
            for (_name, series) in self.series_dict.items():
                df = series.summary()
                df[self.PHASE] = df.index
                df[self.SERIES] = _name
                df = df.reset_index(drop=True)
                dataframes.append(df)
            summary_df = pd.concat(dataframes, axis=0, ignore_index=True)
            summary_df = summary_df.set_index([self.SERIES, self.PHASE])
            return summary_df
        if name == "Main" or len(self.series_dict.keys()) == 1:
            name = self.MAIN
        try:
            series = self.series_dict[name]
        except KeyError:
            raise KeyError(f"@name {name} has not been registered.")
        return series.summary()

    def trend(self, n_points=0,
              set_phases=True, include_init_phase=False,
              show_figure=True, filename=None, **kwargs):
        """
        Perform S-R trend analysis and set phases.

        Args:
            n_points (int): the number of change points
            set_phases (bool):
                - if True and n_points is not 0, set phases automatically
                - if @include_init_phase is False, initial phase will not be used
            include_init_phase (bool): whether use initial phase or not
            show_figure (bool):
                - if True, show the records as a line-plot.
            filename (str): filename of the figure, or None (show figure)
            kwargs: the other keyword arguments of ChangeFinder().run()

        Returns:
            None
        """
        finder = ChangeFinder(
            self.clean_df, self.population,
            country=self.country, province=self.province
        )
        finder.run(n_points=n_points, **kwargs)
        phase_series = finder.show(show_figure=show_figure, filename=filename)
        if n_points != 0 and set_phases:
            for name in self.series_dict.keys():
                self.series_dict[name] = phase_series
                if not include_init_phase:
                    self.series_dict[name].delete("0th")

    def _estimate(self, model, phase, name="Main", **kwargs):
        """
        Estimate the parameters of the model using the records.

        Args:
            phase (str): phase name, like 1st, 2nd...
            model (covsirphy.ModelBase): ODE model
            name (str): phase series name
            kwargs:
                - keyword arguments of the model parameter
                    - tau value cannot be included
                - keyword arguments of covsirphy.Estimator.run()

        Returns:
            self

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
            If @name phase was not registered, new PhaseSeries will be created.
        """
        name = self.MAIN if name == "Main" else name
        if name not in self.series_dict.keys():
            self.series_dict[name] = copy.deepcopy(self.series_dict[self.MAIN])
        # Set parameters
        try:
            setting_dict = self.series_dict[name].to_dict()[phase]
        except KeyError:
            raise KeyError(
                f"{phase} phase has not been registered for @name {name}.")
        start_date = setting_dict[self.START]
        end_date = setting_dict[self.END]
        population = setting_dict[self.N]
        # Run estimation
        print(f"{phase} phase with {model.NAME} model:")
        est_kwargs = {
            p: kwargs[p] for p in model.PARAMETERS if p in kwargs.keys()
        }
        if self.tau is not None:
            if self.TAU in kwargs.keys():
                raise ValueError(f"{self.TAU} cannot be changed.")
            est_kwargs[self.TAU] = self.tau
        estimator = Estimator(
            self.clean_df, model, population,
            country=self.country, province=self.province,
            start_date=start_date, end_date=end_date,
            **est_kwargs
        )
        sign = signature(Estimator.run)
        run_params = list(sign.parameters.keys())
        run_kwargs = {k: v for (k, v) in kwargs.items() if k in run_params}
        estimator.run(**run_kwargs)
        est_df = estimator.summary(phase)
        phase_est_dict = {self.ODE: model.NAME}
        phase_est_dict.update(est_df.to_dict(orient="index")[phase])
        self.tau = phase_est_dict[self.TAU]
        self.series_dict[name].update(phase, **phase_est_dict)
        if name not in self.estimator_dict.keys():
            self.estimator_dict[name] = dict()
        self.estimator_dict[name][phase] = estimator
        self.model_dict[model.NAME] = model

    def estimate(self, model, series_list=None, phases=None, **kwargs):
        """
        Estimate the parameters of the model using the records.

        Args:
            series_list (list[str]): list of phase series name
                - if None, all phase series will be used
            phases (list[str]): list of phase names, like 1st, 2nd...
                - if None, all past phase will be used
            model (covsirphy.ModelBase): ODE model
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()
        """
        # Check model
        model = self.validate_subclass(model, ModelBase, "model")
        # Phase names
        if series_list is None:
            series_list = list(self.series_dict.keys())
        for name in series_list:
            try:
                phase_dict = self.series_dict[name].to_dict()
            except KeyError:
                raise KeyError(f"{name} has not been defined.")
            if len(series_list) > 1:
                print(f"<{name} scenario>")
            past_phases = [k for (k, v) in phase_dict.items()]
            phases = past_phases[:] if phases is None else phases
            future_phases = list(set(phases) - set(past_phases))
            if future_phases:
                raise KeyError(f"{future_phases[0]} is not a past phase.")
            # Run hyperparameter estimation
            for phase in phases:
                self._estimate(model, phase, **kwargs)

    def estimate_history(self, phase, name="Main", **kwargs):
        """
        Show the history of optimization.

        Args:
            phase (str): phase name, like 1st, 2nd...
            name (str): phase series name
            kwargs: keyword arguments of covsirphy.Estimator.history()

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        name = self.MAIN if name == "Main" else name
        if name not in self.series_dict.keys():
            raise KeyError(f"@name {name} has not been defined.")
        try:
            estimator = self.estimator_dict[name][phase]
        except KeyError:
            raise KeyError(
                f"Estimator of {phase} phase in {name} has not been registered."
            )
        estimator.history(**kwargs)

    def estimate_accuracy(self, phase, name="Main", **kwargs):
        """
        Show the accuracy as a figure.

        Args:
            phase (str): phase name, like 1st, 2nd...
            name (str): phase series name
            kwargs: keyword arguments of covsirphy.Estimator.accuracy()

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        name = self.MAIN if name == "Main" else name
        if name not in self.series_dict.keys():
            raise KeyError(f"@name {name} is not defined.")
        try:
            estimator = self.estimator_dict[name][phase]
        except KeyError:
            raise KeyError(
                f"Estimator of {phase} phase in {name} has not been registered."
            )
        estimator.accuracy(**kwargs)

    def predict(self, **kwargs):
        """
        Old method. Please use Scenario.simulate().

        Args:
            kwargs: keyword arguments of Scenario.simulate()

        Returns:
            (pandas.DataFrame)
        """
        warnings.warn(
            "Please use Scenario simulate() rather than Scenario.predict()",
            DeprecationWarning,
            stacklevel=2
        )
        return self.simulate(**kwargs)

    def simulate(self, name="Main", y0_dict=None, show_figure=True, filename=None):
        """
        Simulate ODE models with setted parameter values and show it as a figure.

        Args:
            name (str): phase series name. If 'Main', main PhaseSeries will be used
            y0_dict (dict):
                - key (str): variable name
                - value (float): initial value
                - dictionary of initial values or None
                - if model will be changed in the later phase, must be specified
            show_figure (bool):
                - if True, show the result as a figure.
            filename (str): filename of the figure, or None (show figure)

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (str): date, like 31Dec2020
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - variables of the models (int): Confirmed (int) etc.
        """
        name = self.MAIN if name == "Main" else name
        df = self.series_dict[name].summary()
        # Future phases must be added in advance
        if self.FUTURE not in df[self.TENSE].unique():
            raise KeyError(
                f"Future phases of {name} scenario must be registered by Scenario.add_phase() in advance."
            )
        # Simulation
        dim_df, start_objects = self._simulate(name=name, y0_dict=y0_dict)
        dim_df = dim_df.set_index(self.DATE).resample("D").mean()
        dim_df = dim_df.astype(np.int64)
        fig_df = dim_df.copy()
        dim_df[self.DATE] = dim_df.index.strftime(self.DATE_FORMAT)
        dim_df = dim_df.reset_index(drop=True)
        dim_df = dim_df.loc[:, [self.DATE, *dim_df.columns.tolist()[:-1]]]
        # Return dataframe if figure is not needed
        if not show_figure:
            return dim_df
        # Show figure
        fig_cols_set = set(fig_df.columns) & set(self.FIG_COLUMNS)
        fig_cols = [col for col in self.FIG_COLUMNS if col in fig_cols_set]
        line_plot(
            fig_df[fig_cols],
            title=f"{self.area}: Predicted number of cases",
            filename=filename,
            y_integer=True,
            v=start_objects[1:]
        )
        return dim_df

    def _simulate(self, name, y0_dict):
        """
        Simulate ODE models with setted parameter values.

        Args:
            name (str): phase series name
            y0_dict (dict):
                - key (str): variable name
                - value (float): initial value
                - dictionary of initial values or None
                - if model will be changed in the later phase, must be specified

        Returns:
            (tuple)

                (pandas.DataFrame): output of ODESimulator.dim()

                    Index: reset index
                    Columns:
                        - Date (pd.TimeStamp): Observation date
                        - Country (str): country/region name
                        - Province (str): province/prefecture/state name
                        - variables of the models (int): Confirmed (int) etc.
                (list[pd.TimeStamp]): list of start dates
        """
        df = self.series_dict[name].summary()
        simulator = ODESimulator(
            self.country,
            province="-" if self.province is None else self.province
        )
        start_objects = list()
        for phase in df.index:
            model_name = df.loc[phase, self.ODE]
            model = self.model_dict[model_name]
            start_date = df.loc[phase, self.START]
            start_obj = self.date_obj(start_date)
            start_objects.append(start_obj)
            end_obj = self.date_obj(df.loc[phase, self.END])
            phase_seconds = (end_obj - start_obj).total_seconds() + 1
            step_n = round(phase_seconds / (60 * self.tau))
            population = df.loc[phase, self.N]
            param_dict = df[model.PARAMETERS].to_dict(orient="index")[phase]
            if phase == self.num2str(1):
                # Calculate initial values
                ode_data = ODEData(
                    self.clean_df, country=self.country,
                    province=self.province
                )
                y0_dict_phase = ode_data.y0(
                    model, population, start_date=start_date
                )
            else:
                y0_dict_phase = y0_dict.copy() if y0_dict is not None else None
            simulator.add(
                model, step_n, population,
                param_dict=param_dict,
                y0_dict=y0_dict_phase
            )
        simulator.run()
        first_date = df.loc[self.num2str(1), self.START]
        dim_df = simulator.dim(self.tau, first_date)
        return dim_df, start_objects

    def get(self, param, name="Main", phase="last"):
        """
        Get the parameter value of the phase.

        Args:
            param (str): parameter name (columns in self.summary())
            name (str): phase series name
            phase (str): phase name or 'last'
                - if 'last', the value of the last phase will be returned

        Returns:
            (str or int or float)

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        name = self.MAIN if name == "Main" else name
        if name not in self.series_dict.keys():
            raise KeyError(f"@name {name} scenario has not been registered.")
        df = self.series_dict[name].summary()
        if param not in df.columns:
            raise KeyError(f"@param must be in {', '.join(df.columns)}.")
        if phase == "last":
            phase = df.index[-1]
        return df.loc[phase, param]

    def param_history(self, targets=None, name="Main", divide_by_first=True,
                      show_figure=True, filename=None, show_box_plot=True, **kwargs):
        """
        Return subset of summary.

        Args:
            targets (list[str]/str): parameters to show (Rt etc.)
            name (str): phase series name
            divide_by_first (bool): if True, divide the values by 1st phase's values
            show_box_plot (bool): if True, box plot. if False, line plot.
            show_figure (bool): If True, show the result as a figure.
            filename (str): filename of the figure, or None (show figure)
            kwargs: keword arguments of pd.DataFrame.plot or line_plot()

        Returns:
            (pandas.DataFrame)

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        if "box_plot" in kwargs.keys():
            raise KeyError("Please use 'show_box_plot', not 'box_plot'")
        name = self.MAIN if name == "Main" else name
        if name not in self.series_dict.keys():
            raise KeyError(f"@name {name} scenario has not been registered.")
        df = self.series_dict[name].summary()
        model_param_nest = [m.PARAMETERS for m in self.model_dict.values()]
        model_day_nest = [m.DAY_PARAMETERS for m in self.model_dict.values()]
        model_parameters = self.flatten(model_param_nest)
        model_day_params = self.flatten(model_day_nest)
        selectable_cols = [
            self.N, *model_parameters, self.RT, *model_day_params
        ]
        targets = [targets] if isinstance(targets, str) else targets
        targets = selectable_cols if targets is None else targets
        if not set(targets).issubset(set(selectable_cols)):
            raise KeyError(
                f"@targets must be a subset of {', '.join(selectable_cols)}."
            )
        df = df.loc[:, targets]
        if divide_by_first:
            df = df / df.iloc[0, :]
            title = f"{self.area}: Ratio to 1st phase parameters ({name} scenario)"
        else:
            title = f"{self.area}: History of parameter values ({name} scenario)"
        if show_box_plot:
            h_values = [1.0] if divide_by_first or self.RT in targets else None
            box_plot(df, title, h=h_values, filename=filename)
            return df
        _df = df.reset_index(drop=True)
        _df.index = _df.index + 1
        h = 1.0 if divide_by_first else None
        line_plot(
            _df, title=title,
            xlabel="Phase", ylabel=str(), math_scale=False, h=h,
            show_figure=show_figure, filename=filename
        )
        return df

    def describe(self):
        """
        Describe representative values.

        Returns:
            (pandas.DataFrame)
                Index:
                    (int): scenario name
                Columns:
                    - max(Infected): max value of Infected
                    - argmax(Infected): the date when Infected shows max value
                    - Infected({date}): Infected on the end date of the last phase
                    - Fatal({date}): Fatal on the end date of the last phase
        """
        _dict = dict()
        for (name, _) in self.series_dict.items():
            # Predict the number of cases
            df = self.predict(name=name, show_figure=False)
            df = df.set_index(self.DATE)
            last_date = df.index[-1]
            # Max value of Infected
            max_ci = df[self.CI].max()
            argmax_ci = df[self.CI].idxmax()
            # Infected on the end date of the last phase
            last_ci = df.loc[last_date, self.CI]
            # Fatal on the end date of the last phase
            try:
                last_f = df.loc[last_date, self.F]
            except KeyError:
                last_f = None
            # Save representative values
            _dict[name] = {
                f"max({self.CI})": max_ci,
                f"argmax({self.CI})": argmax_ci,
                f"{self.CI} on {last_date}": last_ci,
                f"{self.F} on {last_date}": last_f,
            }
        return pd.DataFrame.from_dict(_dict, orient="index")
