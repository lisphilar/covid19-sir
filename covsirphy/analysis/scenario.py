#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from datetime import timedelta
import functools
from inspect import signature
from multiprocessing import cpu_count, Pool
import sys
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.util.plotting import line_plot, box_plot
from covsirphy.util.stopwatch import StopWatch
from covsirphy.cleaning.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.population import PopulationData
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.phase_series import PhaseSeries
from covsirphy.phase.sr_change import ChangeFinder
from covsirphy.simulation.estimator import Estimator
from covsirphy.simulation.simulator import ODESimulator


class Scenario(Term):
    """
    Scenario analysis.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        country (str): country name
        province (str or None): province name
    """

    def __init__(self, jhu_data, population_data, country, province=None):
        # Population
        population_data = self.ensure_instance(
            population_data, PopulationData, name="population_data")
        self.population = population_data.value(country, province=province)
        # Records
        self.jhu_data = self.ensure_instance(
            jhu_data, JHUData, name="jhu_data")
        # Area name
        self.country = country
        self.province = province or self.UNKNOWN
        self.area = JHUData.area_name(country, province)
        # First/last date of the area
        df = jhu_data.subset(country=self.country, province=self.province)
        self._first_date = df[self.DATE].min().strftime(self.DATE_FORMAT)
        self._last_date = df[self.DATE].max().strftime(self.DATE_FORMAT)
        # Init
        self.tau = None
        # {model_name: model_class}
        self.model_dict = dict()
        # {scenario_name: PhaseSeries}
        self._init_phase_series()
        # {scenario: {phase: Estimator}}
        self.estimator_dict = {self.MAIN: dict()}

    def _init_phase_series(self):
        """
        Initialize dictionary of phase series.
        """
        self.series_dict = dict()
        self.series_dict[self.MAIN] = PhaseSeries(
            self._first_date, self._last_date, self.population
        )

    @property
    def first_date(self):
        """
        str: the first date of the records
        """
        return self._first_date

    @first_date.setter
    def first_date(self, date):
        if self.to_date_obj(date) >= self.to_date_obj(self._last_date):
            raise ValueError(
                f"@date must be under {self._last_date}, but {date} was applied.")
        self._first_date = date
        self._init_phase_series()

    @property
    def last_date(self):
        """
        str: the last date of the records
        """
        return self._last_date

    @last_date.setter
    def last_date(self, date):
        if self.to_date_obj(date) <= self.to_date_obj(self._first_date):
            raise ValueError(
                f"@date must be under {self._first_date}, but {date} was applied.")
        self._last_date = date
        self._init_phase_series()

    def records(self, show_figure=True, filename=None):
        """
        Return the records as a dataframe.

        Args:
            show_figure (bool): if True, show the records as a line-plot.
            filename (str): filename of the figure, or None (show figure)

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

        Notes:
            Records with Recovered > 0 will be selected.
        """
        df = self.jhu_data.subset(country=self.country, province=self.province)
        if not show_figure:
            return df
        line_plot(
            df.set_index(self.DATE).drop(self.C, axis=1),
            f"{self.area}: Cases over time",
            y_integer=True,
            filename=filename
        )
        return df

    def _new_phase_dates(self, end_date=None, days=None, name="Main"):
        """
        Decide start/end date of the new phase.

        Args:
            end_date (str or None): end date of the new phase
            days (int or None): the number of days to add
            name (str): phase series name, 'Main' or user-defined name

        Returns:
            self

        Notes:
            - If the phases series has not been registered, new phase series will be created.
            - Either @end_date or @days must be specified.
            - If @end_date and @days are None, the end date will be the last date of the records.
            - If both of @end_date and @days were specified, @end_date will be used.
        """
        self._ensure_name(name)
        # Calculate start date
        start_date = self.series_dict[name].next_date()
        # Calculate end date
        if days is None:
            return (start_date, end_date or self._last_date)
        days = self.ensure_natural_int(days, name="days")
        end_obj = self.date_obj(start_date) + timedelta(days=days)
        end_date = end_date or end_obj.strftime(self.DATE_FORMAT)
        return (start_date, end_date)

    def _last_model(self, name="Main"):
        """
        Return the model of the last phase.

        Raises:
            KeyError: model has not been registered

        Returns:
            covsirphy.ModelBase: ODE model
        """
        self._ensure_name(name)
        # Get name of the last model
        df = self.series_dict[name].summary()
        last_phase = df.index[-1]
        model_name = df.loc[last_phase, self.ODE]
        # Return model class
        model = self.model_dict[model_name]
        return model

    @deprecate(old="Scenario.add_phase()", new="Scenario.add()")
    def add_phase(self, **kwargs):
        return self.add(**kwargs)

    def add(self, name="Main", end_date=None, days=None,
            population=None, model=None, **kwargs):
        """
        Add a new phase.
        The start date will be the next date of the last registered phase.

        Args:
            name (str): phase series name, 'Main' or user-defined name
            end_date (str): end date of the new phase
            days (int): the number of days to add
            population (int or None): population value of the start date
            model (covsirphy.ModelBase orNone): ODE model
            kwargs: optional, keyword arguments of ODE model parameters, not including tau value.

        Returns:
            self

        Notes:
            - If the phases series has not been registered, new phase series will be created.
            - Either @end_date or @days must be specified.
            - If @end_date and @days are None, the end date will be the last date of the records.
            - If both of @end_date and @days were specified, @end_date will be used.
            - If @popultion is None, initial value will be used.
            - If @model is None, the model of the last phase will be used.
            - Tau will be fixed as the last phase's value.
            - kwargs: Default values are the parameter values of the last phase.
        """
        self._ensure_name(name)
        # Start/end date
        start_date, end_date = self._new_phase_dates(
            end_date=end_date, days=days, name=name)
        # Population
        population = self.ensure_natural_int(
            population or self.population, "population")
        # Model information is unnecessary if models are not registered in the old phases
        summary_df = self.series_dict[name].summary()
        if model is None and self.ODE not in summary_df.columns:
            self.series_dict[name].add(start_date, end_date, population)
            return self
        # Model
        model = model or self._last_model(name=name)
        model = self.ensure_subclass(model, ModelBase, name="model")
        model_param_dict = {
            param: summary_df.loc[summary_df.index[-1], param]
            for param in model.PARAMETERS
        }
        model_param_dict.update(kwargs)
        model_instance = model(population=population, **model_param_dict)
        # Set phase information with model information
        param_dict = {self.TAU: self.tau, self.ODE: model.NAME}
        param_dict.update(model_param_dict)
        param_dict[self.RT] = model_instance.calc_r0()
        param_dict.update(model_instance.calc_days_dict(self.tau))
        # Add phase with model information
        self.series_dict[name].add(
            start_date, end_date, population, **param_dict)
        return self

    def _ensure_name(self, name):
        """
        Ensure that the phases series is registered.
        If not registered, copy the main series.

        Args:
            name (str): phase series name
        """
        if name in self.series_dict.keys():
            return None
        # Phase series
        series = copy.deepcopy(self.series_dict[self.MAIN])
        series.clear(include_past=False)
        self.series_dict[name] = series
        # Estimators
        self.estimator_dict[name] = copy.deepcopy(
            self.estimator_dict[self.MAIN])

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

        Returns:
            self
        """
        self._ensure_name(name)
        self.series_dict[name].clear(include_past=include_past)
        self.series_dict[name].use_0th = True
        return self

    def _delete(self, phases=None, name="Main"):
        """
        Delete a phase of the phase series.

        Args:
            phase (list[str] or None): phase name
            name (str): name of phase series

        Returns:
            tuple or None
                - (str): the first date of the phases
                - (str): the last date of the phases

        Notes:
            If @phases is None, the phase series will be deleted and returns None.
        """
        # Clear main series or delete sub phase series
        if phases is None:
            if name == self.MAIN:
                self.clear(name=name, include_past=True)
                return None
            self.series_dict.pop(name)
            return None
        # Delete phases
        if not isinstance(phases, list):
            raise TypeError("@phases mut be a list of phase names.")
        first_date = self.get(self.START, name=name, phase=phases[0])
        last_date = self.get(self.END, name=name, phase=phases[-1])
        for phase in phases:
            self.series_dict[name].delete(phase)
        if "0th" in phases:
            self.series_dict[name].use_0th = False
        return (first_date, last_date)

    def delete(self, phases=None, name="Main"):
        """
        Delete a phase of the phase series.

        Args:
            phase (list[str] or None): phase name
            name (str): name of phase series

        Returns:
            self

        Notes:
            If @phases is None, the phase series will be deleted.
        """
        self._ensure_name(name)
        self._delete(phases=phases, name=name)
        return self

    def combine(self, phases, name="Main", population=None, **kwargs):
        """
        Combine the sequential phases as one phase.
        New phase name will be automatically determined.

        Args:
            phases (list[str]): list of sequential phases
            name (str, optional): name of phase series
            population (int): population value of the start date
            kwargs: keyword arguments to save as phase information

        Raises:
            TypeError: @phases is not a list

        Returns:
            self: instance of covsirphy.Scenario
        """
        self._ensure_name(name)
        first_date, last_date = self._delete(phases=phases, name=name)
        self.series_dict[name].add(
            first_date, last_date, population=population, **kwargs
        )
        if "0th" in phases:
            self.series_dict[name].use_0th = True
        self.series_dict[name].reset_phase_names()
        return self

    def separate(self, date, phase, name="Main", population=None, **kwargs):
        """
        Create a new phase with the change point.
        New phase name will be automatically determined.

        Args:
            date (str): change point, i.e. start date of the new phase
            phases (list[str]): list of sequential phases
            name (str, optional): name of phase series
            population (int): population value of the change point
            kwargs: keyword arguments to save as phase information

        Returns:
            self: instance of covsirphy.Scenario
        """
        self._ensure_name(name)
        population_old_phase = self.get(self.N, name=name, phase=phase)
        # Delete the phase that will be separated
        first_date, last_date = self._delete(phases=[phase], name=name)
        # Re-registration of the old phase
        end_obj = self.to_date_obj(date) - timedelta(days=1)
        self.series_dict[name].add(
            first_date, end_obj.strftime(self.DATE_FORMAT),
            population=population_old_phase, **kwargs
        )
        # Add new phase
        self.series_dict[name].add(
            date, last_date, population=population, **kwargs
        )
        # Reset phase names
        if phase == "0th":
            self.series_dict[name].use_0th = True
        self.series_dict[name].reset_phase_names()
        return self

    def _summary(self, name=None):
        """
        Summarize the series of phases and return a dataframe.

        Args:
            name (str): phase series name
                - name of alternative phase series registered by self.add()
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
        if not name and len(self.series_dict.keys()) == 1:
            name = self.MAIN
        self._ensure_name(name)
        series = self.series_dict[name]
        return series.summary()

    def summary(self, columns=None, name=None):
        """
        Summarize the series of phases and return a dataframe.

        Args:
            name (str): phase series name
                - name of alternative phase series registered by self.add()
                - if None, all phase series will be shown
            columns (list[str] or None): columns to show

        Returns:
            (pandas.DataFrame):
            - if @name not None, as the same as PhaseSeries().summary()
            - if @name is None, index will be phase series name and phase name

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
            If @columns is None, all columns will be shown.
        """
        df = self._summary(name=name)
        columns = columns or df.columns.tolist()
        if not isinstance(columns, list):
            raise TypeError("@columns must be None or a list of strings.")
        if not set(columns).issubset(set(df.columns)):
            raise KeyError(
                "Un-registered columns were selected as @columns. Please use {', '.join(df.columns)}."
            )
        return df.loc[:, columns]

    def trend(self, set_phases=True, include_init_phase=False, name="Main",
              show_figure=True, filename=None, **kwargs):
        """
        Perform S-R trend analysis and set phases.

        Args:
            set_phases (bool): if True, set phases automatically with S-R trend analysis
            include_init_phase (bool): whether use initial phase or not
            name (str): phase series name
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of ChangeFinder()

        Returns:
            self

        Notes:
            If @set_phase is True and@include_init_phase is False, initial phase will not be included.
        """
        if "n_points" in kwargs.keys():
            raise ValueError(
                "@n_points argument is un-necessary"
                " because the number of change points will be automatically determined."
            )
        if not set_phases:
            use_0th = self.series_dict[name].use_0th
            init_phase = "0th" if use_0th else "1st"
            first_date = self.get(self.START, name=name, phase=init_phase)
            finder = ChangeFinder(
                self.jhu_data, self.population,
                country=self.country, province=self.province,
                start_date=first_date,
                end_date=self._last_date,
                **kwargs
            )
            finder.use_0th = use_0th
            finder.change_dates = self.series_dict[name].end_objects()[:-1]
            finder.show(show_figure=show_figure, filename=filename)
            return None
        finder = ChangeFinder(
            self.jhu_data, self.population,
            country=self.country, province=self.province,
            start_date=self._first_date, end_date=self._last_date,
            **kwargs
        )
        finder.run()
        phase_series = finder.show(show_figure=show_figure, filename=filename)
        if not include_init_phase:
            phase_series.delete("0th")
        self.series_dict[name] = copy.deepcopy(phase_series)
        return self

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
            (tuple): arguments of self._update_self

        Raises:
            ValueError: if @phase is None

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
            If @name phase was not registered, new PhaseSeries will be created.
        """
        # Phase name
        self._ensure_name(name)
        # Set parameters
        setting_dict = self.series_dict[name].to_dict()[phase]
        start_date = setting_dict[self.START]
        end_date = setting_dict[self.END]
        population = setting_dict[self.N]
        # Set tau value
        est_kwargs = {
            p: kwargs[p] for p in model.PARAMETERS if p in kwargs.keys()
        }
        if self.tau is not None:
            if self.TAU in kwargs.keys():
                raise ValueError(f"{self.TAU} cannot be changed.")
            est_kwargs[self.TAU] = self.tau
        # Run estimation
        estimator = Estimator(
            self.jhu_data, model, population,
            country=self.country, province=self.province,
            start_date=start_date, end_date=end_date,
            **est_kwargs
        )
        sign = signature(Estimator.run)
        run_params = list(sign.parameters.keys())
        run_kwargs = {k: v for (k, v) in kwargs.items() if k in run_params}
        estimator.run(stdout=False, **run_kwargs)
        # Get the summary of estimation
        est_df = estimator.summary(phase)
        phase_est_dict = {self.ODE: model.NAME}
        phase_est_dict.update(est_df.to_dict(orient="index")[phase])
        # Show the number of trials and runtime
        trials = phase_est_dict["Trials"]
        runtime = phase_est_dict["Runtime"]
        print(
            f"\t{phase} phase with {model.NAME} model finished {trials} trials in {runtime}."
        )
        # Return the dictionary of the result of estimation
        return (model, name, phase, estimator, phase_est_dict)

    def _update_self(self, model, name, phase, estimator, phase_est_dict):
        """
        Update self with the result of estimation.

        Args:
            model (covsirphy.ModelBase): ODE model
            name (str): phase series name
            phase (str or None): phase name, like 1st, 2nd...
            estimator (covsirphy.Estimator): instance of estimator class
            phase_est_dict (dict): dictionary of the result of estimation

        Returns:
            self
        """
        self.tau = phase_est_dict[self.TAU]
        self.series_dict[name].update(phase, **phase_est_dict)
        self.estimator_dict[name][phase] = estimator
        self.model_dict[model.NAME] = model
        return self

    def estimate(self, model, phases=None, name="Main", n_jobs=-1, **kwargs):
        """
        Estimate the parameters of the model using the records.

        Args:
            model (covsirphy.ModelBase): ODE model
            phases (list[str]): list of phase names, like 1st, 2nd...
            name (str): phase series name
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()

        Notes:
            - If 'Main' was used as @name, main PhaseSeries will be used.
            - If @name phase was not registered, new PhaseSeries will be created.
            - If @phases is None, all past phase will be used.
        """
        # Check model
        model = self.ensure_subclass(model, ModelBase, "model")
        # Validate the phases
        self._ensure_name(name)
        phase_dict = self.series_dict[name].to_dict()
        past_phases = list(phase_dict.keys())
        phases = past_phases[:] if phases is None else phases
        if not isinstance(phases, list):
            raise TypeError("@phases must be None or a list of phase names.")
        future_phases = list(set(phases) - set(past_phases))
        if future_phases:
            raise KeyError(
                f"{future_phases[0]} is not a past phase or not registered.")
        # Confirm that phases are registered
        if not phases:
            raise ValueError(
                "Scenario.trend(set_phases=True) or Scenario.add() must be done in advance.")
        # The number of parallel jobs
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        # Start optimization
        print(f"\n<{name} scenario: perform parameter estimation>")
        print(f"Running optimization with {n_jobs} CPUs...")
        stopwatch = StopWatch()
        # Estimation of the last phase will be done to determine tau value
        phase_sel, phases = phases[-1], phases[:-1]
        result_tuple_sel = self._estimate(model, phase=phase_sel, **kwargs)
        self._update_self(*result_tuple_sel)
        # Estimation of each phase
        est_f = functools.partial(self._estimate, model, **kwargs)
        with Pool(n_jobs) as p:
            result_nest = p.map(est_f, phases)
        for result_tuple in result_nest:
            self._update_self(*result_tuple)
        # Completion
        stopwatch.stop()
        print(f"Completed optimization. Total: {stopwatch.show()}")

    def _phase_estimator(self, phase, name):
        """
        Return the estimator of the phase.

        Args:
            phase (str): phase name, like 1st, 2nd...
            name (str): phase series name

        Return:
            covsirphy.Estimator: estimator of the phase
        """
        try:
            return self.estimator_dict[name][phase]
        except KeyError:
            raise KeyError(
                f'Scenario.estimate(model, phases=["{phase}"], name={name}) must be done in advance.'
            )

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
        estimator = self._phase_estimator(phase=phase, name=name)
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
        estimator = self._phase_estimator(phase=phase, name=name)
        estimator.accuracy(**kwargs)

    def simulate(self, name="Main", y0_dict=None, show_figure=True, filename=None):
        """
        Simulate ODE models with set parameter values and show it as a figure.

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
        self._ensure_name(name)
        df = self.series_dict[name].summary()
        # Future phases must be added in advance
        if self.FUTURE not in df[self.TENSE].unique():
            raise KeyError(
                f"Future phases of {name} scenario must be registered by Scenario.add() in advance."
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
        Simulate ODE models with set parameter values.

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
        phase_series = copy.deepcopy(self.series_dict[name])
        # Dates and the number of steps
        last_object = phase_series.last_object()
        start_objects = phase_series.start_objects()
        step_n_list = phase_series.number_of_steps(
            start_objects, last_object, self.tau)
        first_date = start_objects[0].strftime(self.DATE_FORMAT)
        # Information of models
        models = [
            self.model_dict[name] for name in phase_series.model_names()
        ]
        # Population values
        population_values = phase_series.population_values()
        # Create simulator
        simulator = ODESimulator(self.country, province=self.province)
        # Add phases to the simulator
        df = phase_series.summary()
        zip_iter = zip(df.index, models, step_n_list, population_values)
        for (i, (phase, model, step_n, population)) in enumerate(zip_iter):
            param_dict = df[model.PARAMETERS].to_dict(orient="index")[phase]
            if i == 0:
                # Calculate initial values
                subset_df = self.jhu_data.subset(
                    country=self.country, province=self.province,
                    start_date=first_date
                )
                subset_df = model.tau_free(subset_df, population, tau=None)
                y0_dict_phase = {
                    k: subset_df.loc[subset_df.index[0], k] for k in model.VARIABLES
                }
            else:
                y0_dict_phase = None if y0_dict is None else y0_dict.copy()
            simulator.add(
                model, step_n, population,
                param_dict=param_dict,
                y0_dict=y0_dict_phase
            )
        # Run simulation
        simulator.run()
        dim_df = simulator.dim(self.tau, first_date)
        return dim_df, start_objects

    def get(self, param, phase="last", name="Main"):
        """
        Get the parameter value of the phase.

        Args:
            param (str): parameter name (columns in self.summary())
            phase (str): phase name or 'last'
                - if 'last', the value of the last phase will be returned
            name (str): phase series name

        Returns:
            (str or int or float)

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        self._ensure_name(name)
        df = self.series_dict[name].summary()
        if param not in df.columns:
            raise KeyError(f"@param must be in {', '.join(df.columns)}.")
        if phase == "last":
            phase = df.index[-1]
        return df.loc[phase, param]

    def _param_history(self, targets, name):
        """
        Return the subset of summary dataframe to select the target of parameter history.

        Args:
            targets (list[str] or str): parameters to show (Rt etc.)
            name (str): phase series name

        Returns:
            (pandas.DataFrame): selected summary dataframe

        Raises:
            KeyError: targets are not in the columns of summary dataframe
        """
        df = self.series_dict[name].summary()
        model_param_nest = [m.PARAMETERS for m in self.model_dict.values()]
        model_day_nest = [m.DAY_PARAMETERS for m in self.model_dict.values()]
        model_parameters = self.flatten(model_param_nest)
        model_day_params = self.flatten(model_day_nest)
        selectable_cols = [
            self.N, *model_parameters, self.RT, *model_day_params
        ]
        selectable_set = set(selectable_cols)
        if not selectable_set.issubset(set(df.columns)):
            raise ValueError(
                f"Scenario.estimate(model, phases=None, name={name}) must be done in advance.")
        targets = [targets] if isinstance(targets, str) else targets
        targets = targets or selectable_cols
        if not set(targets).issubset(selectable_set):
            raise KeyError(
                f"@targets must be selected from {', '.join(selectable_cols)}."
            )
        df = df.loc[:, targets]
        return df

    def param_history(self, targets=None, name="Main", divide_by_first=True,
                      show_figure=True, filename=None, show_box_plot=True, **kwargs):
        """
        Return subset of summary.

        Args:
            targets (list[str] or str): parameters to show (Rt etc.)
            name (str): phase series name
            divide_by_first (bool): if True, divide the values by 1st phase's values
            show_box_plot (bool): if True, box plot. if False, line plot
            show_figure (bool): If True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keword arguments of pd.DataFrame.plot or line_plot()

        Returns:
            (pandas.DataFrame)

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        # Check arguments
        if "box_plot" in kwargs.keys():
            raise KeyError("Please use 'show_box_plot', not 'box_plot'")
        self._ensure_name(name)
        # Select target to show
        df = self._param_history(targets, name)
        # Divide by the first phase parameters
        if divide_by_first:
            df = df / df.iloc[0, :]
            title = f"{self.area}: Ratio to 1st phase parameters ({name} scenario)"
        else:
            title = f"{self.area}: History of parameter values ({name} scenario)"
        if not show_figure:
            return df
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
            filename=filename
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
            df = self.simulate(name=name, show_figure=False)
            df = df.set_index(self.DATE)
            cols = df.columns[:]
            last_date = df.index[-1]
            # Max value of Infected
            max_ci = df[self.CI].max()
            argmax_ci = df[self.CI].idxmax()
            # Infected on the end date of the last phase
            last_ci = df.loc[last_date, self.CI]
            # Fatal on the end date of the last phase
            last_f = df.loc[last_date, self.F] if self.F in cols else None
            # Save representative values
            _dict[name] = {
                f"max({self.CI})": max_ci,
                f"argmax({self.CI})": argmax_ci,
                f"{self.CI} on {last_date}": last_ci,
                f"{self.F} on {last_date}": last_f,
            }
        return pd.DataFrame.from_dict(_dict, orient="index")
