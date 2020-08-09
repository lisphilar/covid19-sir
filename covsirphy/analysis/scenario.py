#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from datetime import timedelta
import functools
from multiprocessing import cpu_count, get_context
import sys
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.util.plotting import line_plot, box_plot
from covsirphy.util.stopwatch import StopWatch
from covsirphy.cleaning.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.population import PopulationData
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.phase_unit import PhaseUnit
from covsirphy.phase.phase_series import PhaseSeries


class Scenario(Term):
    """
    Scenario analysis.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        country (str): country name
        province (str or None): province name
        tau (int or None): tau value
    """

    def __init__(self, jhu_data, population_data, country, province=None, tau=None):
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
        # tau value must be shared
        self.tau = self.ensure_tau(tau)
        # {scenario_name: PhaseSeries}
        self._init_phase_series()

    def _init_phase_series(self):
        """
        Initialize dictionary of phase series.
        """
        self.series_dict = {
            self.MAIN: PhaseSeries(
                self._first_date, self._last_date, self.population, use_0th=False
            )
        }
        self.record_df = self.jhu_data.subset(
            country=self.country,
            province=self.province,
            start_date=self._first_date,
            end_date=self._last_date,
            population=self.population
        )

    @property
    def first_date(self):
        """
        str: the first date of the records
        """
        return self._first_date

    @first_date.setter
    def first_date(self, date):
        if self.date_obj(date) >= self.date_obj(self._last_date):
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
        if self.date_obj(date) <= self.date_obj(self._first_date):
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

    def last_phase(self, name="Main"):
        """
        Return the last phase unit of the phase series.

        Args:
            name (str): phase series name, 'Main' or user-defined name
        """
        phase_series = self._ensure_name(name)
        return phase_series.phase("last")

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
            model (covsirphy.ModelBase or None): ODE model
            kwargs: keyword arguments of ODE model parameters, not including tau value.

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
        try:
            last_phase_unit = self.last_phase(name=name)
        except KeyError:
            last_phase_unit = PhaseUnit(
                self._first_date, self._last_date, self.population)
        # Population
        if population is None:
            population = last_phase_unit.population
        population = self.ensure_population(population)
        # Model information is unnecessary if models are not registered in the old phases
        model = model or last_phase_unit.model
        if model is None:
            self.series_dict[name].add(
                end_date=end_date, days=days, population=population)
            return self
        # Model
        model = self.ensure_subclass(model, ModelBase, name="model")
        ode_dict = last_phase_unit.ode_dict.copy()
        ode_dict.update(kwargs)
        self.series_dict[name].add(
            end_date=end_date, days=days, population=population,
            model=model, **ode_dict
        )
        return self

    def _ensure_name(self, name):
        """
        Ensure that the phases series is registered.
        If not registered, copy the main series.

        Args:
            name (str): phase series name
        """
        if name in self.series_dict.keys():
            return self.series_dict[name]
        # Phase series
        series = copy.deepcopy(self.series_dict[self.MAIN])
        series.clear(include_past=False)
        self.series_dict[name] = series
        return series

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
        self._ensure_name(name)
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
            self.series_dict[name].use_0th = True
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
        first_date, last_date = self._delete(phases=phases, name=name)
        self.series_dict[name].add(
            start_date=first_date, end_date=last_date, population=population, **kwargs
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
        end_obj = self.date_obj(date) - timedelta(days=1)
        self.series_dict[name].add(
            start_date=first_date,
            end_date=end_obj.strftime(self.DATE_FORMAT),
            population=population_old_phase,
            **kwargs
        )
        # Add new phase
        self.series_dict[name].add(
            start_date=date, end_date=last_date, population=population, **kwargs
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
                - name of alternative phase series registered by Scenario.add()
                - if None, all phase series will be shown

        Returns:
            (pandas.DataFrame):
            - if @name not None, as the same as PhaseSeries().summary()
            - if @name is None, index will be phase series name and phase name

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        if name is None and len(self.series_dict.keys()) > 1:
            dataframes = []
            for (name, series) in self.series_dict.items():
                summary_df = series.summary()
                summary_df = summary_df.rename_axis(self.PHASE)
                summary_df[self.SERIES] = name
                dataframes.append(summary_df.reset_index())
            df = pd.concat(dataframes, ignore_index=True, sort=False)
            return df.set_index([self.SERIES, self.PHASE])
        if not name and len(self.series_dict.keys()) == 1:
            name = self.MAIN
        series = self._ensure_name(name)
        return series.summary()

    def summary(self, columns=None, name=None):
        """
        Summarize the series of phases and return a dataframe.

        Args:
            name (str): phase series name
                - name of alternative phase series registered by Scenario.add()
                - if None, all phase series will be shown
            columns (list[str] or None): columns to show

        Returns:
            pandas.DataFrame:
            - if @name not None, as the same as PhaseSeries().summary()
            - if @name is None, index will be phase series name and phase name

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
            If @columns is None, all columns will be shown.
        """
        df = self._summary(name=name)
        all_cols = df.columns.tolist()
        if set(self.EST_COLS).issubset(set(all_cols)):
            all_cols = [col for col in all_cols if col not in self.EST_COLS]
            all_cols += self.EST_COLS
        columns = columns or all_cols
        if not isinstance(columns, list):
            raise TypeError("@columns must be None or a list of strings.")
        if not set(columns).issubset(set(df.columns)):
            raise KeyError(
                "Un-registered columns were selected as @columns. Please use {', '.join(df.columns)}."
            )
        df = df.loc[:, columns]
        return df.dropna(how="all", axis=1).fillna(self.UNKNOWN)

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
            covsirphy.Scenario: self

        Notes:
            If @set_phase is True and@include_init_phase is False, initial phase will not be included.
        """
        if "n_points" in kwargs.keys():
            raise ValueError(
                "@n_points argument is un-necessary"
                " because the number of change points will be automatically determined."
            )
        self._ensure_name(name)
        sr_df = self.jhu_data.to_sr(
            country=self.country, province=self.province, population=self.population
        )
        self.series_dict[name].use_0th = include_init_phase
        self.series_dict[name].trend(
            sr_df=sr_df,
            set_phases=set_phases,
            area=self.area,
            show_figure=show_figure,
            filename=filename,
            **kwargs
        )
        return self

    def _estimate(self, phase, model, series, **kwargs):
        """
        Estimate the parameters of the model using the records.

        Args:
            phase (str): phase name, like 1st, 2nd...
            model (covsirphy.ModelBase): ODE model
            series (covsirphy.PhaseSeries): phase series
            kwargs:
                - keyword arguments of the model parameter
                    - tau value cannot be included
                - keyword arguments of covsirphy.Estimator.run()

        Returns:
            (tuple): arguments of self._update_self

        Notes:
            If 'Main' was used as @name, main PhaseSeries will be used.
            If @name phase was not registered, new PhaseSeries will be created.
        """
        # Set parameters
        phase_unit = series.phase(phase)
        phase_unit.set_ode(model=model, tau=self.tau, **kwargs)
        # Records
        record_df = self.jhu_data.subset(
            country=self.country,
            province=self.province,
            start_date=phase_unit.start_date,
            end_date=self.tomorrow(phase_unit.end_date)
        )
        # Estimation
        phase_unit.estimate(record_df=record_df, **kwargs)
        # Show the number of trials and runtime
        trials = phase_unit.estimator.total_trials
        runtime = phase_unit.estimator.run_time_show
        print(
            f"\t{phase} phase with {model.NAME} model: finished {trials} trials in {runtime}"
        )
        return phase_unit

    def _ensure_past_phases(self, phases=None, name="Main"):
        """
        Ensure that the phases are past phases.

        Args:
            phases (list[str]): list of phase names, like 1st, 2nd...
            name (str): phase series name

        Returns:
            tuple(covsirphy.PhaseSeries, list[str]): phase series and list of past phase names

        Notes:
            If @phases is None, return the all past phases.
        """
        series = self._ensure_name(name)
        past_phases = series.phases(include_future=False)
        if not past_phases:
            raise ValueError(
                "Scenario.trend(set_phases=True) or Scenario.add() must be done in advance.")
        if phases is None:
            return (series, past_phases)
        if not isinstance(phases, list):
            raise TypeError("@phases must be None or a list of phase names.")
        future_phases = list(set(phases) - set(past_phases))
        if future_phases:
            raise KeyError(
                f"{future_phases[0]} is not a past phase or not registered.")
        return (series, phases)

    def estimate(self, model, phases=None, name="Main", n_jobs=-1, stdout=True, **kwargs):
        """
        Estimate the parameters of the model using the records.

        Args:
            model (covsirphy.ModelBase): ODE model
            phases (list[str]): list of phase names, like 1st, 2nd...
            name (str): phase series name
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            stdout (bool): whether show the status of progress or not
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()

        Notes:
            - If 'Main' was used as @name, main PhaseSeries will be used.
            - If @name phase was not registered, new PhaseSeries will be created.
            - If @phases is None, all past phase will be used.
            - In kwargs, tau value cannot be included.
        """
        model = self.ensure_subclass(model, ModelBase, "model")
        series, phases = self._ensure_past_phases(phases=phases, name=name)
        # tau value must be specified in Scenario.__init__
        if self.TAU in kwargs:
            raise ValueError(
                "@tau must be specified when scenario = Scenario(), and cannot be specified here.")
        # The number of parallel jobs
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        # Start optimization
        if stdout:
            print(f"\n<{name} scenario: parameter estimation>")
            print(f"Running optimization with {n_jobs} CPUs...")
            stopwatch = StopWatch()
        # Estimation of the last phase will be done to determine tau value
        if self.tau is None:
            phase_sel, phases = phases[-1], phases[:-1]
            phase_unit_sel = self._estimate(
                model=model, phase=phase_sel, series=series, stdout=False, **kwargs)
            self.tau = phase_unit_sel.tau
            self.series_dict[name].replace(phase_sel, phase_unit_sel)
        # Estimation of each phase
        est_f = functools.partial(
            self._estimate, model=model, series=series, stdout=False, **kwargs)
        with get_context("spawn").Pool(n_jobs) as p:
            units = p.map(est_f, phases)
        for (phase, phase_unit) in zip(phases, units):
            self.series_dict[name].replace(phase, phase_unit)
        # Completion
        if stdout:
            stopwatch.stop()
            print(f"Completed optimization. Total: {stopwatch.show()}")

    def phase_estimator(self, phase, name="Main"):
        """
        Return the estimator of the phase.

        Args:
            phase (str): phase name, like 1st, 2nd...
            name (str): phase series name

        Return:
            covsirphy.Estimator: estimator of the phase
        """
        try:
            return self.series_dict[name].phase(phase).estimator
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
        estimator = self.phase_estimator(phase=phase, name=name)
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
        estimator = self.phase_estimator(phase=phase, name=name)
        estimator.accuracy(**kwargs)

    def simulate(self, name="Main", y0_dict=None, show_figure=True, filename=None):
        """
        Simulate ODE models with set parameter values and show it as a figure.

        Args:
            name (str): phase series name. If 'Main', main PhaseSeries will be used
            y0_dict (dict or None):
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
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - variables of the models (int): Confirmed (int) etc.
        """
        series = self._ensure_name(name)
        # Simulation
        sim_df = series.simulate(
            record_df=self.record_df, tau=self.tau, y0_dict=y0_dict)
        if not show_figure:
            return sim_df
        # Show figure
        fig_cols_set = set(sim_df.columns) & set(self.FIG_COLUMNS)
        fig_cols = [col for col in self.FIG_COLUMNS if col in fig_cols_set]
        line_plot(
            sim_df[fig_cols],
            title=f"{self.area}: Predicted number of cases ({name} scenario)",
            filename=filename,
            y_integer=True,
            v=series.start_objects()[1:]
        )
        return sim_df

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
        df = self._ensure_name(name).summary()
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
        series = self.series_dict[name]
        model_set = {series.phase(phase).model for phase in series.phases()}
        model_set = model_set - set([None])
        parameters = self.flatten([m.PARAMETERS for m in model_set])
        day_params = self.flatten([m.DAY_PARAMETERS for m in model_set])
        selectable_cols = [self.N, *parameters, self.RT, *day_params]
        selectable_set = set(selectable_cols)
        df = series.summary()
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
        Return subset of summary and show a figure to show the history.

        Args:
            targets (list[str] or str): parameters to show (Rt etc.)
            name (str): phase series name
            divide_by_first (bool): if True, divide the values by 1st phase's values
            show_box_plot (bool): if True, box plot. if False, line plot
            show_figure (bool): If True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of pd.DataFrame.plot or line_plot()

        Returns:
            pandas.DataFrame

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

    def describe(self, y0_dict=None):
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
            y0_dict (dict or None):
                - key (str): variable name
                - value (float): initial value
                - dictionary of initial values or None
                - if model will be changed in the later phase, must be specified
        """
        _dict = {}
        for (name, _) in self.series_dict.items():
            # Predict the number of cases
            df = self.simulate(name=name, y0_dict=y0_dict, show_figure=False)
            df = df.set_index(self.DATE)
            cols = df.columns[:]
            last_date = df.index[-1]
            # Max value of Infected
            max_ci = df[self.CI].max()
            argmax_ci = df[self.CI].idxmax().strftime(self.DATE_FORMAT)
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

    def phases(self, name="Main"):
        """
        Return the list of phase names.

        Args:
            name (str): phase series name

        Returns:
            (list[int]): list of phase names
        """
        return self._ensure_name(name).phases()
