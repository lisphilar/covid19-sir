#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from datetime import timedelta
from math import log10, floor
import warnings
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from covsirphy.util.error import deprecate, ScenarioNotFoundError, UnExecutedError
from covsirphy.util.error import NotRegisteredMainError, NotRegisteredExtraError
from covsirphy.util.plotting import line_plot, box_plot
from covsirphy.util.error import NotInteractiveError
from covsirphy.util.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.analysis.param_tracker import ParamTracker
from covsirphy.analysis.data_handler import DataHandler


class Scenario(Term):
    """
    Scenario analysis.

    Args:
        jhu_data (covsirphy.JHUData or None): object of records
        population_data (covsirphy.PopulationData or None): PopulationData object
        country (str): country name (must not be None)
        province (str or None): province name
        tau (int or None): tau value
        auto_complement (bool): if True and necessary, the number of cases will be complemented

    Note:
        @jhu_data and @population_data must be registered with Scenario.register() if not specified here.
    """

    def __init__(self, jhu_data=None, population_data=None,
                 country=None, province=None, tau=None, auto_complement=True):
        # Area name
        if country is None:
            raise ValueError("@country must be specified.")
        self.country = str(country)
        self.province = str(province or self.UNKNOWN)
        self.area = JHUData.area_name(country, province)
        # Initialize data handler
        self._data = DataHandler(country=self.country, province=self.province)
        self._data.switch_complement(whether=auto_complement)
        # Tau value
        self.tau = self._ensure_tau(tau)
        # Register datasets
        self._tracker_dict = {}
        self.register(jhu_data=jhu_data, population_data=population_data, extras=None)
        # Initialize parameter tracker
        try:
            self._init_trackers()
        except NotRegisteredMainError:
            pass
        # Interactive (True) / script (False) mode
        self._interactive = hasattr(sys, "ps1")
        # Prediction of parameter values in the future phases: {name: (regression model, X_target)}
        self._lm_dict = {}

    def __getitem__(self, key):
        """
        Return the phase series object for the scenario.

        Args:
            key (str): scenario name

        Raises:
            ScenarioNotFoundError: the scenario is not registered

        Returns:
            covsirphy.PhaseSeries
        """
        if key in self._tracker_dict:
            return self._tracker_dict[key].series
        raise ScenarioNotFoundError(key)

    def __setitem__(self, key, value):
        """
        Register a phase series.

        Args:
            key (str): scenario name
            value (covsirphy.PhaseSeries): phase series object
        """
        self._tracker_dict[key] = ParamTracker(
            self._data.records(extras=False), value, area=self.area, tau=self.tau)

    @property
    def first_date(self):
        """
        str: the first date of the records
        """
        return self._data.first_date

    @first_date.setter
    def first_date(self, date):
        self._data.timepoints(
            first_date=date, last_date=self._data.last_date, today=self._data.today)

    @property
    def last_date(self):
        """
        str: the last date of the records
        """
        return self._data.last_date

    @last_date.setter
    def last_date(self, date):
        try:
            self._ensure_date_order(self._data.today, date)
        except ValueError:
            today = date
        else:
            today = self._data.today
        self.timepoints(
            first_date=self._data.first_date, last_date=date, today=today)

    @property
    def today(self):
        """
        str: reference date to determine whether a phase is a past phase or a future phase
        """
        return self._data.today

    @today.setter
    def today(self, date):
        self.timepoints(
            first_date=self._data.first_date, last_date=self._data.last_date, today=date)

    @property
    def interactive(self):
        """
        bool: interactive mode (display figures) or not

        Note:
            When running scripts, interactive mode cannot be selected.
        """
        return self._interactive

    @interactive.setter
    def interactive(self, is_interactive):
        if not hasattr(sys, "ps1") and is_interactive:
            raise NotInteractiveError
        self._interactive = hasattr(sys, "ps1") and bool(is_interactive)

    def register(self, jhu_data=None, population_data=None, extras=None):
        """
        Register datasets.

        Args:
            jhu_data (covsirphy.JHUData or None): object of records
            population_data (covsirphy.PopulationData or None): PopulationData object
            extras (list[covsirphy.CleaningBase] or None): extra datasets

        Raises:
            TypeError: non-data cleaning instance was included
            UnExpectedValueError: instance of un-expected data cleaning class was included as an extra dataset
        """
        self._data.register(jhu_data=jhu_data, population_data=population_data, extras=extras)
        if self._data.main_satisfied and not self._tracker_dict:
            self.timepoints()

    def timepoints(self, first_date=None, last_date=None, today=None):
        """
        Set the range of data and reference date to determine past/future of phases.

        Args:
            first_date (str or None): the first date of the records or None (min date of main dataset)
            last_date (str or None): the first date of the records or None (max date of main dataset)
            today (str or None): reference date to determine whether a phase is a past phase or a future phase

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data

        Note:
            When @today is None, the reference date will be the same as @last_date (or max date).
        """
        self._data.timepoints(first_date=first_date, last_date=last_date, today=today)
        self._init_trackers()

    def line_plot(self, df, show_figure=True, filename=None, **kwargs):
        """
        Display or save a line plot of the dataframe.

        Args:
            show_figure (bool): whether show figure when interactive mode or not
            filename (str or None): filename of the figure or None (not save) when script mode

        Note:
            When interactive mode and @show_figure is True, display the figure.
            When script mode and filename is not None, save the figure.
            When using interactive shell, we can change the modes by Scenario.interactive = True/False.
        """
        if self._interactive and show_figure:
            return line_plot(df=df, filename=None, **kwargs)
        if not self._interactive and filename is not None:
            return line_plot(df=df, filename=filename, **kwargs)

    def complement(self, **kwargs):
        """
        Complement the number of recovered cases, if necessary.

        Args:
            kwargs: the other arguments of JHUData.subset_complement()

        Returns:
            covsirphy.Scenario: self
        """
        self._data.switch_complement(whether=True, **kwargs)
        return self

    def complement_reverse(self):
        """
        Restore the raw records. Reverse method of covsirphy.Scenario.complement().

        Returns:
            covsirphy.Scenario: self
        """
        self._data.switch_complement(whether=False)
        return self

    def show_complement(self, **kwargs):
        """
        Show the details of complement that was (or will be) performed for the records.

        Args:
            kwargs: keyword arguments of JHUDataComplementHandler() i.e. control factors of complement

        Returns:
            pandas.DataFrame: as the same as JHUData.show_complement()
        """
        self._data.switch_complement(whether=None, **kwargs)
        return self._data.show_complement()

    def records(self, variables=None, **kwargs):
        """
        Return the records as a dataframe.

        Args:
            variables (list[str] or str or None): list of variables, 'all', None (Infected/Fatal/Recovered)
            kwargs: the other keyword arguments of Scenario.line_plot()

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data
            NotRegisteredExtraError: some variables are not included in the main datasets
            and no extra datasets were registered

        Returns:
            pandas.DataFrame

                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Columns set by @variables (int)

        Note:
            - Records with Recovered > 0 will be selected.
            - If complement was performed by Scenario.complement() or Scenario(auto_complement=True),
            The kind of complement will be added to the title of the figure.
            - @variables can be selected from Susceptible/Confirmed/Infected/Fatal/Recovered.
        """
        # Get necessary data for the variables
        all_df = self._data.records_all()
        if variables is None:
            df = all_df.loc[:, [self.DATE, self.CI, self.F, self.R]]
        elif variables == "all":
            df = all_df.copy()
        else:
            self._ensure_list(variables, candidates=all_df.columns.tolist(), name="variables")
            df = all_df.loc[:, [self.DATE, *variables]]
        # Figure
        if self._data.complemented:
            title = f"{self.area}: Cases over time\nwith {self._data.complemented}"
        else:
            title = f"{self.area}: Cases over time"
        self.line_plot(
            df=df.set_index(self.DATE), title=title, y_integer=True, **kwargs)
        return df

    def records_diff(self, variables=None, window=7, **kwargs):
        """
        Return the number of daily new cases (the first discreate difference of records).

        Args:
            variables (list[str] or str or None): list of variables, 'all', None (Infected/Fatal/Recovered)
            window (int): window of moving average, >= 1
            kwargs: the other keyword arguments of Scenario.line_plot()

        Returns:
            pandas.DataFrame
                Index
                    - Date (pd.TimeStamp): Observation date
                Columns
                    - Confirmed (int): daily new cases of Confirmed, if calculated
                    - Infected (int):  daily new cases of Infected, if calculated
                    - Fatal (int):  daily new cases of Fatal, if calculated
                    - Recovered (int):  daily new cases of Recovered, if calculated

        Note:
            @variables will be selected from Confirmed, Infected, Fatal and Recovered.
            If None was set as @variables, ["Confirmed", "Fatal", "Recovered"] will be used.
        """
        window = self._ensure_natural_int(window, name="window")
        df = self.records(variables=variables, show_figure=False).set_index(self.DATE)
        df = df.diff().dropna()
        df = df.rolling(window=window).mean().dropna().astype(np.int64)
        if self._data.complemented:
            title = f"{self.area}: Daily new cases\nwith {self._data.complemented}"
        else:
            title = f"{self.area}: Daily new cases"
        self.line_plot(df=df, title=title, y_integer=True, **kwargs)
        return df

    def _init_trackers(self):
        """
        Initialize dictionary of trackers.
        """
        data = copy.deepcopy(self._data)
        series = ParamTracker.create_series(
            first_date=data.first_date, last_date=data.last_date, population=data.population)
        tracker = ParamTracker(
            record_df=self._data.records(extras=False), phase_series=series, area=self.area, tau=self.tau)
        self._tracker_dict = {self.MAIN: tracker}

    def _tracker(self, name, template="Main"):
        """
        Ensure that the phases series is registered.
        If not registered, copy the template phase series.

        Args:
            name (str): phase series name
            template (str): name of template phase series

        Returns:
            covsirphy.ParamTracker
        """
        # Registered
        if name in self._tracker_dict:
            return self._tracker_dict[name]
        # Un-registered and create it
        if template not in self._tracker_dict:
            raise ScenarioNotFoundError(template)
        tracker = copy.deepcopy(self._tracker_dict[template])
        self._tracker_dict[name] = tracker
        return tracker

    @deprecate(old="Scenario.add_phase()", new="Scenario.add()")
    def add_phase(self, **kwargs):
        return self.add(**kwargs)

    def add(self, name="Main", end_date=None, days=None, population=None, model=None, **kwargs):
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
            covsirphy.Scenario: self

        Note:
            - If the phases series has not been registered, new phase series will be created.
            - Either @end_date or @days must be specified.
            - If @end_date and @days are None, the end date will be the last date of the records.
            - If both of @end_date and @days were specified, @end_date will be used.
            - If @popultion is None, initial value will be used.
            - If @model is None, the model of the last phase will be used.
            - Tau will be fixed as the last phase's value.
            - kwargs: Default values are the parameter values of the last phase.
        """
        if end_date is not None:
            self._ensure_date(end_date, name="end_date")
        tracker = self._tracker(name)
        try:
            tracker.add(
                end_date=end_date, days=days, population=population, model=model, **kwargs)
        except ValueError:
            last_date = tracker.series.unit("last").end_date
            raise ValueError(
                f'@end_date must be over {last_date}. However, {end_date} was applied.') from None
        self._tracker_dict[name] = tracker
        return self

    def clear(self, name="Main", include_past=False, template="Main"):
        """
        Clear phase information.

        Args:
            name (str): scenario name
            include_past (bool): if True, past phases will be removed as well as future phases
            template (str): name of template scenario

        Returns:
            covsirphy.Scenario: self

        Note:
            If un-registered scenario name was specified, new scenario will be created.
            Future phases will be always deleted.
        """
        tracker = self._tracker(name, template=template)
        if include_past:
            self[name] = tracker.delete_all()
        else:
            self[name] = tracker.delete(phases=tracker.future_phases()[0])
        return self

    def _delete_series(self, name):
        """
        Delete a scenario or initialise main scenario.

        Args:
            name (str): name of phase series

        Returns:
            covsirphy.Scenario: self
        """
        if name == self.MAIN:
            self[self.MAIN] = self._tracker(self.MAIN).delete_all()
        else:
            self._tracker_dict.pop(name)
        return self

    def delete(self, phases=None, name="Main"):
        """
        Delete phases.

        Args:
            phase (list[str] or None): phase names, or ['last']
            name (str): name of phase series

        Returns:
            covsirphy.Scenario: self

        Note:
            If @phases is None, the phase series will be deleted.
            When @phase is '0th', disable 0th phase. 0th phase will not be deleted.
            If the last phase is included in @phases, the dates will be released from phases.
            If the last phase is not included, the dates will be assigned to the previous phase.
        """
        # Clear main series or delete sub phase series
        if phases is None:
            return self._delete_series(name)
        # Delete phases
        tracker = self._tracker(name)
        self[name] = tracker.delete(phases=phases)
        return self

    def disable(self, phases, name="Main"):
        """
        The phases will be disabled and removed from summary.

        Args:
            phase (list[str] or None): phase names or None (all enabled phases)
            name (str): scenario name

        Returns:
            covsirphy.Scenario: self
        """
        self[name] = self._tracker(name).disable(phases)
        return self

    def enable(self, phases, name="Main"):
        """
        The phases will be enabled and appear in summary.

        Args:
            phase (list[str] or None): phase names or None (all disabled phases)
            name (str): scenario name

        Returns:
            covsirphy.Scenario: self
        """
        self[name] = self._tracker(name).enable(phases)
        return self

    def combine(self, phases, name="Main", population=None, **kwargs):
        """
        Combine the sequential phases as one phase.
        New phase name will be automatically determined.

        Args:
            phases (list[str]): list of phases
            name (str, optional): name of phase series
            population (int): population value of the start date
            kwargs: keyword arguments to save as phase information

        Raises:
            TypeError: @phases is not a list

        Returns:
            covsirphy.Scenario: self
        """
        self[name] = self._tracker(name).combine(
            phases=phases, population=population, **kwargs)
        return self

    def separate(self, date, name="Main", population=None, **kwargs):
        """
        Create a new phase with the change point.
        New phase name will be automatically determined.

        Args:
            date (str): change point, i.e. start date of the new phase
            name (str): scenario name
            population (int): population value of the change point
            kwargs: keyword arguments of PhaseUnit.set_ode() if update is necessary

        Returns:
            covsirphy.Scenario: self
        """
        self[name] = self._tracker(name).separate(
            date=date, population=population, **kwargs)
        return self

    def _summary(self, name=None):
        """
        Summarize the series of phases and return a dataframe.

        Args:
            name (str): phase series name
                - name of alternative phase series registered by Scenario.add()
                - if None, all phase series will be shown

        Returns:
            pandas.DataFrame:
            - if @name not None, as the same as PhaseSeries().summary()
            - if @name is None, index will be phase series name and phase name

        Note:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        if name is None:
            if len(self._tracker_dict.keys()) > 1:
                dataframes = []
                for (_name, tracker) in self._tracker_dict.items():
                    summary_df = tracker.series.summary()
                    summary_df = summary_df.rename_axis(self.PHASE)
                    summary_df[self.SERIES] = _name
                    dataframes.append(summary_df.reset_index())
                df = pd.concat(dataframes, ignore_index=True, sort=False)
                return df.set_index([self.SERIES, self.PHASE])
            name = self.MAIN
        return self._tracker(name).series.summary()

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

        Note:
            If 'Main' was used as @name, main PhaseSeries will be used.
            If @columns is None, all columns will be shown.
        """
        df = self._summary(name=name)
        all_cols = df.columns.tolist()
        if set(self.EST_COLS).issubset(all_cols):
            all_cols = [col for col in all_cols if col not in self.EST_COLS]
            all_cols += self.EST_COLS
        columns = columns or all_cols
        self._ensure_list(columns, candidates=all_cols, name="columns")
        df = df.loc[:, columns]
        return df.dropna(how="all", axis=1).fillna(self.UNKNOWN)

    def trend(self, force=True, name="Main", show_figure=True, filename=None, **kwargs):
        """
        Perform S-R trend analysis and set phases.

        Args:
            force (bool): if True, change points will be over-written
            name (str): phase series name
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (display)
            kwargs: keyword arguments of covsirphy.ChangeFinder() and covsirphy.line_plot_multiple()

        Returns:
            covsirphy.Scenario: self
        """
        # Arguments
        if "n_points" in kwargs.keys():
            raise ValueError(
                "@n_points argument is un-necessary"
                " because the number of change points will be automatically determined.")
        try:
            include_init_phase = kwargs.pop("include_init_phase")
            warnings.warn(
                "@include_init_phase was deprecated. Please use Scenario.disable('0th').",
                DeprecationWarning, stacklevel=2)
        except KeyError:
            include_init_phase = True
        try:
            force = kwargs.pop("set_phases")
        except KeyError:
            pass
        # S-R trend analysis
        tracker = self._tracker(name)
        if not self._interactive and filename is None:
            show_figure = False
        filename = None if self._interactive else filename
        self[name] = tracker.trend(
            force=force, show_figure=show_figure, filename=filename, **kwargs)
        # Disable 0th phase, if necessary
        if not include_init_phase:
            self[name] = tracker.disable(phases=["0th"])
        return self

    def estimate(self, model, phases=None, name="Main", n_jobs=-1, **kwargs):
        """
        Perform parameter estimation for each phases.

        Args:
            model (covsirphy.ModelBase): ODE model
            phases (list[str]): list of phase names, like 1st, 2nd...
            name (str): phase series name
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()

        Note:
            - If 'Main' was used as @name, main PhaseSeries will be used.
            - If @name phase was not registered, new PhaseSeries will be created.
            - If @phases is None, all past phase will be used.
            - Phases with estimated parameter values will be ignored.
            - In kwargs, tau value cannot be included.
        """
        if self.TAU in kwargs:
            raise ValueError(
                "@tau must be specified when scenario = Scenario(), and cannot be specified here.")
        self.tau, self[name] = self._tracker(name).estimate(
            model=model, phases=phases, n_jobs=n_jobs, **kwargs)

    def phase_estimator(self, phase, name="Main"):
        """
        Return the estimator of the phase.

        Args:
            phase (str): phase name, like 1st, 2nd...
            name (str): phase series name

        Return:
            covsirphy.Estimator: estimator of the phase
        """
        estimator = self._tracker_dict[name].series.unit(phase).estimator
        if estimator is None:
            raise UnExecutedError(f'Scenario.estimate(model, phases=["{phase}"], name={name})')
        return estimator

    def estimate_history(self, phase, name="Main", **kwargs):
        """
        Show the history of optimization.

        Args:
            phase (str): phase name, like 1st, 2nd...
            name (str): phase series name
            kwargs: keyword arguments of covsirphy.Estimator.history()

        Note:
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

        Note:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        estimator = self.phase_estimator(phase=phase, name=name)
        estimator.accuracy(**kwargs)

    def simulate(self, variables=None, phases=None, name="Main", y0_dict=None, **kwargs):
        """
        Simulate ODE models with set/estimated parameter values and show it as a figure.

        Args:
            variables (list[str] or None): variables to include, Infected/Fatal/Recovered when None
            phases (list[str] or None): phases to shoe or None (all phases)
            name (str): phase series name. If 'Main', main PhaseSeries will be used
            y0_dict(dict[str, float] or None): dictionary of initial values of variables
            kwargs: the other keyword arguments of Scenario.line_plot()

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Variables of the model and dataset (int): Confirmed etc.
        """
        tracker = copy.deepcopy(self._tracker(name))
        # Select phases
        if phases is not None:
            tracker.disable(phases=None)
            tracker.enable(phases=phases)
        # Simulation
        try:
            sim_df = tracker.simulate(y0_dict=y0_dict)
        except UnExecutedError:
            raise UnExecutedError(
                "Scenario.trend() or Scenario.add(), and Scenario.estimate(model)") from None
        # Show figure
        df = sim_df.set_index(self.DATE)
        fig_cols = self._ensure_list(
            variables or [self.CI, self.F, self.R], candidates=df.columns.tolist(), name="variables")
        title = f"{self.area}: Simulated number of cases ({name} scenario)"
        self.line_plot(df=df[fig_cols], title=title, y_integer=True, v=tracker.change_dates(), **kwargs)
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
            str or int or float

        Note:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        df = self.summary(name=name)
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
            pandas.DataFrame: selected summary dataframe

        Raises:
            KeyError: targets are not in the columns of summary dataframe
        """
        series = self._tracker_dict[name].series
        model_set = {unit.model for unit in series}
        model_set = model_set - set([None])
        parameters = self.flatten([m.PARAMETERS for m in model_set])
        day_params = self.flatten([m.DAY_PARAMETERS for m in model_set])
        selectable_cols = [self.N, *parameters, self.RT, *day_params]
        selectable_set = set(selectable_cols)
        df = series.summary().replace(self.UNKNOWN, None)
        if not selectable_set.issubset(df.columns):
            raise UnExecutedError(
                f'Scenario.estimate(model, phases=None, name="{name}")')
        targets = [targets] if isinstance(targets, str) else targets
        targets = targets or selectable_cols
        if not set(targets).issubset(selectable_set):
            raise KeyError(
                f"@targets must be selected from {', '.join(selectable_cols)}."
            )
        df = df.loc[:, targets].dropna(how="any", axis=0)
        return df.astype(np.float64)

    @deprecate(
        old="Scenario.param_history(targets: list)",
        new="Scenario.history(target: str)",
        version="2.7.3-alpha")
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

        Note:
            If 'Main' was used as @name, main PhaseSeries will be used.
        """
        self._tracker(name)
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

    def _describe(self, y0_dict=None):
        """
        Describe representative values.

        Args:
            y0_dict (dict or None): dictionary of initial values or None
                - key (str): variable name
                - value (float): initial value

        Returns:
            pandas.DataFrame
                Index
                    (int): scenario name
                Columns
                    - max(Infected): max value of Infected
                    - argmax(Infected): the date when Infected shows max value
                    - Confirmed({date}): Confirmed on the next date of the last phase
                    - Infected({date}): Infected on the next date of the last phase
                    - Fatal({date}): Fatal on the next date of the last phase
        """
        _dict = {}
        for (name, _) in self._tracker_dict.items():
            # Predict the number of cases
            df = self.simulate(name=name, y0_dict=y0_dict, show_figure=False)
            df = df.set_index(self.DATE)
            cols = df.columns[:]
            last_date = df.index[-1]
            # Max value of Infected
            max_ci = df[self.CI].max()
            argmax_ci = df[self.CI].idxmax().strftime(self.DATE_FORMAT)
            # Confirmed on the next date of the last phase
            last_c = df.loc[last_date, self.C]
            # Infected on the next date of the last phase
            last_ci = df.loc[last_date, self.CI]
            # Fatal on the next date of the last phase
            last_f = df.loc[last_date, self.F] if self.F in cols else None
            # Save representative values
            last_date_str = last_date.strftime(self.DATE_FORMAT)
            _dict[name] = {
                f"max({self.CI})": max_ci,
                f"argmax({self.CI})": argmax_ci,
                f"{self.C} on {last_date_str}": last_c,
                f"{self.CI} on {last_date_str}": last_ci,
                f"{self.F} on {last_date_str}": last_f,
            }
        return pd.DataFrame.from_dict(_dict, orient="index")

    def describe(self, y0_dict=None, with_rt=True):
        """
        Describe representative values.

        Args:
            y0_dict (dict or None): dictionary of initial values or None
                - key (str): variable name
                - value (float): initial value
            with_rt (bool): whether show the history of Rt values

        Returns:
            pandas.DataFrame:
                Index
                    str: scenario name
                Columns
                    - max(Infected): max value of Infected
                    - argmax(Infected): the date when Infected shows max value
                    - Confirmed({date}): Confirmed on the next date of the last phase
                    - Infected({date}): Infected on the next date of the last phase
                    - Fatal({date}): Fatal on the next date of the last phase
                    - nth_Rt etc.: Rt value if the values are not the same values
        """
        df = self._describe(y0_dict=y0_dict)
        if not with_rt or len(self._tracker_dict) == 1:
            return df
        # History of reproduction number
        rt_df = self.summary().reset_index()
        rt_df = rt_df.pivot_table(index=self.SERIES, columns=self.PHASE, values=self.RT)
        rt_df = rt_df.fillna(self.UNKNOWN)
        rt_df = rt_df.loc[:, rt_df.nunique() > 1]
        cols = sorted(rt_df, key=self.str2num)
        return df.join(rt_df[cols].add_suffix(f"_{self.RT}"), how="left")

    def _track_param(self, name):
        """
        Get the history of parameters for the scenario.

        Args:
            name (str): phase series name

        Returns:
            pandas.DataFrame:
                Index Date (pandas.TimeStamp)
                Columns
                    - Population (int)
                    - Rt (float)
                    - parameter values (float)
                    - day parameter values (float)
        """
        df = self.summary(name=name).replace(self.UNKNOWN, None)
        # Date range to dates
        df[self.START] = pd.to_datetime(df[self.START])
        df[self.END] = pd.to_datetime(df[self.END])
        df[self.DATE] = df[[self.START, self.END]].apply(
            lambda x: pd.date_range(x[0], x[1]).tolist(), axis=1)
        df = df.reset_index(drop=True).explode(self.DATE)
        # Columns
        df = df.drop(
            [self.TENSE, self.START, self.END, self.ODE, self.TAU, *self.EST_COLS],
            axis=1, errors="ignore")
        df = df.set_index(self.DATE)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[self.N] = df[self.N].astype(np.int64)
        return df

    def _track(self, phases=None, name="Main", y0_dict=None):
        """
        Show values of parameters and variables in one dataframe for the scenario.

        Args:
            phases (list[str] or None): phases to shoe or None (all phases)
            name (str): phase series name
            y0_dict (dict or None): dictionary of initial values or None
                - key (str): variable name
                - value (float): initial value

        Returns:
            pandas.DataFrame:
                Index reset index
                Columns
                    - Date (pandas.TimeStamp)
                    - variables (int)
                    - Population (int)
                    - Rt (float)
                    - parameter values (float)
                    - day parameter values (float)
        """
        sim_df = self.simulate(phases=phases, name=name, y0_dict=y0_dict, show_figure=False)
        param_df = self._track_param(name=name)
        return pd.merge(
            sim_df, param_df, how="inner", left_on=self.DATE, right_index=True, sort=True)

    def track(self, phases=None, with_actual=True, y0_dict=None):
        """
        Show values of parameters and variables in one dataframe.

        Args:
            phases (list[str] or None): phases to shoe or None (all phases)
            with_actual (bool): if True, show actual number of cases will included as "Actual" scenario
            y0_dict (dict or None): dictionary of initial values or None
                - key (str): variable name
                - value (float): initial value

        Returns:
            pandas.DataFrame: tracking records
                Index
                    reset index
                Columns
                    - Scenario (str)
                    - Date (pandas.TimeStamp)
                    - variables (int)
                    - Population (int)
                    - Rt (float)
                    - parameter values (float)
                    - day parameter values (float)
        """
        dataframes = []
        append = dataframes.append
        for name in self._tracker_dict.keys():
            df = self._track(phases=phases, name=name, y0_dict=y0_dict)
            df.insert(0, self.SERIES, name)
            append(df)
        if with_actual:
            df = self._data.records(extras=False)
            df.insert(0, self.SERIES, self.ACTUAL)
            append(df)
        return pd.concat(dataframes, axis=0, sort=False)

    def _history(self, target, phases=None, with_actual=True, y0_dict=None):
        """
        Show the history of variables and parameter values to compare scenarios.

        Args:
            target (str): parameter or variable name to show (Rt, Infected etc.)
            phases (list[str] or None): phases to shoe or None (all phases)
            with_actual (bool): if True and @target is a variable name, show actual number of cases
            y0_dict (dict or None): dictionary of initial values or None
                - key (str): variable name
                - value (float): initial value

        Returns:
            pandas.DataFrame
        """
        # Include actual data or not
        with_actual = with_actual and target in self.VALUE_COLUMNS
        # Get tracking data
        df = self.track(phases=phases, with_actual=with_actual, y0_dict=y0_dict)
        if target not in df.columns:
            col_str = ", ".join(list(df.columns))
            raise KeyError(f"@target must be selected from {col_str}, but {target} was applied.")
        # Select the records of target variable
        return df.pivot_table(
            values=target, index=self.DATE, columns=self.SERIES, aggfunc="last")

    def history(self, target, phases=None, with_actual=True, y0_dict=None, **kwargs):
        """
        Show the history of variables and parameter values to compare scenarios.

        Args:
            target (str): parameter or variable name to show (Rt, Infected etc.)
            phases (list[str] or None): phases to shoe or None (all phases)
            with_actual (bool): if True and @target is a variable name, show actual number of cases
            y0_dict (dict or None): dictionary of initial values or None
                - key (str): variable name
                - value (float): initial value
            kwargs: the other keyword arguments of Scenario.line_plot()

        Returns:
            pandas.DataFrame
        """
        df = self._history(target=target, phases=phases, with_actual=with_actual, y0_dict=y0_dict)
        df.dropna(subset=[col for col in df.columns if col != self.ACTUAL], inplace=True)
        if target == self.RT:
            ylabel = self.RT_FULL
        elif target in self.VALUE_COLUMNS:
            ylabel = f"The number of {target.lower()} cases"
        else:
            ylabel = target
        title = f"{self.area}: {ylabel} over time"
        tracker = self._tracker(self.MAIN)
        self.line_plot(
            df=df, title=title, ylabel=ylabel, v=tracker.change_dates(), math_scale=False,
            h=1.0 if target == self.RT else None, **kwargs)
        return df

    def history_rate(self, params=None, name="Main", **kwargs):
        """
        Show change rates of parameter values in one figure.
        We can find the parameters which increased/decreased significantly.

        Args:
            params (list[str] or None): parameters to show
            name (str): phase series name
            kwargs: the other keyword arguments of Scenario.line_plot()

        Returns:
            pandas.DataFrame
        """
        df = self._track_param(name=name)
        model = self._tracker(name).last_model
        cols = list(set(df.columns) & set(model.PARAMETERS))
        if params is not None:
            if not isinstance(params, (list, set)):
                raise TypeError(f"@params must be a list of parameters, but {params} were applied.")
            cols = list(set(cols) & set(params)) or cols
        df = df.loc[:, cols] / df.loc[df.index[0], cols]
        # Show figure
        f_date = df.index[0].strftime(self.DATE_FORMAT)
        title = f"{self.area}: {model.NAME} parameter change rates over time (1.0 on {f_date})"
        ylabel = f"Value per that on {f_date}"
        title = f"{self.area}: {ylabel} over time"
        tracker = self._tracker(self.MAIN)
        self.line_plot(
            df=df, title=title, ylabel=ylabel, v=tracker.change_dates(), math_scale=False, **kwargs)
        return df

    def retrospective(self, beginning_date, model, control="Main", target="Target", **kwargs):
        """
        Perform retrospective analysis.
        Compare the actual series of phases (control) and
        series of phases with specified parameters (target).

        Args:
            beginning_date (str): when the parameter values start to be changed from actual values
            model (covsirphy.ModelBase): ODE model
            control (str): scenario name of control
            target (str): scenario name of target
            kwargs: keyword argument of parameter values and Estimator.run()

        Note:
            When parameter values are not specified,
            actual values of the last date before the beginning date will be used.
        """
        param_dict = {k: v for (k, v) in kwargs.items() if k in model.PARAMETERS}
        est_kwargs = dict(kwargs.items() - param_dict.items())
        # Control
        self.clear(name=control, include_past=True)
        self.trend(name=control, show_figure=False)
        try:
            self.separate(date=beginning_date, name=control)
        except ValueError:
            pass
        self.estimate(model, name=control, **est_kwargs)
        # Target
        self.clear(name=target, include_past=False, template=control)
        phases_changed = [
            self.num2str(i) for (i, ph) in enumerate(self._tracker(target).series)
            if ph >= beginning_date]
        self.delete(phases=phases_changed, name=target)
        self.add(name=target, **param_dict)
        self.estimate(model, name=target, **est_kwargs)

    def score(self, metrics="RMSLE", variables=None, phases=None, past_days=None, name="Main", y0_dict=None):
        """
        Evaluate accuracy of phase setting and parameter estimation of all enabled phases all some past days.

        Args:
            metrics (str): "MAE", "MSE", "MSLE", "RMSE" or "RMSLE"
            variables (list[str] or None): variables to use in calculation
            phases (list[str] or None): phases to use in calculation
            past_days (int or None): how many past days to use in calculation, natural integer
            name(str): phase series name. If 'Main', main PhaseSeries will be used
            y0_dict(dict[str, float] or None): dictionary of initial values of variables

        Returns:
            float: score with the specified metrics

        Note:
            If @variables is None, ["Infected", "Fatal", "Recovered"] will be used.
            "Confirmed", "Infected", "Fatal" and "Recovered" can be used in @variables.
            If @phases is None, all phases will be used.
            @phases and @past_days can not be specified at the same time.
        """
        tracker = self._tracker(name)
        if past_days is not None:
            if phases is not None:
                raise ValueError(
                    "@phases and @past_days cannot be specified at the same time.")
            past_days = self._ensure_natural_int(past_days, name="past_days")
            # Separate a phase, if possible
            beginning_date = self.date_change(self._data.last_date, days=0 - past_days)
            try:
                tracker.separate(date=beginning_date)
            except ValueError:
                pass
            # Ge the list of target phases
            phases = [
                self.num2str(num) for (num, unit)
                in enumerate(tracker.series)
                if unit >= beginning_date
            ]
        return tracker.score(
            metrics=metrics, variables=variables, phases=phases, y0_dict=y0_dict)

    def estimate_delay(self, oxcgrt_data=None, indicator="Stringency_index",
                       target="Confirmed", value_range=(7, None)):
        """
        Estimate the average day [days] between the indicator and the target.
        We assume that the indicator impact on the target value with delay.

        Args:
            oxcgrt_data (covsirphy.OxCGRTData): OxCGRT dataset
            indicator (str): indicator name, a column of any registered datasets
            target (str): target name, a column of any registered datasets
            value_range (tuple(int, int or None)): tuple, giving the minimum and maximum range to search for change over time

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data
            UserWarning: failed in calculating and returned the default value (recovery period)

        Returns:
            tuple(int, pandas.DataFrame):
                - int: the estimated number of days of delay [day]
                - pandas.DataFrame:
                    Index
                        reset index
                    Columns
                        - (int or float): column defined by @indicator
                        - (int or float): column defined by @target
                        - (int): column defined by @delay_name [days]

        Note:
            - Average recovered period of JHU dataset will be used as returned value when the estimated value was not in value_range.
            - Very long periods (outside of 99% quantile) are taken out.
            - @oxcgrt_data argument was deprecated. Please use Scenario.register(extras=[oxcgrt_data]).
        """
        # Register OxCGRT data
        if oxcgrt_data is not None:
            warnings.warn(
                "Please use Scenario.register(extras=[oxcgrt_data]) rather than Scenario.fit(oxcgrt_data).",
                DeprecationWarning, stacklevel=1)
            self.register(extras=[oxcgrt_data])
        # Calculate delay values
        df = self._data.estimate_delay(indicator=indicator, target=target, delay_name="Period Length")
        # Filter out very long periods
        df_filtered = df.loc[df["Period Length"] < df["Period Length"].quantile(0.99)]
        if value_range[1] is not None:
            df_filtered = df_filtered.loc[df["Period Length"] < value_range[1]]
        delay_days = self._data.recovery_period() if df_filtered.empty else int(df_filtered["Period Length"].mean())
        return (delay_days, df)

    def _fit_create_data(self, model, name, delay, removed_cols):
        """
        Create train/test dataset for Elastic Net regression,
        assuming that extra variables will impact on ODE parameter values with delay.

        Args:
            model (covsirphy.ModelBase): ODE model
            name (str): scenario name
            delay (int): delay period
            removed_cols (list): list of variables to remove from X dataset

        Returns:
            tuple(pandas.DataFrame):
                - X dataset for linear regression
                - y dataset for linear regression
                - X dataset of the target dates
        """
        # Clear the future phases
        self.clear(name=name, include_past=False)
        # Parameter values
        param_df = self._track_param(name=name)[model.PARAMETERS]
        # Extra datasets (explanatory variables)
        extras_df = self._data.records(main=False, extras=True).set_index(self.DATE)
        extras_df = extras_df.loc[:, ~extras_df.columns.isin(removed_cols)]
        # Apply delay on OxCGRT data
        extras_df.index += timedelta(days=delay)
        # Create training/test dataset
        df = param_df.join(extras_df, how="inner")
        df = df.rolling(window=delay).mean().dropna().drop_duplicates()
        X = df.drop(model.PARAMETERS, axis=1)
        y = df.loc[:, model.PARAMETERS]
        # X dataset of the target dates
        dates = pd.date_range(
            start=param_df.index.max() + timedelta(days=1),
            end=extras_df.index.max(),
            freq="D")
        X_target = extras_df.loc[dates]
        return (X, y, X_target)

    def fit(self, oxcgrt_data=None, name="Main", test_size=0.2, seed=0, delay=None):
        """
        Learn the relationship of ODE parameter values and delayed OxCGRT scores using Elastic Net regression,
        assuming that OxCGRT scores will impact on ODE parameter values with delay.
        Min-max scaling and Elastic net regression with parameter optimization and cross validation.

        Args:
            oxcgrt_data (covsirphy.OxCGRTData): OxCGRT dataset
            name (str): scenario name
            test_size (float): proportion of the test dataset of Elastic Net regression
            seed (int): random seed when spliting the dataset to train/test data
            delay (int): number of days of delay between policy measure and effect
            on number of confirmed cases.

        Raises:
            covsirphy.UnExecutedError: Scenario.estimate() or Scenario.add() were not performed

        Returns:
            dict(str, object):
                - scaler (object): scaler class
                - regressor (object): regressor class
                - alpha (float): alpha value used in Elastic Net regression
                - l1_ratio (float): l1_ratio value used in Elastic Net regression
                - score_train (float): determination coefficient of train dataset
                - score_test (float): determination coefficient of test dataset
                - X_train (numpy.array): X_train
                - y_train (numpy.array): y_train
                - X_test (numpy.array): X_test
                - y_test (numpy.array): y_test
                - X_target (numpy.array): X_target
                - intercept (pandas.DataFrame): intercept values (Index ODE parameters, Columns indicators)
                - delay (int): number of days of delay between policy measure and effect
                  on number of confirmed cases.

        Note:
            @oxcgrt_data argument was deprecated. Please use Scenario.register(extras=[oxcgrt_data]).
        """
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        # Register OxCGRT data
        if oxcgrt_data is not None:
            warnings.warn(
                "Please use Scenario.register(extras=[oxcgrt_data]) rather than Scenario.fit(oxcgrt_data).",
                DeprecationWarning, stacklevel=1)
            self.register(extras=[oxcgrt_data])
        # ODE model
        model = self._tracker(name).last_model
        if model is None:
            raise UnExecutedError(
                "Scenario.estimate() or Scenario.add()",
                message=f", specifying @model (covsirphy.SIRF etc.) and @name='{name}'.")
        # Set delay effect
        if delay is None:
            delay, delay_df = self.estimate_delay(oxcgrt_data)
            removed_cols = delay_df.columns.tolist()
        else:
            delay = self._ensure_natural_int(delay, name="delay")
            removed_cols = []
        # Create training/test dataset
        try:
            X, y, X_target = self._fit_create_data(
                model=model, name=name, delay=delay, removed_cols=removed_cols)
        except NotRegisteredExtraError:
            raise NotRegisteredExtraError(
                "Scenario.register(jhu_data, population_data, extras=[...])",
                message="with extra datasets") from None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        # Create pipeline for learning
        cv = linear_model.MultiTaskElasticNetCV(
            alphas=[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            l1_ratio=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            cv=5, n_jobs=-1)
        steps = [
            ("scaler", MinMaxScaler()),
            ("regressor", cv),
        ]
        pipeline = Pipeline(steps=steps)
        pipeline.fit(X_train, y_train)
        # Register the pipeline and X-target for prediction
        self._lm_dict[name] = (pipeline, X_target)
        # Get train score
        score_train = r2_score(pipeline.predict(X_train), y_train)
        # Get test score
        score_test = r2_score(pipeline.predict(X_test), y_test)
        # Return information regarding regression model
        reg_output = pipeline.named_steps.regressor
        # Intercept
        intercept_df = pd.DataFrame(reg_output.coef_, index=y_train.columns, columns=X_train.columns)
        # Return information
        return {
            **{k: type(v) for (k, v) in steps},
            "alpha": reg_output.alpha_,
            "l1_ratio": reg_output.l1_ratio_,
            "score_train": score_train,
            "score_test": score_test,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_target": X_target,
            "intercept": intercept_df,
            "delay": delay
        }

    def predict(self, name="Main"):
        """
        Predict parameter values of the future phases using Elastic Net regression with OxCGRT scores,
        assuming that OxCGRT scores will impact on ODE parameter values with delay.
        New future phases will be added (over-written).

        Args:
            name (str): scenario name

        Raises:
            covsirphy.UnExecutedError: Scenario.fit() was not performed
            NotRegisteredExtraError: no extra datasets were registered

        Returns:
            covsirphy.Scenario: self
        """
        # Arguments
        if name not in self._lm_dict:
            raise UnExecutedError(f"Scenario.fit(name={name})")
        model = self._tracker(name).last_model
        # Prediction with regression model
        pipeline, X_target = self._lm_dict[name]
        predicted = pipeline.predict(X_target)
        # -> end_date/parameter values
        df = pd.DataFrame(predicted, index=X_target.index, columns=model.PARAMETERS)
        df = df.applymap(lambda x: np.around(x, 4 - int(floor(log10(abs(x)))) - 1))
        df.index = [date.strftime(self.DATE_FORMAT) for date in df.index]
        df.index.name = "end_date"
        phase_df = df.drop_duplicates(keep="last").reset_index()
        # Select the last values
        phase_df = phase_df.iloc[[-1], :]
        # Set new future phases
        for phase_dict in phase_df.to_dict(orient="records"):
            self.add(name=name, **phase_dict)
        return self

    def fit_predict(self, oxcgrt_data=None, name="Main", **kwargs):
        """
        Predict parameter values of the future phases using Elastic Net regression with OxCGRT scores,
        assuming that OxCGRT scores will impact on ODE parameter values with delay.
        New future phases will be added (over-written).

        Args:
            oxcgrt_data (covsirphy.OxCGRTData or None): OxCGRT dataset
            name (str): scenario name
            kwargs: the other arguments of Scenario.fit()

        Raises:
            covsirphy.UnExecutedError: Scenario.estimate() or Scenario.add() were not performed
            NotRegisteredExtraError: no extra datasets were registered

        Returns:
            covsirphy.Scenario: self

        Note:
            @oxcgrt_data argument was deprecated. Please use Scenario.register(extras=[oxcgrt_data]).
        """
        self.fit(oxcgrt_data=oxcgrt_data, name=name, **kwargs)
        self.predict(name=name)
        return self
