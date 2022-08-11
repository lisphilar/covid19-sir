#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import pandas as pd
from covsirphy.util.error import UnExecutedError, deprecate
from covsirphy.util.argument import find_args
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated.trend_detector import TrendDetector
from covsirphy._deprecated.trend_plot import trend_plot
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.phase_unit import PhaseUnit
from covsirphy._deprecated.phase_estimator import MPEstimator
from covsirphy._deprecated.phase_series import PhaseSeries


class ParamTracker(Term):
    """
    Deprecated. Split records with S-R trend analysis and estimate parameter values of the phases.

    Args:
        record_df (pandas.DataFrame): records

            Index
                reset index
            Columns
                - Date (pandas.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - Susceptible (int): the number of susceptible cases
        phase_series (covsirphy.PhaseSeries): phase series object with first/last dates and population
        area (str or None): area name, like Japan/Tokyo, or empty string
        tau (int or None): tau value [min]
    """

    @deprecate("", new="ODEHandler", version="2.19.1-zeta-fu1")
    def __init__(self, record_df, phase_series, area=None, tau=None):
        # Phase series
        self._series = self._ensure_instance(phase_series, PhaseSeries, name="phase_series")
        # Records
        self._ensure_dataframe(record_df, name="record_df", columns=self.SUB_COLUMNS)
        df = record_df.loc[record_df[self.DATE] >= Validator(phase_series.first_date).date()]
        self.record_df = df.loc[df[self.DATE] <= Validator(phase_series.last_date).date()]
        # Area name
        self.area = area or ""
        # Tau value
        self.tau = Validator(tau, "tau").tau(default=None)

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

    def __len__(self):
        return len(self._series)

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

    @staticmethod
    def _ensure_list(target, candidates=None, name="target"):
        """
        Ensure the target is a sub-list of the candidates.

        Args:
            target (list[object]): target to ensure
            candidates (list[object] or None): list of candidates, if we have
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        if not isinstance(target, (list, tuple)):
            raise TypeError(f"@{name} must be a list or tuple, but {type(target)} was applied.")
        if candidates is None:
            return target
        # Check the target is a sub-list of candidates
        try:
            strings = [str(candidate) for candidate in candidates]
        except TypeError:
            raise TypeError(f"@candidates must be a list, but {candidates} was applied.") from None
        ok_list = [element in candidates for element in target]
        if all(ok_list):
            return target
        candidate_str = ", ".join(strings)
        raise KeyError(f"@{name} must be a sub-list of [{candidate_str}], but {target} was applied.") from None

    @staticmethod
    def create_series(first_date, last_date, population):
        """
        Create PhaseSeries instance.

        Args:
            first_date (str): the first date of the records
            last_date (str): the last date of the records
            population (int): population value

        Returns:
            covsirphy.PhaseSeries
        """
        return PhaseSeries(
            first_date=first_date, last_date=last_date, population=population)

    @property
    def series(self):
        """
        covsirphy.PhaseSeries: phase series object (series of covsirphy.PhaseUnit)
        """
        return self._series

    @property
    def last_model(self):
        """
        covsirphy.ModelBase: ODE model if the last phase
        """
        return self._series.unit(phase="last").model

    def trend(self, force=True, show_figure=False, **kwargs):
        """
        Split the records with trend analysis.

        Args:
            force (bool): if True, change points will be over-written
            show_figure (bool): if True, show the result as a figure
            kwargs: keyword arguments of covsirphy.TrendDetector(), .TrendDetector.sr() and .trend_plot()

        Returns:
            covsirphy.PhaseSeries
        """
        detector = TrendDetector(
            data=self.record_df, area=self.area, **find_args(TrendDetector, **kwargs))
        # Perform S-R trend analysis
        detector.sr(**find_args(TrendDetector.sr, **kwargs))
        # Register phases
        if force or not self._series:
            self._series.clear(include_past=True)
            _, end_dates = detector.dates()
            [self._series.add(end_date=end_date) for end_date in end_dates]
        # Show S-R plane
        if show_figure:
            detector.show(**find_args(trend_plot, **kwargs))
        return self._series

    def _ensure_phase_setting(self):
        """
        Ensure that phases were set.
        """
        if not self._series:
            raise UnExecutedError(".trend() or .add()")

    def find_phase(self, date):
        """
        Find the name of the phase which has the date.

        Args:
            date (str): date, like 01Jan2020

        Returns:
            tuple(str, covsirphy.PhaseUnit):
                str: phase name, like 1st, 2nd,...
                covsirphy.PhaseUnit: phase unit
        """
        self._ensure_phase_setting()
        Validator(date).date()
        phase_nest = [
            (self.num2str(i), unit) for (i, unit) in enumerate(self._series) if date in unit]
        try:
            return phase_nest[0]
        except IndexError:
            raise IndexError(f"Phase on {date} is not registered.") from None

    def change_dates(self):
        """
        Return the list of changed dates (start dates of phases since 1st phase).

        Returns:
            list[str]: list of change dates
        """
        return [unit.start_date for unit in self._series][1:]

    def near_change_dates(self):
        """
        Show the list of dates which are yesterday/tomorrow of the start/end dates.

        Returns:
            list[str]: list of dates
        """
        base_dates = [
            date for ph in self._series for date in [ph.start_date, ph.end_date]]
        return [
            date for base_date in base_dates
            for date in [self.yesterday(base_date), base_date, Validator(base_date).date() + timedelta(days=1)]]

    @classmethod
    def yesterday(cls, date_str):
        """
        Yesterday of the date.

        Args:
            date_str (str): today

        Returns:
            str: yesterday
        """
        return cls.date_change(date_str, days=-1)

    def all_phases(self):
        """
        Return the names of all enabled phases.

        Returns:
            list[str]: the names of all enabled phases
        """
        return [self.num2str(num) for (num, unit) in enumerate(self._series) if unit]

    def disable(self, phases):
        """
        The phases will be disabled.

        Args:
            phase (list[str] or None): phase names or None (all enabled phases)

        Returns:
            covsirphy.PhaseSeries
        """
        phases = self._ensure_list(
            phases or self.all_phases(), candidates=None, name="phases")
        for phase in phases:
            self._series.disable(phase)
        return self._series

    def enable(self, phases):
        """
        The phases will be enabled.

        Args:
            phase (list[str] or None): phase names or None (all disabled phases)

        Returns:
            covsirphy.PhaseSeries
        """
        all_dis_phases = [
            self.num2str(num) for (num, unit) in enumerate(self._series) if not unit]
        phases = self._ensure_list(
            phases or all_dis_phases, candidates=None, name="phases")
        for phase in phases:
            self._series.enable(phase)
        return self._series

    def add(self, end_date=None, days=None, population=None, model=None, **kwargs):
        """
        Add a new phase.
        The start date will be the next date of the last registered phase.

        Args:
            end_date (str): end date of the new phase
            days (int): the number of days to add
            population (int or None): population value of the start date
            model (covsirphy.ModelBase or None): ODE model
            kwargs: keyword arguments of ODE model parameters, not including tau value

        Returns:
            covsirphy.PhaseSeries

        Note:
            - If the phases series has not been registered, new phase series will be created.
            - Either @end_date or @days must be specified.
            - If @end_date and @days are None, the end date will be the last date of the records.
            - If both of @end_date and @days were specified, @end_date will be used.
            - If @population is None, initial value will be used.
            - If @model is None, the model of the last phase will be used.
            - Tau will be fixed as the last phase's value.
            - kwargs: Default values are the parameter values of the last phase.
        """
        if end_date is not None:
            Validator(end_date, "end_date").date()
        try:
            self._series.add(
                end_date=end_date, days=days, population=population,
                model=model, tau=self.tau, **kwargs)
        except ValueError:
            last_date = self._series.unit("last").end_date
            raise ValueError(
                f'@end_date must be over {last_date}. However, {end_date} was applied.') from None
        return self._series

    def delete_all(self):
        """
        Delete all phases.

        Returns:
            covsirphy.PhaseSeries
        """
        arg_dict = {
            "first_date": self._series.first_date,
            "last_date": self._series.last_date,
            "population": self._series.unit("last").population
        }
        self._series = PhaseSeries(**arg_dict)
        return self._series

    def delete(self, phases):
        """
        Delete selected phases.

        Args:
            phases (list[str]): phases names to delete

        Returns:
            covsirphy.PhaseSeries
        """
        all_phases = self.all_phases()
        if "last" in set(phases):
            phases = [ph for ph in phases if ph != "last"] + [all_phases[-1]]
        self._ensure_list(phases, candidates=all_phases, name="phases")
        phases = sorted(list(set(phases)), key=self.str2num, reverse=True)
        for ph in phases:
            self._series.delete(ph)
        return self._series

    def combine(self, phases, population=None, **kwargs):
        """
        Combine the sequential phases as one phase.
        New phase name will be automatically determined.

        Args:
            phases (list[str]): list of phases
            population (int): population value of the start date
            kwargs: keyword arguments to save as phase information

        Raises:
            TypeError: @phases is not a list

        Returns:
            covsirphy.Scenario: self
        """
        all_phases = self.all_phases()
        if "last" in set(phases):
            phases.remove("last")
            phases = sorted(phases, key=self.str2num, reverse=False)
            last_phase = "last"
        else:
            phases = sorted(phases, key=self.str2num, reverse=False)
            last_phase = phases[-1]
        self._ensure_list(phases, candidates=all_phases, name="phases")
        # Setting of the new phase
        start_date = self._series.unit(phases[0]).start_date
        end_date = self._series.unit(last_phase).end_date
        population = population or self._series.unit(last_phase).population
        new_unit = PhaseUnit(start_date, end_date, population)
        new_unit.set_ode(**kwargs)
        # Phases to keep
        kept_units = [
            unit for unit in self.series if unit < start_date or unit > end_date]
        # Replace units
        self._series.replaces(
            phase=None, new_list=kept_units + [new_unit], keep_old=False)
        return self._series

    def separate(self, date, population=None, **kwargs):
        """
        Create a new phase with the change point.
        New phase name will be automatically determined.

        Args:
            date (str): change point, i.e. start date of the new phase
            population (int): population value of the change point
            kwargs: keyword arguments of PhaseUnit.set_ode() if update is necessary

        Returns:
            covsirphy.PhaseSeries
        """
        phase, old = self.find_phase(date)
        if date in self.near_change_dates():
            raise ValueError(
                f"Cannot be separated on {date} because this date is too close to registered change dates.")
        new_pre = PhaseUnit(
            old.start_date, self.yesterday(date), old.population)
        setting_dict = old.to_dict()
        setting_dict.update(kwargs)
        new_pre.set_ode(**setting_dict)
        new_fol = PhaseUnit(date, old.end_date, population or old.population)
        new_fol.set_ode(model=old.model, **setting_dict)
        self._series.replaces(phase, [new_pre, new_fol])
        return self._series

    def past_phases(self, phases=None):
        """
        Return names and phase units of the past phases.

        Args:
            phases (tuple/list[str]): list of phase names, like 1st, 2nd...

        Returns:
            tuple(list[str], list[covsirphy.PhaseUnit]):
                list[str]: list of phase names
                list[covsirphy.PhaseUnit]: list of phase units

        Note:
            If @phases is None, return the all past phases.
            If @phases is not None, intersection will be selected.
        """
        self._ensure_phase_setting()
        # List of past phases
        last_date = self.record_df[self.DATE].max().strftime(self.DATE_FORMAT)
        past_nest = [
            [self.num2str(num), unit]
            for (num, unit) in enumerate(self._series)
            if unit and unit <= last_date
        ]
        past_phases, _ = zip(*past_nest)
        # Select phases to use
        selected_phases = self._ensure_list(
            phases or past_phases, candidates=past_phases, name="phases")
        final_phases = [
            ph for ph in past_phases if ph in set(selected_phases)]
        # Convert phase names to phase units
        return (final_phases, [self._series.unit(ph) for ph in final_phases])

    def future_phases(self):
        """
        Return names and phase units of the future phases.

        Returns:
            tuple(list[str], list[covsirphy.PhaseUnit]):
                list[str]: list of phase names
                list[covsirphy.PhaseUnit]: list of phase units
        """
        self._ensure_phase_setting()
        # All phases
        all_nest = [
            [self.num2str(num), unit] for (num, unit) in enumerate(self._series) if unit]
        # Past phases
        past_phases, _ = self.past_phases(phases=None)
        # Future phases
        future_nest = [
            [ph, unit] for (ph, unit) in all_nest if ph not in past_phases]
        if not future_nest:
            return ([], [])
        return tuple(zip(*future_nest))

    def estimate(self, model, phases=None, n_jobs=-1, **kwargs):
        """
        Perform parameter estimation for each phases.

        Args:
            model (covsirphy.ModelBase): ODE model
            phases (list[str]): list of phase names, like 1st, 2nd...
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()

        Returns:
            tuple(int, covsirphy.PhaseSeries): tau value [min] and phase series

        Note:
            - If @phases is None, all past phase will be used.
            - Phases with estimated parameter values will be ignored.
            - In kwargs, tau value cannot be included.
        """
        self._ensure_phase_setting()
        model = Validator(model, "model").subclass(ModelBase)
        units = [
            unit.set_id(phase=phase)
            for (phase, unit) in zip(*self.past_phases(phases=phases))
            if unit.id_dict is None
        ]
        if not units:
            raise IndexError("All phases have completed parameter estimation.")
        # Parameter estimation
        mp_estimator = MPEstimator(
            record_df=self.record_df, model=model, tau=self.tau, **kwargs
        )
        mp_estimator.add(units)
        results = mp_estimator.run(n_jobs=n_jobs, **kwargs)
        self.tau = mp_estimator.tau
        # Register the results
        self._series.replaces(phase=None, new_list=results, keep_old=True)
        return (self.tau, self._series)

    def simulate(self, y0_dict=None):
        """
        Simulate ODE models with set/estimated parameter values.

        Args:
            y0_dict(dict[str, float] or None): dictionary of initial values of variables

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Variables of the model and dataset (int): Confirmed etc.
        """
        self._ensure_phase_setting()
        try:
            return self._series.simulate(record_df=self.record_df, y0_dict=y0_dict)
        except NameError:
            raise UnExecutedError(".estimate()") from None

    def _compare_with_actual(self, variables, y0_dict=None):
        """
        Compare actual/simulated number of cases.

        Args:
            variables (list[str]): variables to use in calculation
            y0_dict(dict[str, float] or None): dictionary of initial values of variables

        Returns:
            tuple(pandas.DataFrame, pandas.DataFrame):
                - actual (pandas.DataFrame):
                    Index Date (pd.Timestamp)
                    Columns variables defined by @variables
                - simulated (pandas.DataFrame):
                    Index Date (pd.Timestamp)
                    Columns variables defined by @variables
        """
        record_df = self.record_df.copy().set_index(self.DATE)
        simulated_df = self.simulate(y0_dict=y0_dict).set_index(self.DATE)
        df = record_df.join(simulated_df, how="inner", rsuffix="_sim").dropna()
        rec_df = df.loc[:, variables]
        sim_df = df.loc[:, [f"{col}_sim" for col in variables]]
        sim_df.columns = variables
        return (rec_df, sim_df)

    def score(self, variables=None, phases=None, y0_dict=None, **kwargs):
        """
        Evaluate accuracy of phase setting and parameter estimation of selected enabled phases.

        Args:
            variables (list[str] or None): variables to use in calculation
            phases (list[str] or None): phases to use in calculation
            y0_dict(dict[str, float] or None): dictionary of initial values of variables
            kwargs: keyword arguments of covsirphy.Evaluator.score()

        Returns:
            float: score with the specified metrics

        Note:
            If @variables is None, ["Infected", "Fatal", "Recovered"] will be used.
            "Confirmed", "Infected", "Fatal" and "Recovered" can be used in @variables.
            If @phases is None, all phases will be used.
        """
        # Arguments
        variables = variables or [self.CI, self.F, self.R]
        variables = self._ensure_list(
            variables, self.VALUE_COLUMNS, name="variables")
        # Disable the non-target phases
        all_phases, _ = self.past_phases(phases=None)
        target_phases, _ = self.past_phases(phases=phases)
        ignored_phases = list(set(all_phases) - set(target_phases))
        if ignored_phases:
            self.disable(ignored_phases)
        # Get the number of cases
        rec_df, sim_df = self._compare_with_actual(
            variables=variables, y0_dict=y0_dict)
        # Calculate score
        evaluator = Evaluator(rec_df, sim_df)
        score = evaluator.score(**find_args(Evaluator.score, **kwargs))
        # Enable the disabled non-target phases
        if ignored_phases:
            self.enable(ignored_phases)
        return score

    def last_end_date(self):
        """
        Return the last end date of the series.

        Returns:
            str: the last end date
        """
        return self._series.unit(phase="last").end_date
