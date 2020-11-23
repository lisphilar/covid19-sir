#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.phase_unit import PhaseUnit
from covsirphy.phase.phase_estimator import MPEstimator
from covsirphy.phase.phase_series import PhaseSeries


class ParamTracker(Term):
    """
    Split records with S-R trend analysis and estimate parameter values of the phases.

    Args:
        record_df (pandas.DataFrame): records
            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - Susceptible (int): the number of susceptible cases
        phase_series (covsirphy.PhaseSeries): phase series object with first/last dates and population
        area (str or None): area name, like Japan/Tokyo, or empty string
        tau (int or None): tau value [min]
    """

    def __init__(self, record_df, phase_series, area=None, tau=None):
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.SUB_COLUMNS)
        self.series = self.ensure_instance(
            phase_series, PhaseSeries, name="phase_seres")
        self.area = area or ""
        self.tau = self.ensure_tau(tau)

    def trend(self, force=True, show_figure=True, filename=None, **kwargs):
        """
        Split the records with trend analysis.

        Args:
            force (bool): if True, change points will be over-written
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of ChangeFinder()

        Returns:
            covsirphy.PhaseSeries
        """
        sr_df = self.record_df.set_index(self.DATE).loc[:, [self.R, self.S]]
        if force or not self.series:
            self.series.trend(sr_df=sr_df, **kwargs)
        if show_figure:
            self.series.trend_show(
                sr_df=sr_df, area=self.area, filename=filename)
        return self.series

    def _ensure_phase_setting(self):
        """
        Ensure that phases were set.
        """
        if not self.series:
            raise ValueError(
                "No phases were registered. Please use .trend() or .add() in advance.")

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
        self.ensure_date(date)
        phase_nest = [
            (self.num2str(i), unit) for (i, unit) in enumerate(self.series) if date in unit]
        try:
            return phase_nest[0]
        except IndexError:
            raise IndexError(f"Phase on {date} is not registered.") from None

    def near_change_dates(self):
        """
        Show the list of dates which are yesterday/tomorrow of the start/end dates.

        Returns:
            list[str]: list of dates
        """
        base_dates = [
            date for ph in self.series for date in [ph.start_date, ph.end_date]]
        return [
            date for base_date in base_dates
            for date in [self.yesterday(base_date), base_date, self.tomorrow(base_date)]]

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
        self.series.replaces(phase, [new_pre, new_fol])
        return self.series

    def past_phases(self, phases=None):
        """
        Return names and phase units of the past phases.

        Args:
            phases (tuple/list[str]): list of phase names, like 1st, 2nd...

        Returns:
            tuple(list[str], list[covsirphy.PhaseUnit]):
                list[str]: list of phase names
                list[covsirphy.PhaseUnit]: list of phase units

        Notes:
            If @phases is None, return the all past phases.
            If @phases is not None, intersection will be selected.
        """
        self._ensure_phase_setting()
        # List of past phases
        last_date = self.record_df[self.DATE].max().strftime(self.DATE_FORMAT)
        past_nest = [
            [self.num2str(num), unit]
            for (num, unit) in enumerate(self.series)
            if unit and unit <= last_date
        ]
        past_phases, past_units = zip(*past_nest)
        # Select phases to use
        selected_phases = phases or past_phases[:]
        if not isinstance(selected_phases, (list, tuple)):
            raise TypeError(
                "@phases must be None or a list/tuple of phase names.")
        final_phases = list(set(selected_phases) & set(past_phases))
        # Convert phase names to phase units
        selected_units = [self.series.unit(ph) for ph in selected_phases]
        final_units = list(set(selected_units) & set(past_units))
        return (final_phases, final_units)

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

        Notes:
            - If @phases is None, all past phase will be used.
            - Phases with estimated parameter values will be ignored.
            - In kwargs, tau value cannot be included.
        """
        model = self.ensure_subclass(model, ModelBase, "model")
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
        self.series.replaces(phase=None, new_list=results, keep_old=True)
        return (self.tau, self.series)
