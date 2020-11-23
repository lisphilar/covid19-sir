#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.term import Term
from covsirphy.phase.phase_unit import PhaseUnit
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
    """

    def __init__(self, record_df, phase_series, area=None):
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.SUB_COLUMNS)
        self.series = self.ensure_instance(
            phase_series, PhaseSeries, name="phase_seres")
        self.area = area or ""

    def trend(self, show_figure=True, filename=None, **kwargs):
        """
        Split the records with trend analysis.

        Args:
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of ChangeFinder()

        Returns:
            covsirphy.PhaseSeries
        """
        sr_df = self.record_df.set_index(self.DATE).loc[:, [self.R, self.S]]
        self.series.trend(sr_df=sr_df, **kwargs)
        if show_figure:
            self.series.trend_show(
                sr_df=sr_df, area=self.area, filename=filename)
        return self.series

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
        if not self.series:
            raise ValueError(
                "No phases were registered. Please use .trend() or .add() in advance.")
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
