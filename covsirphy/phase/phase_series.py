#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.cleaning.term import Term
from covsirphy.phase.phase_unit import PhaseUnit
from covsirphy.phase.sr_change import ChangeFinder


class PhaseSeries(Term):
    """
    A series of phases.

    Args:
        first_date (str): the first date of the series, like 22Jan2020
        last_date (str): the last date of the records, like 25May2020
        population (int): initial value of total population in the place
    """

    def __init__(self, first_date, last_date, population):
        self.first_date = self.ensure_date(first_date, "first_date")
        self.last_date = self.ensure_date(last_date, "last_date")
        self.init_population = self.ensure_population(population)
        # List of PhaseUnit
        self._units = []
        self.clear(include_past=True)

    def __iter__(self):
        yield from self._units

    def __len__(self):
        return len([unit for unit in self._units if unit])

    def unit(self, phase="last"):
        """
        Return the unit of the phase.

        Args:
            phase (str): phase name (1st etc.) or "last"

        Returns:
            covsirphy.PhaseUnit: the unit of the phase

        Notes:
            When @phase is 'last' and no phases were registered, returns A phase
            with the start/end dates are the previous date of the first date and initial population value.
        """
        if phase == "last":
            if self._units:
                return self._units[-1]
            pre_date = self.yesterday(self.first_date)
            return PhaseUnit(pre_date, pre_date, self.init_population)
        num = self.str2num(phase)
        try:
            return self._units[num]
        except IndexError:
            raise KeyError(f"{phase} phase is not registered.")

    def clear(self, include_past=False):
        """
        Clear phase information. Future phases will be always deleted.

        Args:
            include_past (bool): if True, include past phases.

        Returns:
            covsirphy.PhaseSeries: self
        """
        if include_past:
            self._units = []
        self._units = [unit for unit in self._units if unit <= self.last_date]
        return self

    def _calc_end_date(self, start_date, end_date=None, days=None):
        """
        Return the end date.

        Args:
            start_date (str): start date of the phase
            end_date (str): end date of the past phase, like 22Jan2020
            days (int or None): the number of days to add

        Returns:
            str: end date
        """
        if end_date is not None:
            self.ensure_date_order(start_date, end_date, name="end_date")
            return end_date
        if days is None:
            return self.last_date
        return self.date_change(start_date, days=days)

    def add(self, end_date=None, days=None, population=None, model=None, tau=None, **kwargs):
        """
        Add a past phase.

        Args:
            end_date (str): end date of the past phase, like 22Jan2020
            days (int or None): the number of days to add
            population (int or None): population value
            model (covsirphy.ModelBase): ODE model
            tau (int or None): tau value [min], a divisor of 1440 (prioritize the previous value)
            kwargs: keyword arguments of model parameters

        Returns:
            covsirphy.PhaseSeries: self

        Notes:
            If @population is None, the previous initial value will be used.
            When addition of past phases was not completed and the new phase is future phase, fill in the blank.
        """
        last_unit = self.unit(phase="last")
        # Basic information
        start_date = self.tomorrow(last_unit.end_date)
        end_date = self._calc_end_date(
            start_date, end_date=end_date, days=days)
        population = self.ensure_population(population or last_unit.population)
        model = model or last_unit.model
        tau = last_unit.tau or tau
        if model is None:
            param_dict = {}
        else:
            param_dict = {
                k: v for (k, v) in {**last_unit.to_dict(), **kwargs}.items()
                if k in model.PARAMETERS}
        # Create PhaseUnit
        unit = PhaseUnit(start_date, end_date, population)
        # Add phase if the last date is not included
        if self.last_date not in unit or unit <= self.last_date:
            unit.set_ode(model=model, tau=tau, **param_dict)
            self._units.append(unit)
            return self
        # Fill in the blank of past dates
        filling = PhaseUnit(start_date, self.last_date, population)
        filling.set_ode(model=model, tau=tau, **param_dict)
        target = PhaseUnit(
            self.tomorrow(self.last_date), end_date, population)
        target.set_ode(model=model, tau=tau, **param_dict)
        # Add new phase
        self._units.extend([filling, target])
        return self

    def delete(self, phase="last"):
        """
        Delete a phase. The phase will be combined to the previous phase.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd... or 'last'

        Returns:
            covsirphy.PhaseSeries: self

        Notes:
            When @phase is '0th', disable 0th phase. 0th phase will not be deleted.
            When @phase is 'last', the last phase will be deleted.
        """
        if phase == "0th":
            self.disable("0th")
            return self
        if self.unit(phase) == self.unit("last"):
            self._units = self._units[:-1]
            return self
        phase_pre = self.num2str(self.str2num(phase) - 1)
        unit_pre, unit_fol = self.unit(phase_pre), self.unit(phase)
        if unit_pre <= self.last_date and unit_fol >= self.last_date:
            phase_next = self.num2str(self.str2num(phase) + 1)
            unit_next = self.unit(phase_next)
            model = unit_next.model
            param_dict = {
                k: v for (k, v) in unit_next.to_dict().items() if k in model.PARAMETERS}
            unit_new = PhaseUnit(
                unit_fol.start_date, unit_next.end_date, unit_next.population)
            unit_new.set_ode(model=model, tau=unit_next.tau, **param_dict)
            return self
        unit_new = PhaseUnit(
            unit_pre.start_date, unit_fol.end_date, unit_pre.population)
        model = unit_pre.model
        if model is None:
            param_dict = {}
        else:
            param_dict = {
                k: v for (k, v) in unit_pre.to_dict().items() if k in model.PARAMETERS}
        unit_new.set_ode(model=model, tau=unit_pre.tau, **param_dict)
        units = [
            unit for unit in [unit_new, *self._units] if unit not in [unit_pre, unit_fol]]
        self._units = sorted(units)
        return self

    def disable(self, phase):
        """
        The phase will be disabled and removed from summary.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd...

        Returns:
            covsirphy.PhaseSeries: self
        """
        phase_id = self.str2num(phase)
        self._units[phase_id].disable()
        return self

    def enable(self, phase):
        """
        The phase will be enabled and appears in summary.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd...

        Returns:
            covsirphy.PhaseSeries: self
        """
        phase_id = self.str2num(phase)
        self._units[phase_id].enable()
        return self

    def summary(self):
        """
        Summarize the series of phases in a dataframe.

        Returns:
            (pandas.DataFrame):
                Index:
                    - phase name, like 1st, 2nd, 3rd...
                Columns:
                    - Type: 'Past' or 'Future'
                    - Start: start date of the phase
                    - End: end date of the phase
                    - Population: population value of the start date
                    - other information registered to the phases
        """
        info_dict = self.to_dict()
        if not info_dict:
            return pd.DataFrame(columns=[self.TENSE, self.START, self.END, self.N])
        # Convert to dataframe
        df = pd.DataFrame.from_dict(info_dict, orient="index")
        return df.dropna(how="all", axis=1).fillna(self.UNKNOWN)

    def to_dict(self):
        """
        Summarize the series of phase in a dictionary.

        Returns:
            (dict): nested dictionary of phase information
                - key (str): phase number, like 1th, 2nd,...
                - value (dict): phase information
                    - 'Type': (str) 'Past' or 'Future'
                    - values of PhaseUnit.to_dict()
        """
        return {
            self.num2str(phase_id): {
                self.TENSE: self.PAST if unit <= self.last_date else self.FUTURE,
                **unit.to_dict()
            }
            for (phase_id, unit) in enumerate(self._units) if unit
        }

    def replace(self, phase, new):
        """
        Replace phase object.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd...
            new (covsirphy.PhaseUnit): new phase object

        Returns:
            covsirphy.PhaseSeries: self
        """
        old = self.unit(phase)
        if old != new:
            raise ValueError(
                "Combination of start/end date is different. old: {old}, new: {new}")
        units = [unit for unit in self._units if unit != old]
        self._units = sorted(units + [new])
        return self

    def replaces(self, phase=None, new_list=None, keep_old=False):
        """
        Replace phase object.

        Args:
            phase (str or None): phase name, like 0th, 1st, 2nd...
            new_list (list[covsirphy.PhaseUnit]): new phase objects

        Returns:
            covsirphy.PhaseSeries: self

        Notes:
            If @phase is None and @keep_old is False, all old phases will be deleted.
            If @phase is not None, the phase will be deleted.
            @new_list must be specified.
        """
        if not isinstance(new_list, list):
            raise TypeError("@new_list must be a list of covsirphy.PhaseUnit.")
        type_ok = all(isinstance(unit, PhaseUnit) for unit in new_list)
        if not type_ok:
            raise TypeError("@new_list must be a list of covsirphy.PhaseUnit.")
        if phase is None:
            old_units = [
                unit for unit in self._units if unit not in new_list] if keep_old else []
        else:
            exc_unit = self.unit(phase)
            old_units = [unit for unit in self._units if unit != exc_unit]
        self._units = self._ensure_series(old_units + new_list)

    @classmethod
    def _ensure_series(cls, units):
        """
        Ensure that the list is a series of phases.

        Args:
            units (list[covsirphy.PhaseUnit]): list of units

        Returns:
            list[covsirphy.PhaseUnit]: sorted list of units

        Raises:
            ValueError: Phases are not series.
        """
        sorted_units = sorted(units)
        s = ", ".join([str(unit) for unit in sorted_units])
        for (i, unit) in enumerate(sorted_units):
            if i in [0, len(sorted_units) - 1]:
                continue
            sta_app = unit.start_date
            end_app = unit.end_date
            sta = cls.tomorrow(sorted_units[i - 1].end_date)
            end = cls.yesterday(sorted_units[i + 1].start_date)
            if sta != sta_app or end != end_app:
                raise ValueError(
                    f"The list of units does not a series of phases. Applied: {s}")
        return sorted_units

    def trend(self, sr_df, set_phases=True, area=None, show_figure=True, filename=None, **kwargs):
        """
        Perform S-R trend analysis.

        Args:
            sr_df (pandas.DataFrame)
                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases
                    - any other columns will be ignored
            set_phases (bool): if True, update phases
            area (str or None): area name
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of ChangeFinder()

        Returns:
            covsirphy.PhaseSeries: self
        """
        area = area or self.UNKNOWN
        sta = self.date_obj(self.first_date)
        end = self.date_obj(self.last_date)
        sr_df = sr_df.loc[(sr_df.index >= sta) & (sr_df.index <= end), :]
        finder = ChangeFinder(sr_df, **kwargs)
        if not set_phases:
            if show_figure:
                change_dates = [
                    unit.start_date for unit in self._units[1:] if unit <= self.last_date]
                finder.show(
                    area=area, change_dates=change_dates, filename=filename)
            return self
        # Find change points
        finder.run()
        # Show trends
        if show_figure:
            finder.show(area=area, filename=filename)
        # Register phases
        self.clear(include_past=True)
        _, end_dates = finder.date_range()
        [self.add(end_date=end_date) for end_date in end_dates]
        self.disable("0th")
        return self

    def simulate(self, record_df, y0_dict=None):
        """
        Simulate ODE models with set parameter values.

        Args:
            record_df (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases
            y0_dict (dict or None):
                - key (str): variable name
                - value (float): initial value
                - dictionary of initial values or None
                - if model will be changed in the later phase, must be specified

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
        dataframes = []
        rec_dates = record_df[self.DATE].dt.strftime(self.DATE_FORMAT).unique()
        for (num, unit) in enumerate(self._units):
            if not unit:
                continue
            if unit.start_date in rec_dates:
                unit.set_y0(record_df)
            else:
                try:
                    unit.set_y0(dataframes[-1])
                except IndexError:
                    pass
            df = unit.simulate(y0_dict=y0_dict)
            dataframes.append(df)
        sim_df = pd.concat(dataframes, ignore_index=True, sort=True)
        sim_df = sim_df.set_index(self.DATE).resample("D").last()
        sim_df = sim_df.astype(np.int64)
        return sim_df.reset_index()
