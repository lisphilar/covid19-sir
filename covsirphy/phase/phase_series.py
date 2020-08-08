#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import numpy as np
import pandas as pd
import scipy
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
        use_0th (bool): if True, phase names will be 0th, 1st,... If False, 1st, 2nd,...
    """

    def __init__(self, first_date, last_date, population, use_0th=True):
        self.first_date = self.ensure_date(first_date, "first_date")
        self.last_date = self.ensure_date(last_date, "last_date")
        self.init_population = self.ensure_population(population)
        self.clear(include_past=True)
        # Whether use 0th phase or not
        self.use_0th = use_0th
        # {phase name: PhaseUnit}
        self._phase_dict = {}

    @property
    def phase_dict(self):
        """
        dict(str, covsirphy.PhaseUnit): dictionary of phase
        """
        return self._phase_dict

    def clear(self, include_past=False):
        """
        Clear phase information.

        Args:
            include_past (bool): if True, include past phases.

        Returns:
            self

        Notes:
            Future phases will be always deleted.
        """
        if include_past:
            self._phase_dict = {}
        self._phase_dict = {
            name: instance for (name, instance) in self._phase_dict.items()
            if self._tense(instance.start_date) == self.PAST
        }
        return self

    def _tense(self, target_date, ref_date=None):
        """
        Return 'Past' or 'Future' for the target date.

        Args:
            target_date (str): target date, like 22Jan2020
            ref_date (str or None): reference date

        Returns:
            (str): 'Past' or 'Future'

        Notes:
            If @ref_date is None, the last date of the records will be used.
        """
        target_obj = self.date_obj(target_date)
        ref_obj = self.date_obj(ref_date or self.last_date)
        if target_obj <= ref_obj:
            return self.PAST
        return self.FUTURE

    def _last_phase(self):
        """
        Return PhaseUnit instance of the last phase.

        Returns:
            tuple(str or None, covsirphy.PhaseUnit or None): phase name and phase information

        Notes:
            if no phases were registered, return None.
        """
        if not self._phase_dict:
            return (None, None)
        last_num = len(self._phase_dict)
        if self.use_0th:
            last_num -= 1
        last_id = self.num2str(last_num)
        return (last_id, self._phase_dict[last_id])

    @classmethod
    def _calc_end_date(cls, start_date, days):
        """
        Return the end date of a phase with the number of days.

        Returns:
            str: end date
        """
        days = cls.ensure_natural_int(days, name="days", none_ok=False)
        sta = cls.date_obj(start_date)
        end = sta + timedelta(days=days)
        return end.strftime(cls.DATE_FORMAT)

    def add(self, start_date=None, end_date=None, days=None, population=None, **kwargs):
        """
        Add a past phase.

        Args:
            start_date (str): start date of the phase
            end_date (str): end date of the past phase, like 22Jan2020
            days (int or None): the number of days to add
            population (int or None): population value
            **kwargs: keyword arguments of PhaseUnit.set_ode()

        Notes:
            If @population is None, the previous initial value will be used.
        """
        # Phase information
        last_id, last_phase = self._last_phase()
        if last_phase is None:
            # Initial phase
            phase_id = self.num2str(0) if self.use_0th else self.num2str(1)
            start_date = start_date or self.first_date
            population = population or self.init_population
        else:
            phase_id = self.num2str(self.str2num(last_id) + 1)
            last_dict = last_phase.to_dict()
            start_date = start_date or self.tomorrow(last_dict[self.END])
            population = population or last_dict[self.N]
        # End date
        if end_date is None:
            if days is None:
                raise ValueError(
                    "Either @end_date or @days must be specified.")
            end_date = self._calc_end_date(start_date=start_date, days=days)
        else:
            end_date = self.ensure_date(end_date)
        if self._tense(start_date) != self._tense(end_date):
            raise ValueError(
                f"@start_date={start_date} is past, but @end_date={end_date} is future."
            )
        # Register PhaseUnit
        phase = PhaseUnit(start_date, end_date, population)
        if "model" in kwargs:
            model = kwargs.pop("model")
            phase.set_ode(model=model, **kwargs)
        self._phase_dict[phase_id] = phase

    def phase(self, name="last"):
        """
        Return the phase as a instance of PhaseUnit.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd...

        Returns:
            covsirphy.PhaseSeries: self
        """
        if name == "last":
            phase_numbers = [
                self.str2num(ph) for ph in self._phase_dict.keys()]
            name = self.num2str(max(phase_numbers))
        try:
            return self._phase_dict[name]
        except KeyError:
            raise KeyError(f"{name} phase is not registered.")

    def reset_phase_names(self):
        """
        Reset phase names.
        eg. 1st, 4th, 2nd,.. to 1st, 2nd, 3rd, 4th,...

        Returns:
            covsirphy.PhaseSeries: self
        """
        phase_dict = self._phase_dict.copy()
        # Calculate order
        start_objects = np.array(
            [
                self.date_obj(phase.start_date) for phase in phase_dict.values()
            ]
        )
        ascending = scipy.stats.rankdata(start_objects)
        if self.use_0th:
            ascending = ascending - 1
        corres_dict = {
            old_id: int(rank) for (old_id, rank)
            in zip(phase_dict.keys(), ascending)
        }
        # Renumber
        phase_dict = {
            corres_dict[k]: v for (k, v) in phase_dict.items()
        }
        # Reorder
        sort_nest = sorted(phase_dict.items(), key=lambda x: x[0])
        self._phase_dict = {self.num2str(k): v for (k, v) in sort_nest}
        return self

    def delete(self, phase):
        """
        Delete a phase. The phase included in the previous phase.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd...

        Returns:
            covsirphy.PhaseSeries: self
        """
        # Delete the target phase
        post_phase = self._phase_dict.pop(phase)
        # Whether the previous phase exists or not
        pre_phase_id = self.num2str(self.str2num(phase) - 1)
        if pre_phase_id not in self.phase_dict:
            self.reset_phase_names()
            return self
        # Delete the previous phase
        pre_phase = self._phase_dict.pop(pre_phase_id)
        # Register new phase
        self._phase_dict[pre_phase_id] = PhaseUnit(
            start_date=pre_phase.start_date,
            end_date=post_phase.end_date,
            population=pre_phase.population
        )
        self.reset_phase_names()
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
        if not self._phase_dict:
            return pd.DataFrame(columns=[self.TENSE, self.START, self.END, self.N])
        # Convert to dataframe
        info_dict = self.to_dict()
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
        self.reset_phase_names()
        return {
            phase_id: {
                self.TENSE: self._tense(phase.start_date),
                **phase.to_dict()
            }
            for (phase_id, phase) in self._phase_dict.items()
        }

    def last_object(self):
        """
        Return the end date of the last registered phase.

        Returns:
            (datetime.datetime): the end date of the last registered phase
        """
        _, last_phase = self._last_phase()
        if last_phase is None:
            return self.date_obj(self.first_date)
        return last_phase.end_date

    def next_date(self):
        """
        Return the next date of the end date of the last registered phase.
        Returns:
            (str): like 01Feb2020
        """
        last_date_obj = self.last_object()
        next_date_obj = last_date_obj + timedelta(days=1)
        return next_date_obj.strftime(self.DATE_FORMAT)

    def start_objects(self):
        """
        Return the list of start dates as datetime.datetime objects of phases.

        Returns:
            (list[datetime.datetime]): list of start dates
        """
        return [
            self.date_obj(phase.start_date) for phase in self._phase_dict.values()
        ]

    def end_objects(self):
        """
        Return the list of end dates as datetime.datetime objects of phases.

        Returns:
            (list[datetime.datetime]): list of end dates
        """
        return [
            self.date_obj(phase.end_date) for phase in self._phase_dict.values()
        ]

    def tenses(self):
        """
        Return the list of tense of start dates, 'Past' or 'Future'.

        Returns:
            list[str]: list of tenses
        """
        return [
            self._tense(phase.end_date) for phase in self._phase_dict.values()
        ]

    @ staticmethod
    def number_of_steps(start_objects, last_object, tau):
        """
        Return the list of the number of steps of phases.

        Args:
            start_objects (list[datetime.datetime]): list of start dates
            last_object (datetime.datetime): the end date of the last registered phase
            tau (int): tau value

        Returns:
            (list[int]): list of the number of steps
        """
        date_array = np.array([*start_objects, last_object])
        return [
            round(diff.total_seconds() / 60 / tau) for diff
            in date_array[1:] - date_array[:-1]
        ]

    def model_names(self):
        """
        Return the names of the registered models if available.

        Returns:
            (list[str]): list of model names
        """
        try:
            names = [
                phase.to_dict[self.ODE] for phase in self._phase_dict.values()
            ]
        except KeyError:
            names = []
        return names

    def population_values(self):
        """
        Return the list of population values.

        Returns:
            (list[int]): list of population values
        """
        return [
            phase.population for phase in self._phase_dict.values()
        ]

    def phases(self, include_future=True):
        """
        Return the list of phase names.

        Args:
            include_future (bool): include future phases or not

        Returns:
            (list[str]): list of phase names
        """
        if include_future:
            return list(self._phase_dict.keys())
        return [
            phase for (phase, unit) in self._phase_dict.items()
            if self._tense(unit.start_date) == self.PAST
        ]

    def replace(self, phase, new):
        """
        Replace phase object.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd...
            new (covsirphy.PhaseUnit): new phase object
        """
        if phase not in self._phase_dict:
            raise KeyError(f"{phase} phase is not registered.")
        old = self._phase_dict[phase]
        if old.start_date != new.start_date:
            raise ValueError(
                f"Start date is different from {old.start_date}, {new.start_date} was applied.")
        if old.end_date != new.end_date:
            raise ValueError(
                f"Start date is different from {old.start_date}, {new.start_date} was applied.")
        self._phase_dict[phase] = new

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
        finder = ChangeFinder(sr_df, **kwargs)
        if not set_phases:
            if show_figure:
                start_dates = [
                    date.strftime(self.DATE_FORMAT) for date in self.start_objects]
                finder.show(
                    area or self.UNKNOWN,
                    change_dates=start_dates[1:],
                    use_0th=self.use_0th,
                    filename=filename
                )
            return self
        # Find change points
        finder.run()
        # Show trends
        if show_figure:
            finder.show(
                area=area or self.UNKNOWN, use_0th=True, filename=filename)
        # Register phases
        self.clear(include_past=True)
        _, end_dates = finder.date_range()
        use_0th, self.use_0th = self.use_0th, True
        [self.add(end_date=end_date) for end_date in end_dates]
        self.delete("0th")
        self.use_0th = use_0th
        return self
