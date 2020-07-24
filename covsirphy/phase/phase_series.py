#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import itertools
import numpy as np
import pandas as pd
import scipy
from covsirphy.cleaning.term import Term


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
        self.first_date = first_date
        self.last_date = last_date
        self.init_population = population
        self.clear(include_past=True)
        # Whether use 0th phase or not
        self.use_0th = use_0th

    def clear(self, include_past=False):
        """
        Clear phase information.

        Args:
            include_past (bool):
                - if True, include past phases.
                - future phase are always included

        Returns:
            self
        """
        self.phase_dict = self._init_phase_dict(include_past=include_past)
        self.info_dict = self._init_info_dict(include_past=include_past)
        return self

    def _init_phase_dict(self, include_past=False):
        """
        Return initialized dictionary which is to remember phase ID of each date.

        Args:
            include_past (bool):
                - if True, include past phases.
                - future phase are always included

        Returns:
            dict
                - key (pd.TimeStamp): dates from the first date to the last date of the records
                - value (int): -1 (phase ID)
        """
        if include_past or not hasattr(self, "phase_dict"):
            past_date_objects = pd.date_range(
                start=self.first_date, end=self.last_date, freq="D"
            )
            return dict.fromkeys(past_date_objects, -1)
        last_date_obj = self.date_obj(self.last_date)
        phase_dict = {
            k: v for (k, v) in self.phase_dict.items()
            if k <= last_date_obj
        }
        return phase_dict

    def _init_info_dict(self, include_past=False):
        """
        Return initialized dictionary which is to remember phase information.

        Args:
            include_past (bool):
                - if True, include past phases.
                - future phase are always included

        Attributes:
            info_dict (dict)
                - 'Start': the first date of the records
                - 'End': the last date of the records
                - 'Population': initial value of total population
        """
        if include_past or not hasattr(self, "info_dict"):
            info_dict = dict()
            return info_dict
        last_date_obj = self.date_obj(self.last_date)
        info_dict = {
            k: v for (k, v) in self.info_dict.items()
            if self.date_obj(v[self.END]) <= last_date_obj
        }
        return info_dict

    def _phase_name2id(self, phase):
        """
        Return phase ID of the phase.

        Args:
            phase (str): phase name, like 1st, 2nd, 3rd,...

        Returns:
            int
        """
        try:
            num = int(phase[:-2])
        except ValueError:
            raise ValueError(
                "@phase must be a phase name, like 0th, 1st, 2nd...")
        grouped_ids = list(itertools.groupby(self.phase_dict.values()))
        return grouped_ids[num][0]

    def _add(self, new_id, start_date, end_date):
        """
        Add new phase to self.

        Args:
            new_id (int): ID number of the new phase
            start_date (str): start date of the new phase
            end_date (str): end date of the new phase
        """
        date_series = pd.date_range(
            start=start_date, end=end_date, freq="D"
        )
        new_phase_dict = {
            date_obj: new_id for date_obj in date_series
        }
        unfree_set = set(k for (k, v) in self.phase_dict.items() if v != -1)
        intersection = set(new_phase_dict.keys()) & unfree_set
        if intersection:
            date_strings = [
                date_obj.strftime(self.DATE_FORMAT) for date_obj in intersection
            ]
            dates_str = ", ".join(date_strings)
            raise KeyError(
                f"Phases have been registered for {dates_str}.")
        self.phase_dict.update(new_phase_dict)

    def add(self, start_date, end_date, population=None, **kwargs):
        """
        Add a new phase.

        Args:
            start_date (str): start date of the new phase
            end_date (str): end date of the new phase
            population (int): population value of the start date
            kwargs: keyword arguments to save as phase information

        Returns:
            self

        Notes:
            if @population is None, initial value will be used.
        """
        # Arguments
        population = population or self.init_population
        # Phase ID
        new_id = max(self.phase_dict.values()) + 1
        if new_id == 0 and not self.use_0th:
            new_id = 1
        # Check tense of dates
        start_tense = self._tense(start_date)
        end_tense = self._tense(end_date)
        if start_tense != end_tense:
            raise ValueError(
                f"@start_date is {start_tense}, but @end_date is {end_tense}."
            )
        # end_date must be over start_date - 2
        start_obj = self.date_obj(start_date)
        end_obj = self.date_obj(end_date)
        min_end_obj = start_obj + timedelta(days=2)
        if end_obj < min_end_obj:
            min_end_date = min_end_obj.strftime(self.DATE_FORMAT)
            raise ValueError(
                f"@end_date must be the same or over {min_end_date}, but {end_date} was applied."
            )
        # Add new phase
        self._add(new_id, start_date, end_date)
        # Add phase information
        self.info_dict[new_id] = {
            self.TENSE: start_tense,
            self.START: start_date,
            self.END: end_date,
            self.N: population
        }
        self.info_dict[new_id].update(**kwargs)
        return self

    def delete(self, phase):
        """
        Delete a phase.

        Args:
            phase (str): phase name, like 0th, 1st, 2nd...

        Returns:
            self
        """
        phase_id = self._phase_name2id(phase)
        self.phase_dict = {
            k: -1 if v == phase_id else v
            for (k, v) in self.phase_dict.items()
        }
        self.info_dict.pop(phase_id)
        if not phase_id:
            self.use_0th = False
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
                    - values added by self.update()
        """
        self.reset_phase_names()
        # Convert phase ID to phase name
        info_dict = self.to_dict()
        # Convert to dataframe
        df = pd.DataFrame.from_dict(info_dict, orient="index")
        df = df.fillna(self.UNKNOWN)
        # If empty, add column names
        if df.empty:
            return pd.DataFrame(columns=[self.TENSE, self.START, self.END, self.N])
        return df

    def to_dict(self):
        """
        Summarize the series of phase in a dictionary.

        Returns:
            (dict): nested dictionary of phase information
                - key (str): phase number, like 1th, 2nd,...
                - value (dict): phase information
                    - 'Type': (str) 'Past' or 'Future'
                    - 'Start': (str) start date of the phase,
                    - 'End': (str) end date of the phase,
                    - 'Population': (int) population value at the start date
                    - values added by PhaseSeries.update()
        """
        # Convert phase ID to phase name
        info_dict = {
            self.num2str(num): self.info_dict[num]
            for num in self.info_dict.keys() if num != -1
        }
        # Return the dictionary
        return info_dict

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

    def update(self, phase, **kwargs):
        """
        Update information of the phase.

        Args:
            phase (str): phase name
            kwargs: keyword arguments to add
        """
        phase_id = self._phase_name2id(phase)
        self.info_dict[phase_id].update(kwargs)
        return self

    def last_object(self):
        """
        Return the end date of the last registered phase.

        Returns:
            (datetime.datetime): the end date of the last registered phase
        """
        un_date_objects = [
            k for (k, v) in self.phase_dict.items()
            if v != -1
        ]
        if not un_date_objects:
            return self.to_date_obj(self.first_date) - timedelta(days=1)
        return un_date_objects[-1]

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
        start_objects = [
            self.date_obj(v[self.START]) for v in self.info_dict.values()
        ]
        return start_objects

    def end_objects(self):
        """
        Return the list of end dates as datetime.datetime objects of phases.

        Returns:
            (list[datetime.datetime]): list of end dates
        """
        end_objects = [
            self.date_obj(v[self.END]) for v in self.info_dict.values()
        ]
        return end_objects

    def reset_phase_names(self):
        """
        Reset phase names.
        eg. 1st, 4th, 2nd,.. to 1st, 2nd, 3rd, 4th,...

        Returns:
            self: instance of covsirphy.PhaseSeries
        """
        start_objects = np.array(self.start_objects())
        ascending = scipy.stats.rankdata(start_objects)
        if self.use_0th:
            ascending = ascending - 1
        corres_dict = {
            phase_id: int(rank)
            for (phase_id, rank) in zip(self.info_dict.keys(), ascending)
        }
        corres_dict[-1] = -1
        self.phase_dict = {
            date_obj: corres_dict[phase_id]
            for (date_obj, phase_id) in self.phase_dict.items()
        }
        info_dict = {
            corres_dict[phase_id]: info
            for (phase_id, info) in self.info_dict.items()
        }
        self.info_dict = dict(sorted(info_dict.items()))
        return self

    @staticmethod
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
        step_n_list = [
            round(diff.total_seconds() / 60 / tau) for diff
            in date_array[1:] - date_array[:-1]
        ]
        return step_n_list

    def model_names(self):
        """
        Return the names of the registered models if available.

        Returns:
            (list[str]): list of model names
        """
        try:
            names = [
                v[self.ODE] for v in self.info_dict.values()
            ]
        except KeyError:
            names = list()
        return names

    def population_values(self):
        """
        Return the list of population values.

        Returns:
            (list[int]): list of population values
        """
        values = [
            v[self.N] for v in self.info_dict.values()
        ]
        return values

    def phases(self):
        """
        Return the list of phase names.

        Returns:
            (list[int]): list of phase names
        """
        phase_list = [
            self.num2str(num) for num
            in self.info_dict.keys() if num != -1
        ]
        return phase_list
