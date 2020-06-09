#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import itertools
import pandas as pd
from covsirphy.cleaning.word import Word


class PhaseSeries(Word):
    """
    A series of phases.
    """

    def __init__(self, first_date, last_record_date, population):
        """
        @first_date <str>: the first date of the series, like 22Jan2020
        @last_record_date <str>: the last date of the records, like 25May2020
        @population <int>: initial value of total population in the place
        """
        self.first_date = first_date
        self.last_record_date = last_record_date
        self.init_population = population
        self.clear(include_past=True)

    def clear(self, include_past=False):
        """
        Clear phase information.
        @include_past <bool>:
            - if True, include past phases.
            - future phase are always included
        return self
        """
        self.phase_dict = self._init_phase_dict(include_past=include_past)
        self.info_dict = self._init_info_dict(include_past=include_past)
        return self

    def _init_phase_dict(self, include_past=False):
        """
        Return initialized dictionary which is to remember phase ID of each date.
        @include_past <bool>:
            - if True, include past phases.
            - future phase are always included
        return <dict[pd.TimeStamp]=int>:
            - key: dates from the first date to the last date of the records
            - value: 0 (phase ID)
        """
        past_date_objects = pd.date_range(
            start=self.first_date, end=self.last_record_date, freq="D"
        )
        if include_past:
            return dict.fromkeys(past_date_objects, 0)
        last_date_obj = self.date_obj(self.last_record_date)
        phase_dict = {
            k: v for (k, v) in self.phase_dict.items()
            if k <= last_date_obj
        }
        return phase_dict

    def _init_info_dict(self, include_past=False):
        """
        Return initialized dictionary which is to remember phase information.
        @include_past <bool>:
            - if True, include past phases.
            - future phase are always included
        return <dict[str]=str/int>:
            - 'Start': the first date of the records
            - 'End': the last date of the records
            - 'Population': initial value of total population
        """
        if include_past:
            info_dict = {
                0: {
                    self.TENSE: self.PAST,
                    self.START: self.first_date,
                    self.END: self.last_record_date,
                    self.N: self.init_population
                }
            }
            return info_dict
        last_date_obj = self.date_obj(self.last_record_date)
        info_dict = {
            k: v for (k, v) in self.info_dict.items()
            if self.date_obj(v[self.END]) <= last_date_obj
        }
        return info_dict

    def _phase_name2id(self, phase):
        """
        Return phase ID of the phase.
        @phase <str>: phase name, like 1st, 2nd, 3rd,...
        @return <int>
        """
        try:
            num = int(phase[:-2])
        except ValueError:
            raise ValueError("@phase is phase name, like 0th, 1st, 2nd...")
        grouped_ids = list(itertools.groupby(self.phase_dict.values()))
        return grouped_ids[num][0]

    def add(self, start_date, end_date, population=None, **kwargs):
        """
        Add a new phase.
        @start_date <str>: start date of the new phase
        @end_date <str>: end date of the new phase
        @population <int>: population value of the start date
            - if None, initial value will be used
        @kwargs: keyword arguments to save as phase information
        @return self
        """
        # Arguments
        if population is None:
            population = self.population
        date_series = pd.date_range(
            start=start_date, end=end_date, freq="D"
        )
        new_id = max(self.phase_dict.values()) + 1
        # Tense of dates
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
        if end_obj <= min_end_obj:
            min_end_date = min_end_obj.strftime(self.DATE_FORMAT)
            raise ValueError(f"@end_date must be over {min_end_date}.")
        # Add new phase
        for date_obj in date_series:
            if date_obj in self.phase_dict.keys():
                if self.phase_dict[date_obj] != 0:
                    date_str = date_obj.strftime(self.DATE_FORMAT)
                    raise KeyError(
                        f"Phase has been registered for {date_str}.")
            self.phase_dict[date_obj] = new_id
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
        @phase <str>: phase name, like 0th, 1st, 2nd...
        @return self
        """
        phase_id = self._phase_name2id(phase)
        self.phase_dict = {
            k: 0 if v is phase_id else v
            for (k, v) in self.phase_dict.items()
        }
        self.info_dict.pop(phase_id)
        return self

    def summary(self):
        """
        Summarize the series of phases in a dataframe.
        @return <pd.DataFrame>:
            - index: phase name, like 1st, 2nd, 3rd...
            - Type: 'Past' or 'Future'
            - Start: start date of the phase
            - End: end date of the phase
            - Population: population value of the start date
            - values added by self.update()
        """
        # Convert phase ID to phase name
        info_dict = self.to_dict()
        # Convert to dataframe
        df = pd.DataFrame.from_dict(info_dict, orient="index")
        return df.fillna(self.UNKNOWN)

    def to_dict(self):
        """
        Summarize the series of phase in a dictionary.
        @return <dict[str]={str: str/int}>:
            - key: phase number, like 1th, 2nd,...
            - value: {
                'Type': <str> 'Past' or 'Future'
                'Start': <str> start date of the phase,
                'End': <str> end date of the phase,
                'Population': <int> population value at the start date
                - values added by self.update()
            }
        """
        # Convert phase ID to phase name
        info_dict = {
            self.num2str(num): self.info_dict[num]
            for num in self.info_dict.keys()
        }
        # Convert to dataframe
        return info_dict

    def _tense(self, target_date, ref_date=None):
        """
        Return 'Past' or 'Future' for the target date.
        @target_date <str>: target date, like 22Jan2020
        @ref_date <str/None>: reference date
            - if None, will use last date of the records
        @return <str>: 'Past' or 'Future'
        """
        target_obj = datetime.strptime(target_date, self.DATE_FORMAT)
        ref_date = self.last_record_date if ref_date is None else ref_date
        ref_obj = datetime.strptime(ref_date, self.DATE_FORMAT)
        if target_obj <= ref_obj:
            return self.PAST
        return self.FUTURE

    def update(self, phase, **kwargs):
        """
        Update information of the phase.
        @phase <str>: phase name
        @kwargs: keyword arguments to add
        """
        phase_id = phase_id = self._phase_name2id(phase)
        self.info_dict[phase_id].update(kwargs)
        return self

    def next_date(self):
        """
        Return the next date of the end date of the last registered phase.
        return <str>: like 01Feb2020
        """
        un_date_objects = [
            k for (k, v) in self.phase_dict.items()
            if v != 0
        ]
        if not un_date_objects:
            last_end_date_obj = list(self.phase_dict.keys())[-1]
        else:
            last_end_date_obj = un_date_objects[-1]
        next_date_obj = last_end_date_obj + timedelta(days=1)
        return next_date_obj.strftime(self.DATE_FORMAT)
