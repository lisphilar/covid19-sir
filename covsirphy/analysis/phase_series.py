#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import pandas as pd
from covsirphy.cleaning.word import Word


class PhaseSeries(Word):
    """
    A series of phases.
    """

    def __init__(self, first_date, population):
        """
        @first_date <str>: the first date of the series, like 22Jan2020
        @population <int>: initial value of total population in the place
        """
        # Phase ID of each date: {pd.TimeStamp: int}
        now = datetime.now()
        self.phase_dict = {
            date_obj: 0 for date_obj
            in pd.date_range(start=first_date, end=now, freq="D")
        }
        # Information of each phase ID: {int: dict[str]=str/dict}
        self.info_dict = {
            0: {
                self.START: first_date,
                self.END: now.strftime(self.DATE_FORMAT),
                self.N: population
            }
        }

    def add(self, start_date, end_date, population=None):
        """
        Add a new phase.
        @start_date <str>: start date of the new phase
        @end_date <str>: end date of the new phase
        @population <int>: population value of the start date
            - if None, initial value will be used
        @return self
        """
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
        return self

    def delete(self, phase):
        """
        Delete a phase.
        @phase <str>: phase name, like 0th, 1st, 2nd...
        @return self
        """
        try:
            phase_id = int(phase[:-2])
        except ValueError:
            raise ValueError("@phase is phase name, like 0th, 1st, 2nd...")
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
        """
        # Conver phase ID to phase name
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
                'Start': <str> start date of the phase,
                'End': <str> end date of the phase,
                'Population': <int>: population value at the start date
            }
        """
        # Add population value of initial phase using 1st phase
        try:
            self.info_dict[0][self.N] = self.info_dict[1][self.N]
        except KeyError:
            # 1st phase is not defined
            pass
        # Convert phase ID to phase name
        info_dict = {
            self.num2str(num): self.info_dict[num]
            for num in self.info_dict.keys()
        }
        # Convert to dataframe
        return info_dict

    def _tense(self, target_date, ref_date=None):
        """
        Return 'Past' or 'Future' for the targrt date.
        @target_date <str>: target date, like 22Jan2020
        @ref_date <str/None>: reference date
            - if None, will use today
        @return <str>: 'Past' or 'Future'
        """
        target_obj = datetime.strptime(target_date, self.DATE_FORMAT)
        if ref_date is None:
            ref_obj = datetime.now()
        else:
            ref_obj = datetime.strptime(ref_date, self.DATE_FORMAT)
        if target_obj <= ref_obj:
            return self.PAST
        return self.FUTURE
