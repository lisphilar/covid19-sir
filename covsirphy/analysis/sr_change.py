#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ruptures as rpt
from covsirphy.analysis.phase_series import PhaseSeries
from covsirphy.cleaning.word import Word
from covsirphy.phase.sr_data import SRData


class ChangeFinder(Word):
    """
    Find change points of S-R trend.

    Args:
        clean_df (pandas.DataFrame): cleaned data

            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Country (str): country/region name
                - Province (str): province/prefecture/sstate name
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases

        population (int): initial value of total population in the place
        country (str): country name
        province (str): province name
        population_change_dict (dict): dictionary of total population
            - key (str): start date of population change
            - value (int or None): total population
    """

    def __init__(self, clean_df, population, country, province=None,
                 population_change_dict=None):
        # Arguments
        self.clean_df = clean_df.copy()
        self.country = country
        self.province = province
        if province is None:
            self.area = country
        else:
            self.area = f"{country}{self.SEP}{province}"
        self.dates = self._get_dates(clean_df, population, country, province)
        self.pop_dict = self._read_population_data(
            self.dates, population, population_change_dict
        )
        self.population = population
        # Setting for optimization

    def run(self):
        """
        Run optimization and find change points.
        # This method must be revised for issue3

        Returns:
            self
        """
        # TODO: This method must be revised for issue3
        return self

    def show(self, show_figure=True, filename=None):
        """
        show the result as a figure and return a dictionary of phases.

        Args:
        @show_figure (bool):
            - if True, show the result as a figure.
        @filename (str): filename of the figure, or None (show figure)

        Returns:
            (covsirphy.PhaseSeries)
        """
        # This method must be revised for issue3
        # TODO: Create phase dictionary
        phase_series = self._create_phases()
        return phase_series

    def _get_dates(self, clean_df, population, country, province):
        """
        Get dates from the dataset.

        Args:
            clean_df (pandas.DataFrame): cleaned data

                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/sstate name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            population (int): initial value of total population in the place
            country (str): country name
            province (str): province name

        Returns:
            (list[str]): list of dates, like 22Jan2020
        """
        sr_data = SRData(clean_df, country=country, province=province)
        df = sr_data.make(population)
        dates = [date_obj.strftime(self.DATE_FORMAT) for date_obj in df.index]
        return dates

    def _read_population_data(self, dates, population, change_dict=None):
        """
        Make population dictionary easy to use in this class.

        Args:
            dates (list[str]): list of dates, like 22Jan2020
            population (int): initial value of total population in the place
            change_dict (dict): dictionary of total population
                    - key (str): start date of population change
                    - value (int or None): total population

        Returns:
            (dict)
                - key (str): date, like 22Jan2020
                - value (int): total population on the date
        """
        change_dict = dict() if change_dict is None else change_dict.copy()
        population_now = population
        pop_dict = dict()
        for date in dates:
            if date in change_dict.keys():
                population_now = change_dict[date]
            pop_dict[date] = population_now
        return pop_dict

    def _phase_range(self, change_dates):
        """
        Return the start date and end date of the phases.

        Args:
            change_dates (list[str]): list of change points, like 22Jan2020

        Returns:
            (tuple)
                list[str]: list of start dates
                list[str]: list of end dates
        """
        start_dates = [self.dates[0], *change_dates]
        end_dates_without_last = [
            (
                datetime.strptime(date, self.DATE_FORMAT) - timedelta(days=1)
            ).strftime(self.DATE_FORMAT)
            for date in change_dates
        ]
        end_dates = [*end_dates_without_last, self.dates[-1]]
        return (start_dates, end_dates)

    def _create_phases(self):
        """
        Create a dictionary of phases.

        Returns:
            (covsirphy.PhaseSeries)
        """
        start_dates, end_dates = self._phase_range(self.change_dates)
        pop_list = [self.pop_dict[date] for date in start_dates]
        phases = [self.num2str(num) for num in range(len(start_dates))]
        phase_series = PhaseSeries(
            self.dates[0], self.dates[-1], self.population
        )
        phase_itr = enumerate(zip(start_dates, end_dates, pop_list, phases))
        for (i, (start_date, end_date, population, phase)) in phase_itr:
            if i == 0:
                continue
            phase_series.add(
                start_date=start_date,
                end_date=end_date,
                population=population
            )
        return phase_series
