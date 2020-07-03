#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt
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
        self.dates = self.get_dates(clean_df, population, country, province)
        self.pop_dict = self._read_population_data(
            self.dates, population, population_change_dict
        )
        self.population = population
        # Dataset for analysis
        # Index: Date(pd.TimeStamp, Columns: Recovered (int) and Susceptible_actual (int)
        sr_data = SRData(clean_df, country=country, province=province)
        self.sr_df = sr_data.make(population)
        # Setting for optimization
        self.change_dates = list()

    def run(self):
        """
        Run optimization and find change points.

        Returns:
            self
        """
        # TODO: This method must be revised for issue3
        # Dataset for analysis
        # Index: Date(pd.TimeStamp, Columns: Recovered (int) and Susceptible_actual (int)
        # As defined in Word class,
        # 'Date' == self.DATE, 'Recovered' == self.R, 'Susceptible_actual' == f'{self.S}{self.A}'
        # TODO: Convert the dataframe for S-R analysis
        sr_df = self.sr_df.rename({f"{self.S}{self.A}": self.S}, axis=1)
        df = sr_df.pivot_table(index=self.R, values=self.S, aggfunc="last")
        serial_df = pd.DataFrame(np.arange(1, df.index.max() + 1, 1))
        serial_df.index += 1
        df = pd.merge(
            df, serial_df, left_index=True, right_index=True, how="outer"
        )
        series = df.reset_index(drop=True).iloc[:, 0]
        series = series.interpolate(limit_direction="both")
        series = series.astype(np.int64)
        samples = np.linspace(
            0, series.index.max(), len(self.sr_df), dtype=np.int64
        )
        series = series[samples]
        # Detection
        # TODO: Revise parameters
        algorithm = rpt.Pelt(model="rbf", jump=2, min_size=6)
        results = algorithm.fit_predict(series.values, pen=0.5)
        reset_series = series.reset_index(drop=True)
        reset_series.index += 1
        susceptible_df = reset_series[results].reset_index()
        df = pd.merge_asof(
            susceptible_df.sort_values(self.S),
            sr_df.reset_index().sort_values(self.S),
            on=self.S, direction="nearest"
        )
        # TODO: Set change points
        change_dates = df["Date"].sort_values()[:-1].dt.strftime("%d%b%Y")
        self.change_dates = change_dates.tolist()
        return self

    def show(self, show_figure=True, filename=None):
        """
        show the result as a figure and return a dictionary of phases.

        Args:
            @show_figure (bool): if True, show the result as a figure.
            @filename (str): filename of the figure, or None (display figure)

        Returns:
            (covsirphy.PhaseSeries)
        """
        # TODO: This method must be revised for issue3
        phase_series = self._create_phases()
        # Show figure or save figure
        if show_figure and filename is None:
            # rpt.display(self.sr_df_for_analysis, self.result)
            # plt.show()
            pass
        return phase_series

    def get_dates(self, clean_df, population, country, province):
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
