#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import optuna
import pandas as pd
from covsirphy.analysis.phase_series import PhaseSeries
from covsirphy.cleaning.word import Word
from covsirphy.phase.trend import Trend
from covsirphy.phase.sr_data import SRData
from covsirphy.util.stopwatch import StopWatch


class ChangeFinder(Word):
    """
    Find change points of S-R trend.
    """
    optuna.logging.disable_default_handler()

    def __init__(self, clean_df, population, country, province=None,
                 population_change_dict=None):
        """

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
        # Setting for optimization
        self.n_points = 0
        self.min_duration = 0
        self.change_dates = list()
        self.change_dates_previous = list()
        self.study = None
        self.run_time = 0
        self.total_trials = 0

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

    def _init_study(self, seed=None):
        """
        Initialize Optuna study.

        Args:
            seed (int or None): random seed of hyperparameter optimization
                - this will effective when @n_jobs is 1
        """
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )

    def add_trial(self, n_trials_iteration=10, n_jobs=-1):
        """
        Run trial.

        Args:
            n_trials_iteration (int): the number of trials in one iteration
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
        """
        self.study.optimize(
            lambda x: self.objective(x),
            n_trials=n_trials_iteration,
            n_jobs=n_jobs
        )

    def run(self, n_points, min_duration=7, allowance=3,
            timeout=60, n_trials_iteration=10, n_jobs=-1, seed=None):
        """
        Run optimization.

        Args:
            n_points (int): the number of change points
            min_duration (int): minimum duration of one phase [days]
                - must be over 2
            allowance (int): allowance of change points [days]
                - if the estimated change points was equal to previous iteration
                with this allowance, stop running.
            timeout (int): time-out of run
            n_trials_iteration (int): the number of trials in one iteration
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            seed (int or None): random seed of hyperparameter optimization
                - this will effective when @n_jobs is 1

        Returns:
            self
        """
        self.n_points = n_points
        self.min_duration = min_duration
        stopwatch = StopWatch()
        if min_duration <= 2:
            raise ValueError("@min_duration must be over 2.")
        if n_points <= 0:
            self.run_time = 0
            self.total_trials = 0
            return self
        if seed is not None and n_jobs != 1:
            raise ValueError(
                "@seed must be None when @n_jobs is not equal to 1.")
        if self.study is None:
            self._init_study(seed=seed)
        print("Finding change points of S-R trend...")
        while True:
            self.add_trial(n_trials_iteration, n_jobs)
            # Check whether the change points are fixed (with allowance) or not
            allow_obj = timedelta(days=allowance)
            fixed_ok = [
                abs(self.date_obj(this) - self.date_obj(previous)) <= allow_obj
                for (this, previous)
                in zip(self.change_dates, self.change_dates_previous)
            ]
            # Calculate cumulative run-time
            self.run_time = stopwatch.stop()
            self.total_trials = len(self.study.trials)
            # If fixed or time-out, break
            if (all(fixed_ok) and self.change_dates_previous) or (self.run_time > timeout):
                print(
                    f"\rFinished {self.total_trials} trials in {stopwatch.show()}.\n",
                    end=str()
                )
                break
            stopwatch.stop()
            print(
                f"\rPerformed {self.total_trials} trials in {stopwatch.show()}.",
                end=str()
            )
            self.change_dates_previous = self.change_dates[:]
        return self

    def objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the parameter values using Optuna.

        Args:
            trial (optuna.trial): a trial of the study

        Returns:
            (float): score of the error function to minimize
        """
        # Suggest change points
        id_selected = [len(self.dates) - 1]
        for i in range(self.n_points):
            id_min = self.min_duration * (self.n_points - len(id_selected) + 1)
            id_max = id_selected[-1] - self.min_duration
            if id_min + self.min_duration > id_max:
                return np.inf
            id_new = trial.suggest_int(str(i), id_min, id_max)
            id_selected.append(id_new)
        self.change_dates = [
            self.dates[num] for num in sorted(id_selected[1:])
        ]
        # Calculate the start/end date of the phases
        start_dates, end_dates = self.phase_range(self.change_dates)
        return self.error_f(start_dates, end_dates)

    def phase_range(self, change_dates):
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

    def error_f(self, start_dates, end_dates):
        """
        Definition of error score to minimize in the study.
        This is weighted average of RMSLE scores.

        Args:
            start_dates (list[str]): list of start date of phases (candidates)
            end_dates (list[str]): list of end date of phases (candidates)

        Returns:
            (float) : score of the error function to minimize
        """
        scores = list()
        for (start_date, end_date) in zip(start_dates, end_dates):
            population = self.pop_dict[start_date]
            trend = Trend(
                self.clean_df, population, self.country, province=self.province,
                start_date=start_date, end_date=end_date
            )
            trend.analyse()
            scores.append(trend.rmsle())
        return np.average(scores, weights=range(1, len(scores) + 1))

    def _create_phases(self):
        """
        Create a dictionary of phases.

        Returns:
            (covsirphy.PhaseSeries)
        """
        start_dates, end_dates = self.phase_range(self.change_dates)
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
        # Create phase dictionary
        phase_series = self._create_phases()
        phase_dict = phase_series.to_dict()
        # Curve fitting
        df_list = list()
        vlines = list()
        for (phase, info) in phase_dict.items():
            start_date = info[self.START]
            end_date = info[self.END]
            population = info[self.N]
            trend = Trend(
                self.clean_df, population, self.country, province=self.province,
                start_date=start_date, end_date=end_date
            )
            trend.analyse()
            df = trend.result()
            # Get min value for vline
            vlines.append(df[self.R].min())
            # Rename the columns
            phase = phase.replace("0th", self.INITIAL)
            df = df.rename({f"{self.S}{self.P}": f"{phase}{self.P}"}, axis=1)
            df = df.rename({f"{self.S}{self.A}": f"{phase}{self.A}"}, axis=1)
            df = df.rename({f"{self.R}": f"{phase}_{self.R}"}, axis=1)
            df_list.append(df)
        if self.n_points == 0:
            comp_df = pd.concat(df_list, axis=1)
        else:
            comp_df = pd.concat(df_list[1:], axis=1)
        comp_df[self.R] = comp_df.fillna(0).loc[
            :, comp_df.columns.str.endswith(self.R)
        ].sum(axis=1)
        comp_df[f"{self.S}{self.A}"] = comp_df.fillna(0).loc[
            :, comp_df.columns.str.endswith(self.A)
        ].sum(axis=1)
        comp_df = comp_df.apply(
            lambda x: pd.to_numeric(x, errors="coerce", downcast="integer"),
            axis=0
        )
        # Show figure
        if not show_figure:
            return phase_series
        pred_cols = comp_df.loc[
            :, comp_df.columns.str.endswith(self.P)
        ].columns.tolist()
        if len(pred_cols) == 1:
            title = f"{self.area}: S-R trend without change points"
        else:
            change_str = ", ".join(self.change_dates)
            title = f"{self.area}: S-R trend changed on {change_str}"
        Trend.show_with_many(
            result_df=comp_df,
            predicted_cols=pred_cols,
            title=title,
            vlines=vlines[2:],
            filename=filename
        )
        return phase_series
