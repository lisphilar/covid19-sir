#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from covsirphy.cleaning.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.phase.trend import Trend
from covsirphy.phase.phase_series import PhaseSeries


class ChangeFinder(Term):
    """
    Find change points of S-R trend.

    Args:
        jhu_data (covsirphy.JHUData): object of records
        population (int): initial value of total population in the place
        country (str): country name
        province (str): province name
        population_change_dict (dict): dictionary of total population
            - key (str): start date of population change
            - value (int or None): total population
        min_size (int): minimum value of phase length [days], over 2
        max_rmsle (float): minmum value of RMSLE score
        start_date (str or None): start date, like 22Jan2020
        end_date (str or None): end date, like 01Feb2020

    Notes:
        When RMSLE score > max_rmsle, predicted values will be None
    """

    def __init__(self, jhu_data, population, country, province=None,
                 population_change_dict=None, min_size=7, max_rmsle=20.0,
                 start_date=None, end_date=None):
        # Dataset
        if isinstance(jhu_data, pd.DataFrame):
            warnings.warn(
                "Please use instance of JHUData as the first argument of ChangeFinder class.",
                DeprecationWarning,
                stacklevel=2
            )
            jhu_data = JHUData.from_dataframe(jhu_data)
        self.jhu_data = self.ensure_instance(
            jhu_data, JHUData, name="jhu_data"
        )
        # Area
        self.country = country
        self.province = province or self.UNKNOWN
        self.area = JHUData.area_name(country, self.province)
        # Minimum size of records
        self.min_size = self.ensure_natural_int(min_size, "min_size")
        # Dataset for analysis
        self.population = self.ensure_natural_int(
            population, name="population")
        self.sr_df = jhu_data.to_sr(
            country=self.country, province=self.province, population=self.population,
            start_date=start_date, end_date=end_date
        )
        self.dates = self.get_dates(self.sr_df)
        # Check length of records
        if self.min_size < 3:
            raise ValueError(
                f"@min_size must be over 2, but {min_size} was applied.")
        if len(self.dates) < self.min_size * 2:
            raise ValueError(
                f"More than {min_size * 2} records must be included.")
        # Population
        self.pop_dict = self._read_population_data(
            self.dates, self.population, population_change_dict
        )
        # Minimum value of RMSLE score
        self.max_rmsle = self.ensure_float(max_rmsle)
        # Setting for optimization
        self._change_dates = list()
        # Whether use 0th phase or not
        self._use_0th = True

    def run(self):
        """
        Run optimization and find change points.

        Returns:
            self
        """
        # Convert the dataset, index: Recovered, column: log10(Susceptible)
        sr_df = self.sr_df.copy()
        sr_df[self.S] = np.log10(sr_df[self.S].astype(np.float64))
        df = sr_df.pivot_table(index=self.R, values=self.S, aggfunc="last")
        # Convert index to serial numbers
        serial_df = pd.DataFrame(np.arange(1, df.index.max() + 1, 1))
        serial_df.index += 1
        df = pd.merge(
            df, serial_df, left_index=True, right_index=True, how="outer"
        )
        series = df.reset_index(drop=True).iloc[:, 0]
        series = series.interpolate(limit_direction="both")
        # Sampling to reduce run-time of Ruptures
        samples = np.linspace(
            0, series.index.max(), len(self.sr_df), dtype=np.int64
        )
        series = series[samples]
        # Detection with Ruptures
        algorithm = rpt.Pelt(model="rbf", jump=2, min_size=self.min_size)
        results = algorithm.fit_predict(series.values, pen=0.5)
        # Convert index values to Susceptible values
        reset_series = series.reset_index(drop=True)
        reset_series.index += 1
        susceptible_df = reset_series[results].reset_index()
        # Convert Susceptible values to dates
        df = pd.merge_asof(
            susceptible_df.sort_values(self.S),
            sr_df.reset_index().sort_values(self.S),
            on=self.S, direction="nearest"
        )
        found_list = df[self.DATE].sort_values()[:-1]
        # Only use dates when the previous phase has more than {min_size + 1} days
        delta_days = timedelta(days=self.min_size)
        first_obj = self.to_date_obj(self.dates[0])
        last_obj = self.to_date_obj(self.dates[-1])
        effective_list = [first_obj]
        for found in found_list:
            if effective_list[-1] + delta_days < found:
                effective_list.append(found)
        # The last change date must be under the last date of records {- min_size} days
        if effective_list[-1] >= last_obj - delta_days:
            effective_list = effective_list[:-1]
        # Set change points
        self._change_dates = [
            date.strftime(self.DATE_FORMAT) for date in effective_list[1:]
        ]
        return self

    @property
    def change_dates(self):
        """
        list[str]: list of change points (01Feb2020 etc.)
        """
        return self._change_dates

    @change_dates.setter
    def change_dates(self, dates):
        self._change_dates = [
            date.strftime(self.DATE_FORMAT) for date in dates
        ]

    @property
    def use_0th(self):
        """
        bool: if True, phase names will be 0th, 1st,... If False, 1st, 2nd,...
        """
        return self._use_0th

    @use_0th.setter
    def use_0th(self, should_use_0th):
        self._use_0th = True if should_use_0th else False

    def _curve_fitting(self, phase, info):
        """
        Perform curve fitting for the phase.

        Args:
            phase (str): phase name
            info (dict[str]): start date, end date and population

        Returns:
            (tuple)
                (pandas.DataFrame): Result of curve fitting
                    Index: reset index
                    Columns:
                        - (phase name)_predicted: predicted value of Susceptible
                        - (phase_name)_actual: actual value of Susceptible
                        - (phase_name)_Recovered: Recovered
                (int): minimum value of R, which is the change point of the curve
        """
        start_date = info[self.START]
        end_date = info[self.END]
        population = info[self.N]
        trend = Trend(
            self.jhu_data, population, self.country, province=self.province,
            start_date=start_date, end_date=end_date
        )
        trend.analyse()
        df = trend.result()
        if trend.rmsle() > self.max_rmsle:
            df[f"{self.S}{self.P}"] = None
        # Get min value for vline
        r_value = int(df[self.R].min())
        # Rename the columns
        phase = self.INITIAL if phase == "0th" else phase
        df = df.rename({f"{self.S}{self.P}": f"{phase}{self.P}"}, axis=1)
        df = df.rename({f"{self.S}{self.A}": f"{phase}{self.A}"}, axis=1)
        df = df.rename({f"{self.R}": f"{phase}_{self.R}"}, axis=1)
        return (df, r_value)

    def show(self, show_figure=True, filename=None):
        """
        show the result as a figure and return a dictionary of phases.

        Args:
        @show_figure (bool): if True, show the result as a figure.
        @filename (str): filename of the figure, or None (display figure)

        Returns:
            (covsirphy.PhaseSeries)
        """
        # Create phase dictionary
        phase_series = self._create_phases()
        if not show_figure:
            return phase_series
        phase_dict = phase_series.to_dict()
        # Curve fitting
        nested = [
            self._curve_fitting(phase, info)
            for (phase, info) in phase_dict.items()
        ]
        df_list, vlines = zip(*nested)
        comp_df = pd.concat([self.sr_df, *df_list], axis=1)
        comp_df = comp_df.rename({self.S: f"{self.S}{self.A}"}, axis=1)
        comp_df = comp_df.apply(
            lambda x: pd.to_numeric(x, errors="coerce", downcast="integer"),
            axis=0
        )
        # Show figure
        pred_cols = [
            col for col in comp_df.columns if col.endswith(self.P)
        ]
        if len(pred_cols) == 1:
            title = f"{self.area}: S-R trend without change points"
        else:
            _list = self._change_dates[:]
            strings = [
                ", ".join(_list[i: i + 6]) for i in range(0, len(_list), 6)
            ]
            change_str = ",\n".join(strings)
            title = f"{self.area}: S-R trend changed on\n{change_str}"
        Trend.show_with_many(
            result_df=comp_df,
            predicted_cols=pred_cols,
            title=title,
            vlines=vlines[1:],
            filename=filename
        )
        return phase_series

    def get_dates(self, sr_df):
        """
        Get dates from the dataset.

        Args:
            sr_df (pandas.DataFrame)
                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual data of Susceptible

        Returns:
            (list[str]): list of dates, like 22Jan2020
        """
        dates = [
            date_obj.strftime(self.DATE_FORMAT) for date_obj in sr_df.index
        ]
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
        population_dict = {
            date: change_dict[date] if date in change_dict.keys(
            ) else population
            for date in dates
        }
        return population_dict

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
        start_dates, end_dates = self._phase_range(self._change_dates)
        pop_list = [self.pop_dict[date] for date in start_dates]
        phase_series = PhaseSeries(
            self.dates[0], self.dates[-1], self.population, use_0th=self._use_0th
        )
        phase_itr = enumerate(zip(start_dates, end_dates, pop_list))
        for (i, (start_date, end_date, population)) in phase_itr:
            phase_series.add(
                start_date=start_date,
                end_date=end_date,
                population=population
            )
        return phase_series
