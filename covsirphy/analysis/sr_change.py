#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from covsirphy.cleaning import Term, JHUData
from covsirphy.phase.trend import Trend
from covsirphy.analysis.phase_series import PhaseSeries
from covsirphy.analysis.example_data import ExampleData


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
    """

    def __init__(self, jhu_data, population, country, province=None,
                 population_change_dict=None):
        # Dataset
        if isinstance(jhu_data, pd.DataFrame):
            warnings.warn(
                "Please use instance of JHUData as the first argument of ChangeFinder class.",
                DeprecationWarning,
                stacklevel=2
            )
            df = jhu_data.copy()
            df = self.validate_dataframe(
                df, name="jhu_data", columns=self.COLUMNS
            )
            jhu_data = ExampleData(df)
        self.jhu_data = self.validate_instance(
            jhu_data, JHUData, name="jhu_data"
        )
        # Dataset for analysis
        # Index: Date(pd.TimeStamp): Observation date
        # Columns: Recovered (int), Susceptible_actual (int)
        sr_df = jhu_data.subset(
            country=country, province=province, population=population
        )
        sr_df = sr_df.set_index(self.DATE)
        sr_df = sr_df.loc[:, [self.R, self.S]]
        sr_df = sr_df.rename({self.S: f"{self.S}{self.A}"}, axis=1)
        self.sr_df = sr_df.copy()
        self.dates = self.get_dates(self.sr_df)
        # Arguments
        self.country = country
        self.province = province
        self.area = JHUData.area_name(country, province)
        self.pop_dict = self._read_population_data(
            self.dates, population, population_change_dict
        )
        self.population = population
        # Setting for optimization
        self.change_dates = list()

    def run(self):
        """
        Run optimization and find change points.
        Returns:
            self
        """
        delta_days = timedelta(days=2)
        first_obj = self.to_date_obj(self.dates[0])
        last_obj = self.to_date_obj(self.dates[-1])
        if first_obj >= last_obj - delta_days:
            raise ValueError("More than 2 records must be included.")
        # Convert the dataset, index: Recovered, column: Susceptible
        sr_df = self.sr_df.rename({f"{self.S}{self.A}": self.S}, axis=1)
        sr_df[self.S] = np.log10(sr_df[self.S])
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
        algorithm = rpt.Pelt(model="rbf", jump=2, min_size=6)
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
        # Only use dates when the previous phase has more than 3 days
        effective_list = [first_obj]
        for found in found_list:
            if effective_list[-1] + delta_days < found:
                effective_list.append(found)
        # The last change date must be under the last date of records - 2 days
        if effective_list[-1] >= last_obj - delta_days:
            effective_list = effective_list[:-1]
        # Set change points
        self.change_dates = [
            date.strftime("%d%b%Y") for date in effective_list[1:]
        ]
        return self

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
        phase_dict = phase_series.to_dict()
        # Curve fitting
        nested = [
            self._curve_fitting(phase, info)
            for (phase, info) in phase_dict.items()
        ]
        df_list, vlines = zip(*nested)
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
            _list = self.change_dates[:]
            strings = [
                ", ".join(_list[i: i + 6]) for i in range(0, len(_list), 6)
            ]
            change_str = ",\n".join(strings)
            title = f"{self.area}: S-R trend changed on\n{change_str}"
        Trend.show_with_many(
            result_df=comp_df,
            predicted_cols=pred_cols,
            title=title,
            vlines=vlines[2:],
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
