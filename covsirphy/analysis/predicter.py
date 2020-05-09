#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from covsirphy.analysis.simulation import simulation
from covsirphy.util.plotting import line_plot


class Predicter(object):
    """
    Predict the future using models.
    """

    def __init__(self, name, total_population,
                 start_time, tau, initials, date_format="%d%b%Y"):
        """
        @name <str>: place name
        @total_population <int>: total population
        @start_time <datatime>: the start time
        @tau <int>: tau value (time step)
        @initials <list/tupple/np.array[float]>:
            initial values of the first model
        @date_format <str>: date format to display in figures
        """
        self.name = name
        self.total_population = total_population
        self.start_time = start_time.replace(
            hour=0, minute=0, second=0, microsecond=0)
        self.tau = tau
        self.date_format = date_format
        # Un-fixed
        self.last_time = start_time
        self.axvlines = list()
        self.initials = initials
        self.df = pd.DataFrame()
        self.title_list = list()
        self.reverse_f = lambda x: x
        self.model_names = list()

    def add(self, model, end_day_n=None, count_from_last=False, vline=True,
            **param_dict):
        """
        @model <ModelBase>: the epidemic model
        @end_day_n <int/None>:
            day number of the end date (0, 1, 2,...), or None (now)
            - if @count_from_last <bool> is True,
              start point will be the last date registered to Predicter
        @vline <bool>: if True, vertical line will be shown at the end date
        @**param_dict <dict>: keyword arguments of the model
        """
        # Validate day number, and calculate step number
        vline_yesterday = False
        if end_day_n == 0:
            end_day_n = 1
            vline_yesterday = True
        if end_day_n is None:
            end_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            if count_from_last:
                end_time = self.last_time + timedelta(days=end_day_n)
            else:
                end_time = self.start_time + timedelta(days=end_day_n)
        if end_time <= self.last_time:
            raise Exception(
                f"Model on {end_time.strftime(self.date_format)} has been registered!")
        step_n = int(
            (end_time - self.last_time).total_seconds() / 60 / self.tau) + 1
        self.last_time = end_time.replace(
            hour=0, minute=0, second=0, microsecond=0)
        # Perform simulation
        new_df = simulation(model, self.initials, step_n=step_n, **param_dict)
        new_df["t"] = new_df["t"] + len(self.df)
        self.df = pd.concat([self.df, new_df.iloc[1:, :]], axis=0).fillna(0)
        self.initials = new_df.set_index("t").iloc[-1, :]
        # For title
        self.model_names.append(model.NAME)
        if vline:
            vline_date = end_time.replace(
                hour=0, minute=0, second=0, microsecond=0)
            if vline_yesterday:
                vline_date -= timedelta(days=1)
            self.axvlines.append(vline_date)
            r0 = model(**param_dict).calc_r0()
            if len(self.axvlines) == 1:
                self.title_list.append(
                    f"{model.NAME}(R0={r0}, -{vline_date.strftime(self.date_format)})")
            else:
                if model.NAME == self.model_names[-2]:
                    self.title_list.append(
                        f"({r0}, -{vline_date.strftime(self.date_format)})")
                else:
                    self.title_list.append(
                        f"{model.NAME}({r0}, -{end_time.strftime(self.date_format)})")
        # Update reverse function (X, Y,.. to Susceptible, Infected,...)
        self.reverse_f = model.calc_variables_reverse
        return self

    def restore_df(self, min_infected=1):
        """
        Return the dimentional simulated data.
        @min_infected <int>: if Infected < min_infected, the records will not be used
        @return <pd.DataFrame>
        """
        df = self.df.copy()
        df["Time"] = self.start_time + \
            df["t"].apply(lambda x: timedelta(minutes=x * self.tau))
        df = df.drop("t", axis=1).set_index("Time") * self.total_population
        df = df.astype(np.int64)
        upper_cols = [n.upper() for n in df.columns]
        df.columns = upper_cols
        df = self.reverse_f(df, self.total_population).drop(upper_cols, axis=1)
        df = df.loc[df["Infected"] >= min_infected, :]
        return df

    def restore_graph(self, drop_cols=None, min_infected=1, **kwargs):
        """
        Show the dimentional simulate data as a figure.
        @drop_cols <list[str]>: the columns not to be shown
        @min_infected <int>: if Infected < min_infected, the records will not be used
        @kwargs: keyword arguments of line_plot() function
        """
        df = self.restore_df(min_infected=min_infected)
        if drop_cols is not None:
            df = df.drop(drop_cols, axis=1)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        axvlines = [
            today, *self.axvlines] if len(self.axvlines) == 1 else self.axvlines[:]
        line_plot(
            df,
            title=f"{self.name}: {', '.join(self.title_list)}",
            v=axvlines[:-1],
            h=self.total_population,
            **kwargs
        )
