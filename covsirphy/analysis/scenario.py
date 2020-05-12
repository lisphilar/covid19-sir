#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from covsirphy.util.plotting import line_plot
from covsirphy.selection.area import select_area
from covsirphy.ode.mbase import ModelBase
from covsirphy.analysis.estimator import Estimator
from covsirphy.analysis.predicter import Predicter
from covsirphy.analysis.sr_trend import Trend


class Scenario(object):
    """
    Class for scenario analysis.
    """
    # TODO: Refactoring with method separation
    SUFFIX_DICT = defaultdict(lambda: "th")
    SUFFIX_DICT.update({1: "st", 2: "nd", 3: "rd"})

    def __init__(self, ncov_df, name, is_country=True, date_format="%d%b%Y",
                 total_population=None, population_dict=None, **kwargs):
        """
        @ncov_df <pd.DataFrame>: the cleaned data
        @name <str>: name of the country/area
        @is_country <bool>: if True, @places arguments can be ommited
        @date_format <str>: string format of date
        @total_population <int>: total population of the area
        @population_dict <dict[country] = int>:
            dictionary of total population of each country
        @kwargs: keyword arguments of select_area() function
        """
        # Select area
        kwargs_key_set = set(kwargs.keys())
        key_used_set = set(["places", "excluded_places", "areas"])
        if is_country and not key_used_set.issubset(kwargs_key_set):
            record_df = select_area(
                ncov_df, places=[(name, None)], areas=None, excluded_places=None,
                **kwargs
            )
        else:
            record_df = select_area(ncov_df, **kwargs)
        # Total population
        try:
            total_population = population_dict[name]
        except (TypeError, KeyError):
            pass
        if not isinstance(total_population, int) \
                and total_population is not None:
            raise TypeError("total_population must be a integer!")
        self.total_population = total_population
        # Arrange dataset
        record_df = record_df.set_index("Date").resample("D").last()
        record_df = record_df.interpolate(method="linear")
        record_df = record_df.loc[:, [
            "Confirmed", "Infected", "Fatal", "Recovered"]]
        self.record_df = record_df.reset_index()
        self.name = name
        self.date_format = date_format
        self.phase_dict = dict()
        self.estimator_dict = dict()
        self.param_df = pd.DataFrame()
        self.future_phase_dict = dict()
        self.future_param_dict = dict()
        self.phases_without_vline = list()
        self.last_model = ModelBase

    def show_record(self):
        """
        Show the records.
        """
        line_plot(
            self.record_df.drop("Confirmed", axis=1).set_index("Date"),
            f"{self.name}: Cases over time",
            y_integer=True
        )
        return self.record_df

    def growth_factor(self, days_to_predict=0, until_stopping=False,
                      show_figure=True):
        """
        Return growth factor group and the history of growth factor values.
        @days_to_predict <int>: how many days to predict
        @until_stopping <bool>:
            if True and days_to_predict > 0,
            calculate growth factor values until the group will shift stopping
            after the last observation date
        @show_figure <bool>: if True, show line plot of cases over time
        """
        # Calculate growth factor
        if days_to_predict <= 0:
            # Value
            records = self.record_df.set_index("Date")["Confirmed"]
            growth = records.diff() / records.diff().shift(freq="D")
            growth = growth[:self.record_df["Date"].max()]
        else:
            records = self.predict(days=days_to_predict, show_figure=False)
            records = records["Confirmed"].fillna("ffill")
            growth = records.diff() / records.diff().shift()
        growth = growth.replace(np.inf, np.nan).fillna(1.0)
        growth = growth.rolling(7).mean().dropna().round(2)
        # Group
        if days_to_predict > 0 and until_stopping:
            last_observe_date = self.record_df["Date"].max().round("D")
            df = pd.DataFrame(
                {"Date": growth.index.round("D"), "Value": growth}
            )
            df = df.set_index("Date").resample("D").last().reset_index()
            df = df.loc[df["Date"] > (
                last_observe_date - timedelta(days=8)), :]
            date_df = df.loc[(df["Value"] < 1).rolling(7).sum() >= 7, "Date"]
            try:
                calc_date = date_df.reset_index(drop=True)[0]
            except IndexError:
                calc_date = df["Date"].max()
            group = "Stopping"
            growth = df.loc[df["Date"] <= calc_date, :]
            more_n = (growth["Value"] > 1)[::-1].cumprod().sum()
            less_n = (growth["Value"] < 1)[::-1].cumprod().sum()
            growth = growth.set_index("Date")
            date_str = calc_date.strftime("%d%b%Y")
            fig_title = f"{self.name}: Growth factor over time"
            fig_title += f"with prediction until {date_str}"
        else:
            more_n = (growth > 1)[::-1].cumprod().sum()
            less_n = (growth < 1)[::-1].cumprod().sum()
            calc_date = growth.index[-1]
            if more_n >= 7:
                group = "Outbreaking"
            else:
                group = "Stopping" if less_n >= 7 else "Crossroad"
            fig_title = f"{self.name}: Growth Factor over time"
        group_df = pd.DataFrame(
            {
                "Date": calc_date,
                "Group": group,
                "GF > 1 [straight days]": more_n,
                "GF < 1 [straight days]": less_n
            },
            index=[self.name]
        )
        # Growth factor over time
        if show_figure:
            growth.plot(title=fig_title, legend=False)
            plt.axhline(1.0, color="black", linestyle="--")
            plt.xlabel(None)
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            plt.axvline(today, color="black", linestyle="--")
            plt.show()
        return group_df

    def trend(self, n_points=0, total_population=None, **kwargs):
        """
        Perform trend analysis.
        @n_points <int>: the number of change points
        @total_population <int or list[int]>: total population
        @kwargs: the other keyword arguments of Trend.analyse() method
        """
        # Total population
        if total_population is None:
            if self.total_population is None:
                raise Exception("Please set total_population!")
            total_population = self.total_population
        else:
            if self.total_population is None:
                self.total_population = total_population
        # Trend analysis
        trend_obj = Trend(self.record_df, total_population, name=self.name)
        return trend_obj.analyse(n_points=n_points, **kwargs)

    def set_phase(self, start_dates, total_population=None):
        """
        Set phase for hyperparameter estimation.
        @start_dates <list[str]>: list of start dates of the phases
        @total_population <int or list[int]>:
            total population or list of total population
            - if None, use the value set by self.init() or self.trend()
        """
        if total_population is None:
            if self.total_population is None:
                raise Exception("Please set total_population!")
            total_population = self.total_population
        end_dates = [
            (
                datetime.strptime(s, self.date_format) - timedelta(days=1)).strftime(self.date_format)
            for s in start_dates[1:]
        ]
        end_dates.append(None)
        if isinstance(total_population, int):
            population_values = [
                total_population for _ in range(len(start_dates))]
        elif len(total_population) == len(start_dates):
            population_values = total_population[:]
        else:
            raise Exception(
                "start_date and population must have the same length!")
        self.phase_dict = {
            self._num2str(n): {"start_date": s, "end_date": e, "population": p}
            for (n, (s, e, p))
            in enumerate(zip(start_dates, end_dates, population_values), 1)
        }
        self.estimator_dict = dict()
        return pd.DataFrame.from_dict(self.phase_dict, orient="index").fillna("-")

    def estimate(self, model, n_trials=100, same_tau=True):
        """
        Perform hyperparameter estimation.
        @model <ModelBase>: math model
        @n_trials <int>: the number of trials
        @same_tau <bool>:
            whether apply the tau value of first phase to the following phases or not.
        """
        if not self.phase_dict:
            raise Exception("Please use Scenario.set_phase() at first.")
        tau = None
        est_start_time = datetime.now()
        for num in self.phase_dict.keys():
            print(f"Hyperparameter estimation of {num} phase.")
            target_dict = self.phase_dict[num]
            est_obj_count = 0
            while True:
                # Create estimator
                est_start_time_class = datetime.now()
                self.estimator_dict[num] = Estimator(
                    model, self.record_df, target_dict["population"],
                    name=self.name,
                    start_date=target_dict["start_date"],
                    end_date=target_dict["end_date"],
                    date_format=self.date_format,
                    tau=tau
                )
                print("\tEstimator was created.")
                # Run trials
                while True:
                    print(f"\t\t{n_trials} trials", end=" ")
                    est_start_time_run = datetime.now()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _ = self.estimator_dict[num].run(n_trials=n_trials)
                    minutes, seconds = divmod(
                        int((datetime.now() - est_start_time_run).total_seconds()), 60)
                    print(f"finished in {minutes} min {seconds} sec.")
                    # Check if estimated in (observed * 0.8, observed * 1.2)
                    compare_df = self.estimator_dict[num].compare_df()
                    targets = [
                        (compare_df[f"{val}_estimated"],
                         compare_df[f"{val}_observed"])
                        for val in model.MONOTONIC
                    ]
                    max_ok = [
                        obs.max() * 0.8 <= est.max() <= obs.max() * 1.2 for (est, obs) in targets]
                    monotonic_ok = [
                        target[0].is_monotonic for target in targets]
                    elapsed = (
                        datetime.now() - est_start_time_class
                    ).total_seconds()
                    if all(max_ok) or not all(monotonic_ok) or elapsed > 60 * 3:
                        break
                if all(monotonic_ok) and all(max_ok):
                    print("\tSuccessfully estimated.")
                    break
                vals = [val for (val, ok) in zip(
                    model.MONOTONIC, monotonic_ok) if not ok]
                try:
                    s = "\tEstimator will be replaced because estimated"
                    s += f" {vals[0]} is non-monotonic."
                    print(s)
                except IndexError:
                    est_obj_count += 0
                    if est_obj_count == 2:
                        s = "Estimater replacemnt was skipped"
                        s += " to break closed loop."
                        print(s)
                        break
                    else:
                        s = "\tEstimator will be replaced "
                        s += "because it is incapable of improvement."
                        print(s)
            tau = self.estimator_dict[num].param_dict["tau"]
        minutes, seconds = divmod(
            int((datetime.now() - est_start_time).total_seconds()), 60)
        print(f"Total: {minutes} min {seconds} sec.")
        self.show_parameters()
        self.last_model = model

    def accuracy_graph(self, phase_n=1):
        """
        Show observed - estimated graph.
        @phase_n <int>: phase number
        """
        phase_numbers = self.estimator_dict.keys()
        phase = self._num2str(phase_n)
        if phase not in phase_numbers:
            s = f"phase_n must be in {list(phase_numbers)[0]}"
            s += f" - {list(phase_numbers)[-1]}"
            raise KeyError(s)
        self.estimator_dict[phase].compare_graph()

    def _num2str(self, num):
        """
        Convert numbers to 1st, 2nd etc.
        @num <int>: number
        @return <str>
        """
        q, mod = divmod(num, 10)
        suffix = "th" if q == 1 else self.SUFFIX_DICT[mod]
        return f"{num}{suffix}"

    def show_parameters(self):
        """
        Show the parameter values.
        @retunr <pd.DataFrame>
        """
        # Phase information
        phase_dict = self.phase_dict.copy()
        phase_dict.update(self.future_phase_dict)
        df1 = pd.DataFrame.from_dict(phase_dict, orient="index")
        # Parameter information
        _dict = {
            k: estimator.param_dict
            for (k, estimator) in self.estimator_dict.items()
        }
        _future_dict = {
            k: {
                "tau": _dict["1st"]["tau"],
                **param_dict,
                "R0": self.last_model(**param_dict).calc_r0(),
                "score": None,
                **self.last_model(**param_dict).calc_days_dict(_dict["1st"]["tau"])
            }
            for (k, param_dict) in self.future_param_dict.items()
        }
        _dict.update(_future_dict)
        df2 = pd.DataFrame.from_dict(_dict, orient="index")
        # Rename R0 to Rt
        df2 = df2.rename({"R0": "Rt"}, axis=1)
        self.param_df = pd.concat([df1, df2], axis=1).fillna("-")
        return self.param_df

    def param(self, phase, param_name):
        """
        Return parameter value.
        @phase <str>: phase name, like 1st, 2nd..., or last
        @param_name <str>: name of parameter, like rho
        """
        if phase == "last":
            phase = list(self.phase_dict.items())[-1][0]
        try:
            estimator = self.estimator_dict[phase]
        except KeyError:
            s = "Please revise phase name(NOT iinclude future params)."
            s += " e.g. 1st, 2nd,... or last"
            raise KeyError(s)
        try:
            param_name = "R0" if param_name == "Rt" else param_name
            return estimator.param_dict[param_name]
        except KeyError:
            raise KeyError(
                "Please revise parameter name. e.g. rho, gamma, R0 or R0")

    def param_history(self, targets=None, box_plot=True, **kwargs):
        """
        Show the ratio to 1st parameters as a figure (bar plot).
        @targets <list[str] or str>: parameters to show (including Rt etc.)
        @box_plot <bool>: if True, box plot. if False, line plot.
        @kwargs: keword arguments of pd.DataFrame.plot or line_plot()
        """
        _ = self.show_parameters()
        targets = self.param_df.columns if targets is None else targets
        targets = [targets] if isinstance(targets, str) else targets
        if "R0" in targets:
            targets = [t.replace("R0", "Rt") for t in targets]
        df = self.param_df.loc[:, targets]
        df.index = self.param_df[["start_date", "end_date"]].apply(
            lambda x: f"{x[0]}-{x[1].replace('-', 'today')}",
            axis=1
        )
        df = df / df.iloc[0]
        if box_plot:
            df.plot.bar(title="Ratio to 1st parameters", **kwargs)
            plt.xticks(rotation=0)
            plt.legend(bbox_to_anchor=(1.02, 0),
                       loc="lower left", borderaxespad=0)
            plt.show()
        else:
            _df = df.reset_index(drop=True)
            _df.index = _df.index + 1
            line_plot(
                _df, title="Ratio to 1st parameters",
                xlabel="Phase", ylabel=str(), math_scale=False,
                **kwargs
            )

    def compare_estimated_numbers(self, phases=None):
        """
        Compare the number of confimred cases estimated
         with the parameters and show graph.
        @variable <str>: variable to compare
        @phases <list[str]>: phase to show (if None, all)
        """
        phases = list(self.phase_dict.keys()) if phases is None else phases
        # Observed
        df = pd.DataFrame(self.record_df.set_index("Date")["Confirmed"])
        # Estimated
        for (num, estimator) in self.estimator_dict.items():
            model, info_dict, param_dict = estimator.info()
            diff = (datetime.today() - info_dict["start_time"]).total_seconds()
            day_n = int(diff / 60 / 60 / 24 + 1)
            predicter = Predicter(**info_dict)
            predicter.add(model, end_day_n=day_n, **param_dict)
            # Calculate the number of confirmed cases
            new_df = predicter.restore_df().drop(
                "Susceptible", axis=1
            ).sum(axis=1)
            new_df = new_df.resample("D").last()
            df = pd.concat([df, new_df], axis=1)
        # Show graph
        df = df.fillna(0).astype(np.int64)
        df.columns = ["Observed"] + \
            [f"{phase}_param" for phase in self.phase_dict.keys()]
        df = df.loc[
            self.phase_dict["1st"]["start_date"]: self.record_df["Date"].max(), :]
        for col in df.columns[1:]:
            if col[:col.find("_")] not in phases:
                continue
            line_plot(
                df.replace(0, np.nan)[["Observed", col]],
                f"Confirmed cases over time: Actual and predicted with {col}",
                y_integer=True
            )

    def clear_future_param(self):
        """
        Clear the future parameters.
        """
        self.future_param_dict = dict()
        self.future_phase_dict = dict()
        last_phase = list(self.phase_dict.items())[-1][0]
        self.phase_dict[last_phase]["end_date"] = None
        return self

    def add_future_param(self, start_date, vline=True, **kwargs):
        """
        Add parameters of the future.
        @start_date <str>: the start date of the phase
        @vline <bool>:
            if True, add vertical line in the figure of predicted number of cases
        @kwargs: keword argument of parameters to change
        """
        yesterday_of_start = (pd.to_datetime(
            start_date) - timedelta(days=1)).strftime(self.date_format)
        # Last phase registered
        phase_dict = self.phase_dict.copy()
        phase_dict.update(self.future_phase_dict)
        last_phase = list(phase_dict.items())[-1][0]
        # Set the end date of the last phase
        if self.future_phase_dict:
            self.future_phase_dict[last_phase]["end_date"] = yesterday_of_start
        else:
            self.phase_dict[last_phase]["end_date"] = yesterday_of_start
        # Set the new phase
        try:
            param_dict = self.estimator_dict[last_phase].info()[2]
            population = self.phase_dict[last_phase]["population"]
        except KeyError:
            param_dict = self.future_param_dict[last_phase].copy()
            population = self.future_phase_dict[last_phase]["population"]
        param_dict.update(**kwargs)
        new_phase = self._num2str(len(phase_dict) + 1)
        self.future_param_dict[new_phase] = param_dict
        self.future_phase_dict[new_phase] = {
            "start_date": start_date,
            "end_date": None,
            "population": population
        }
        if not vline:
            self.phases_without_vline.append(new_phase)
        return pd.DataFrame.from_dict(self.future_param_dict, orient="index")

    def add_future_param_gradually(self, start_date, end_date, param, first, last):
        """
        Set the future parameters. The value will be gradually (log-scale) changed.
        @start_date <str>: the start date of change
        @end_date <str>: the end date of change
        @param <str>: parameter name
        @first <float>: parameter value of the start date
        @last <float>: parameter value of the end date
        """
        start = (pd.to_datetime(start_date) - timedelta(days=1)
                 ).strftime(self.date_format)
        dates = pd.date_range(start=start, end=end_date, freq="D")
        values = np.logspace(
            start=np.log10(first), stop=np.log10(last), num=len(dates), base=10.0
        )
        for (d, v) in zip(dates[1:], values[1:]):
            vline = True if d in dates[-2:] else False
            self.add_future_param(d.strftime(
                self.date_format), vline=vline, **{param: v})
        return pd.DataFrame.from_dict(self.future_param_dict, orient="index")

    def predict(self, days=1000, min_infected=1, show_figure=True):
        """
        Predict the future.
        @days <int or None>: how many days to predict from the last records date
        @min_infected <int>: if Infected < min_infected, the records will not be used
        @show_figure <bool>: if True, show line plot of cases over time
        """
        if not isinstance(days, int):
            raise TypeError("days must be integer!")
        # Create parameter dictionary
        predict_param_dict = {
            phase: self.estimator_dict[phase].info()[2]
            for (phase, _) in self.phase_dict.items()
        }
        predict_param_dict.update(self.future_param_dict)
        # Define phases
        model, info_dict, _ = self.estimator_dict["1st"].info()
        predicter = Predicter(**info_dict)
        phase_dict = self.phase_dict.copy()
        phase_dict.update(self.future_phase_dict)
        # Simulation with Predicter
        for (phase, date_dict) in phase_dict.items():
            start = pd.to_datetime(date_dict["start_date"])
            end = pd.to_datetime(date_dict["end_date"])
            if end is None:
                recorded_last_date = self.record_df.loc[
                    self.record_df.index[-1], "Date"
                ]
                day_n = days + \
                    int((recorded_last_date - start).total_seconds() / 60 / 60 / 24)
            elif start == end:
                day_n = 0
            else:
                day_n = int((end - start).total_seconds() / 60 / 60 / 24) + 1
            param_dict = predict_param_dict[phase].copy()
            vline = False if phase in self.phases_without_vline else True
            predicter.add(model, end_day_n=day_n,
                          count_from_last=True, vline=vline, **param_dict)
        # Restore
        df = predicter.restore_df(min_infected=min_infected)
        try:
            df["Confirmed"] = df["Infected"] + df["Recovered"] + df["Fatal"]
        except KeyError:
            pass
        # Graph: If max(other variables) < min(Susceptible), not show Susceptible
        if show_figure:
            without_s = df.drop("Susceptible", axis=1).sum(axis=1).max()
            drop_cols = [
                "Susceptible"] if without_s < df["Susceptible"].min() else None
            predicter.restore_graph(
                drop_cols=drop_cols, min_infected=min_infected, y_integer=True)
        return df
