#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from covsirphy.trend.trend_plot import trend_plot
from covsirphy.util.argument import find_args
from covsirphy.util.error import NAFoundError, UnExecutedError
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.ode.ode_handler import ODEHandler
from covsirphy.trend.trend_detector import TrendDetector


class Dynamics(Term):
    """Class to calculate parameter values of phase-dependent SIR-derived ODE model.

    Args:
        model (covsirphy.ModelBase): ODE model
        data (pandas.DataFrame): rw data to calculate initial values of the model and ODE parameter estimation
            Index
                reset index
            Columns
                - Date (pandas.Timestamp): Observation date
                - Susceptible (int): the number of susceptible cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - (numpy.float64): ODE parameter values defined with model.PARAMETERS
        tau (int or None): tau value [min] or None (to be estimated with covsirphy.Dynamics.estimate())
        area (str): area name (used in the figure title)
        **kwargs: keyword arguments of covsirphy.Dynamics.timepoints()

    Raises:
        NAFoundError: records at the first date has NAs

    Note:
        "Phase" means a sequential dates in which the parameters of SIR-derived models are fixed.
        "Change points" means the dates when trend was changed.
        "Change points" is the same as the start dates of phases except for the 0th phase.
    """
    _PH = "Phase_ID"
    _SIFR = [Term.S, Term.CI, Term.F, Term.R]

    def __init__(self, model, data, tau=None, area="Selected area", **kwargs):
        self._model = self._ensure_subclass(model, ModelBase, name="model")
        self._area = str(area)
        self._tau = self._ensure_tau(tau, accept_none=True)
        # Records and date
        self._ensure_dataframe(data, columns=[self.DATE, *self._SIFR, *self._model.PARAMETERS], name="data")
        self._data_df = data.set_index(self.DATE)
        self._first, self._today, self._last = None, None, None
        self.timepoints(**kwargs)

    def timepoints(self, first_date=None, today=None, last_date=None):
        """
        Set the range of data and reference date to determine past/future of phases.

        Args:
            first_date (str or None): the first date to focus on or None (min date of the dataset)
            today (str or None): reference date to determine whether a phase is a past phase or a future phase
            last_date (str or None): the first date to focus on or None (max date of the dataset)

        Raises:
            NAFoundError: records at the first date has NAs

        Note:
            When @today is None, the reference date will be the same as @last_date (or max date).

        Note:
            When executed, ODE parameters and phases will be reset.
        """
        first = self._ensure_date(first_date, name="first_date", default=self._data_df.index.min())
        last = self._ensure_date(last_date, name="last_date", default=self._data_df.index.max())
        self._ensure_date_order(first, last, name="last_date")
        _today = self._ensure_date(today, name="today", default=last)
        self._ensure_date_order(first, _today, name="today")
        self._ensure_date_order(_today, last, name="last_date")
        df = pd.DataFrame(
            {self._PH: 0, self.ODE: self._model.NAME, self.TAU: self._tau},
            index=pd.date_range(start=first, end=last, freq="D"),
            columns=[self._PH, *self._SIFR, self.ODE, *self._model.PARAMETERS, self.TAU])
        df.update(self._data_df)
        df.index.name = self.DATE
        if df.loc[df.index[0], self._SIFR].isna().any():
            raise NAFoundError(
                name=f"data on {first_date}", value=df.loc[df.index[0], self._SIFR],
                details="They will be used to calculate initial values of the model.")
        self._first, self._today, self._last = first, _today, last
        self._all_df = df.copy()

    @classmethod
    def from_sample(cls, model, **kwargs):
        """Create covsirphy.Dynamics instance with sample data.

        Args:
            model (covsirphy.ModelBase): ODE model
            **kwargs: keyword arguments of covsirphy.Dynamics except for @model and @data
                first_date (str or pandas.Timestamp): the first date of simulation, default is today when executed
                last_date (str or pandas.Timestamp): the last date of simulation, default is today when executed + 180 days
                tau (int or None): tau value [min] or None, default is 1440
                area (str): area name, default is @model.NAME
        """
        today = datetime.now()
        cls_dict = {
            "first_date": today,
            "today": None,
            "last_date": today + timedelta(days=180),
            "tau": 1440,
            "area": model.NAME,
        }
        cls_dict.update(kwargs)
        df = model.convert_reverse(
            pd.DataFrame([model.EXAMPLE["y0_dict"]]), start=cls._ensure_date(cls_dict["first_date"]), tau=1440)
        param_df = pd.DataFrame([model.EXAMPLE["param_dict"]])
        df = pd.concat([df, param_df], axis=1)
        return cls(model=model, data=df, **cls_dict)

    def segment(self, points=None, start_date=None, end_date=None, **kwargs):
        """Perform time-series segmentation manually or with S-R trend analysis.

        Args:
            points (list[str] or None): dates of change points or None (use S-R trend analysis)
            start_date (str or pandas.Timestamp or None): start date of the records to focus on or None (covsirphy.Dynamics(first_date))
            end_date (str or pandas.Timestamp or None): end date of the records to focus on or None (covsirphy.Dynamics(last_date))
            kwargs: keyword arguments of S-R trend analysis
                - algo (str): detection algorithms and models, default is "Binseg-normal", refer to covsirphy.TrendDector.sr()
                - min_size (int): minimum value of phase length [days], over 2, default value is 7
                - the other arguments of algorithm classes (ruptures.Pelt, .Binseg, BottomUp)

        Raises:
            NAFoundError: records from the start date to end date has NAs when @points is None

        Returns:
            covsirphy.Dynamics: self

        Note:
            @points can include the start date, but unnecessary.

        Note:
            Tomorrow (tomorrow of covsirphy.Dynamics.timepoints(today)) will be regarded as a change point automatically.
        """
        start = self._ensure_date(start_date, name="start_date", default=self._first)
        end = self._ensure_date(end_date, name="end_date", default=self._last)
        df = self._all_df.copy()
        # Find change points manually or with S-R trend analysis
        if points is not None:
            starts = [self._ensure_date(date, name="a change point") for date in points] + [start]
        else:
            detector = self._detector(data=self._all_df, start=start, end=end, **kwargs)
            starts = [self._ensure_date(date) for date in detector.dates()[0]]
        # Add tomorrow
        tomorrow = self._ensure_date(self.tomorrow(self._today))
        start_set = set(starts) | ({tomorrow} if start < self._today < end else set())
        # Set phases
        for point in sorted(start_set):
            self._ensure_date_order(start, point, name="a change point")
            self._ensure_date_order(point, end, name="end_date")
            df.loc[point: end, self._PH] = max(df[self._PH].abs()) + 1
        self._all_df = df.copy()
        return self

    def sr(self, start_date=None, end_date=None, metric="MSE", simulated=False, **kwargs):
        """Perform S-R trend analysis and show S-R plane, not changing phase settings.

        Args:
            start_date (str or pandas.Timestamp or None): start date of the records to focus on or None (covsirphy.Dynamics(first_date))
            end_date (str or pandas.Timestamp or None): end date of the records to focus on or None (covsirphy.Dynamics(last_date))
            metric (str): metrics name of evaluation, default is "MSE", refer to covsirphy.Evaluator.score()
            simulated (bool): whether use simulated number of cases for analysis or not
            kwargs: keyword arguments of S-R trend analysis
                - algo (str): detection algorithms and models, default is "Binseg-normal", refer to covsirphy.TrendDector.sr()
                - min_size (int): minimum value of phase length [days], over 2, default value is 7
                - the other arguments of algorithm classes (ruptures.Pelt, .Binseg, BottomUp)
                - keyword arguments of covsirphy.TrendDetector.show() and covsirphy.trend_plot()

        Raises:
            NAFoundError: records from the start date to end date has NAs

        Returns:
            pandas.DataFrame: as-is TrendDetector.summary()
        """
        start = self._ensure_date(start_date, name="start_date", default=self._first)
        end = self._ensure_date(end_date, name="end_date", default=self._last)
        # Data for S-R analysis
        df = self.simulate(ffill=True).set_index(self.DATE) if simulated else self._all_df.copy()
        # Perform S-R trend analysis
        detector = self._detector(data=df, start=start, end=end, **kwargs)
        # Show results
        detector.show(**find_args([TrendDetector.show, trend_plot], **kwargs))
        return detector.summary(metric=metric)

    def _detector(self, data, start, end, **kwargs):
        """Create covsirphy.TrendDetector instance.

        Args:
            data (pandas.DataFrame): data for S-R trend analysis
                Index
                    Date (pandas.Timestamp): Observation date
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Recovered (int): the number of recovered cases
            start (pandas.Timestamp): start date of the records to focus on
            end (pandas.Timestamp): end date of the records to focus on
            kwargs: keyword arguments of S-R trend analysis
                - algo (str): detection algorithms and models, default is "Binseg-normal", refer to covsirphy.TrendDector.sr()
                - min_size (int): minimum value of phase length [days], over 2, default value is 7
                - the other arguments of algorithm classes (ruptures.Pelt, .Binseg, BottomUp)

        Raises:
            NAFoundError: records from the start date to end date has NAs

        Returns:
            covsirphy.TrendDetector
        """
        df = data.loc[start: end, [self.S, self.R]].astype(np.int64).reset_index()
        if df.isna().any().any():
            raise NAFoundError(
                name=f"Records from {start} to {end}", value=None,
                details="They are required to perform S-R trend analysis correctly.")
        algo = kwargs.get("algo", "Binseg-normal")
        min_size = kwargs.get("min_size", 7)
        detector = TrendDetector(data=df, area=self._area, min_size=min_size)
        return detector.sr(algo=algo, **kwargs)

    def estimate(self, metric="RMSLE", n_jobs=-1, **kwargs):
        """Estimate ODE parameter values and tau value of phases.

        Args:
            metric (str): metric name for estimation
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            kwargs: keyword arguments of ODEHandler.estimate_tau() and .estimate_param()

        Raises:
            UnExecutedError: no phases are filled with records

        Returns:
            covsirphy.Dynamics: self

        Note:
            Records except for NAs until today will be used for ODE parameter estimation.
        """
        data_df = self._all_df.loc[:self._today, [self._PH, *self._SIFR]]
        data_df = data_df.dropna(axis=0, how="any").reset_index()
        data_df[self._PH], _ = data_df[self._PH].factorize()
        start_dates = data_df.groupby(self._PH).first()[self.DATE].sort_values()
        end_dates = data_df.groupby(self._PH).last()[self.DATE].sort_values()
        handler = ODEHandler(model=self._model, first_date=self._first, tau=self._tau, metric=metric, n_jobs=n_jobs)
        for start, end in zip(start_dates, end_dates):
            if end < start + timedelta(days=2):
                continue
            if data_df[data_df[self.DATE] == end].isna().any().any():
                continue
            y0_series = self._model.convert(data_df.loc[data_df[self.DATE] >= start], tau=None).iloc[0]
            _ = handler.add(end, y0_dict=y0_series.to_dict())
        # Estimate tau (if necessary) and ODE parameer values
        try:
            self._tau, est_dict = handler.estimate(data_df, **kwargs)
        except UnExecutedError:
            raise UnExecutedError("covsirphy.Dynamics.update()", details="No phases are filled with records.") from None
        # Register phase information to self
        df = pd.DataFrame.from_dict(est_dict, orient="index")
        df[self.DATE] = df[[self.START, self.END]].apply(lambda x: pd.date_range(x[0], x[1]), axis=1)
        df = df.explode(self.DATE).drop([self.START, self.END], axis=1).set_index(self.DATE)
        df[self.TAU] = self._tau
        self._all_df = self._all_df.combine_first(df)
        return self

    def update(self, start_date, end_date, variable, value):
        """Update data and parameter values.

        Args:
            start_date (str or pandas.Timestamp): start date of the records to update
            end_date (str or pandas.Timestamp): end date of the records to update
            variable (str): variable name, "Susceptible", "Infected", "Fatal", "Recovered", ODE parameters
            value (int or float): value to update the variable with

        Returns:
            covsirphy.Dynamics: self
        """
        start = self._ensure_date(start_date, name="start_date")
        end = self._ensure_date(end_date, name="end_date")
        parameters = self._model.PARAMETERS[:]
        self._ensure_selectable(variable, candidates=[*self._SIFR, *parameters], name="variable")
        if variable in self._SIFR:
            new_df = pd.DataFrame(
                {variable: self._ensure_natural_int(value, name=f"value of {variable}", include_zero=True)},
                index=pd.date_range(start=start, end=end, freq="D"))
        else:
            new_df = self._all_df.loc[self._ensure_date(start_date): self._ensure_date(end_date), parameters]
            new_df[variable] = self._ensure_float(value, name=f"value of {variable}", value_range=(0, 1))
            new_df[self.RT] = new_df.apply(lambda x: self._model(population=1, **x.to_dict()).calc_r0(), axis=1)
            if self._tau is not None:
                days_df = new_df.drop(self.RT, axis=1).apply(
                    lambda x: self._model(population=1, **x.to_dict()).calc_days_dict(tau=self._tau), axis=1)
                new_df = pd.concat([new_df, days_df], axis=1)
        self._all_df.update(new_df)
        return self

    def get(self, date, variable):
        """Update data and parameter values.

        Args:
            date (str or pandas.Timestamp): date of the record to get
            variable (str): variable name, "Susceptible", "Infected", "Fatal", "Recovered", ODE parameters

        Returns:
            covsirphy.Dynamics: self
        """
        self._ensure_selectable(variable, candidates=[*self._SIFR, *self._model.PARAMETERS], name="variable")
        return self._all_df.loc[self._ensure_date(date, name="date"), variable]

    def simulate(self, ffill=True, model_specific=False):
        """Perform simulation with the multi-phased ODE model.

        Args:
            ffill (bool): whether propagate last valid ODE parameter values forward to next valid or not
            model_specific (bool): whether convert S, I, F, R to model-specific variables or not

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - if @model_specific is False:
                        - Susceptible (int): the number of susceptible cases
                        - Infected (int): the number of currently infected cases
                        - Fatal (int): the number of fatal cases
                        - Recovered (int): the number of recovered cases
                    - if @model_specific is True, variables defined by model.VARIABLES of covsirphy.Dynamics(model)
        """
        all_df = self._all_df.copy()
        date_df = all_df.loc[:, [self._PH]].reset_index()
        start_dates = date_df.groupby(self._PH).first()[self.DATE].sort_values()
        end_dates = date_df.groupby(self._PH).last()[self.DATE].sort_values()
        param_df = all_df.loc[:, self._model.PARAMETERS]
        if ffill:
            param_df.ffill(inplace=True)
        # Simulation
        handler = ODEHandler(model=self._model, first_date=self._first, tau=self._tau)
        for start, end in zip(start_dates, end_dates):
            param_dict = param_df.loc[start].to_dict()
            if None in param_dict.values():
                for k in [k for k, v in param_dict.items() if v is None]:
                    raise UnExecutedError(
                        name=f"covsirphy.Dynamics.update(start_date='{start}', end_date='{end}', variable='{k}', value=<expected value>)")
            ph_df = all_df.loc[start:, self._SIFR].reset_index()
            if ph_df.iloc[0].isna().any():
                y0_dict = None
            else:
                y0_dict = self._model.convert(ph_df, tau=None).iloc[0].to_dict()
            _ = handler.add(end, param_dict=param_dict, y0_dict=y0_dict)
        sim_df = handler.simulate()
        if model_specific:
            return self._model.convert(sim_df, tau=self._tau).convert_dtypes()
        return sim_df.convert_dtypes()

    def track(self, simulated=False, ffill=True):
        """Track data with all dates.

        Args
            simulated (bool): whether use simulated number of cases for analysis or not
            ffill (bool): whether propagate last valid ODE parameter values forward to next valid or not

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases
                    - If available,
                        - Rt (float): phase-dependent reproduction number
                        - (str, float): estimated parameter values, including rho
                        - tau (int): tau value [min]
                        - (int or float): day parameters, including 1/beta [days]
                        - {metric}: score with the estimated parameter values
                        - Trials (int): the number of trials
                        - Runtime (str): runtime of optimization
        """
        df = self._all_df.reset_index()
        parameters = self._model.PARAMETERS[:]
        if simulated:
            sim_df = self.simulate(ffill=ffill)
            df = df.drop(self._SIFR, axis=1).join(sim_df, how="left")
        if ffill:
            df.loc[:, parameters] = df.loc[:, parameters].ffill()
        df[self.C] = df[[self.CI, self.F, self.R]].sum()
        # Reproduction number
        if self.RT not in df:
            df[self.RT] = None
        df.loc[df[self.RT].isna(), self.RT] = df.loc[df[self.RT].isna(), parameters].apply(
            lambda x: self._model(population=1, **x.to_dict()).calc_r0(), axis=1)
        # Days-parameters
        if self._tau is not None:
            days_df = df[parameters].apply(
                lambda x: self._model(population=1, **x.to_dict()).calc_days_dict(tau=self._tau),
                axis=1, result_type="expand")
            df = df.combine_first(days_df)
        # Set the order of columns
        fixed_cols = [
            self.DATE, self.C, self.CI, self.F, self.R, self.S, self.RT, *parameters, self.TAU, *self._model.DAY_PARAMETERS]
        others = [col for col in df.columns if col not in set(fixed_cols)]
        return df.reindex(columns=[*fixed_cols, *others]).dropna(how="all", axis=1).convert_dtypes()

    def summary(self, **kwargs):
        """Summarize phase information.

        Args
            **kwargs: keyword arguments of covsirphy.Dynamics.track()

        Returns:
            pandas.DataFrame
                Index
                    Phase (str): phase names, 0th, 1st,...
                Columns
                    - Start (pandas.Timestamp): start date of the phase
                    - End (pandas.Timestamp): end date of the phase
                    - Population (numpy.Int64): population value of the start date
                    - ODE (str): ODE model names
                    - (float): estimated parameter values, including rho
                    - tau (int): tau value [min]
                    - If available,
                        - Rt (float): phase-dependent reproduction number
                        - (int or float): day parameters, including 1/beta [days]
                        - {metric}: score with the estimated parameter values
                        - Trials (int): the number of trials
                        - Runtime (str): runtime of optimization
        """
        df = self.track(**kwargs)
        # Show only defined phases
        df = df.loc[df[self._PH] > 0]
        # Set index with phases
        df[self._PH], _ = df[self._PH].factorize()
        first_df = df.groupby(self._PH).first()
        df = first_df.join(df.groupby(self._PH).last(), rsuffix="_last")
        df = df.rename(columns={self.DATE: self.START, f"{self.DATE}_last": self.END})
        df = df.loc[:, [col for col in df.columns if "_last" not in col]]
        df.index.name = self.PHASE
        df.index = [self.num2str(num) for num in df.index]
        # Calculate population values
        df[self.N] = df[self._SIFR].sum(axis=1).replace(0.0, np.nan).ffill()
        # Set the order of columns
        fixed_cols = [
            self.START, self.END, self.N, self.RT, *self._model.PARAMETERS, self.TAU, *self._model.DAY_PARAMETERS]
        others = [col for col in df.columns if col not in set(fixed_cols) | set(self._SIFR)]
        return df.reindex(columns=[*fixed_cols, *others]).dropna(how="all", axis=1).convert_dtypes()
