#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from datetime import timedelta
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate, NAFoundError, UnExecutedError, UnExpectedValueError
from covsirphy.util.argument import find_args
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated.trend_plot import trend_plot
from covsirphy._deprecated.trend_detector import TrendDetector
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.ode_handler import ODEHandler


class PhaseTracker(Term):
    """
    Track phase information of one scenario.

    Args:
        data (pandas.DataFrame):
            Index
                reset index
            Columns
                - Date (pandas.Timestamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - Susceptible (int): the number of susceptible cases
        today (str or pandas.Timestamp): reference date to determine whether a phase is a past phase or not
        area (str): area name, like Japan/Tokyo

    Note:
        (Internally) ID=0 means not registered, ID < 0 means disabled, IDs (>0) are active phase ID.
    """

    @deprecate(old="PhaseTracker", version="2.24.0-xi")
    def __init__(self, data, today, area):
        self._ensure_dataframe(data, name="data", columns=self.SUB_COLUMNS)
        self._today = Validator(today, "today").date()
        self._area = str(area)
        # Tracker of phase information: index=Date, records of C/I/F/R/S, phase ID (0: not defined)
        self._track_df = data.set_index(self.DATE)
        self._track_df[self.ID] = 0
        # For simulation (determined in self.estimate())
        self._model = None
        self._tau = None

    def __len__(self):
        """
        int: the number of registered phases, including deactivated phases
        """
        df = self._track_df.copy()
        df = df.loc[df[self.ID] != 0]
        return df[self.ID].nunique()

    @staticmethod
    def _ensure_dataframe(target, name="df", time_index=False, columns=None, empty_ok=True):
        """
        Ensure the dataframe has the columns.

        Args:
            target (pandas.DataFrame): the dataframe to ensure
            name (str): argument name of the dataframe
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have
            empty_ok (bool): whether give permission to empty dataframe or not

        Returns:
            pandas.DataFrame:
                Index
                    as-is
                Columns:
                    columns specified with @columns or all columns of @target (when @columns is None)
        """
        if not isinstance(target, pd.DataFrame):
            raise TypeError(f"@{name} must be a instance of (pandas.DataFrame).")
        df = target.copy()
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if not empty_ok and target.empty:
            raise ValueError(f"@{name} must not be a empty dataframe.")
        if columns is None:
            return df
        if not set(columns).issubset(df.columns):
            cols_str = ", ".join(col for col in columns if col not in df.columns)
            included = ", ".join(df.columns.tolist())
            s1 = f"Expected columns were not included in {name} with {included}."
            raise KeyError(f"{s1} {cols_str} must be included.")
        return df.loc[:, columns]

    @staticmethod
    def _ensure_list(target, candidates=None, name="target"):
        """
        Ensure the target is a sub-list of the candidates.

        Args:
            target (list[object]): target to ensure
            candidates (list[object] or None): list of candidates, if we have
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        if not isinstance(target, (list, tuple)):
            raise TypeError(f"@{name} must be a list or tuple, but {type(target)} was applied.")
        if candidates is None:
            return target
        # Check the target is a sub-list of candidates
        try:
            strings = [str(candidate) for candidate in candidates]
        except TypeError:
            raise TypeError(f"@candidates must be a list, but {candidates} was applied.") from None
        ok_list = [element in candidates for element in target]
        if all(ok_list):
            return target
        candidate_str = ", ".join(strings)
        raise KeyError(f"@{name} must be a sub-list of [{candidate_str}], but {target} was applied.") from None

    def define_phase(self, start, end):
        """
        Define an active phase with the series of dates.

        Args:
            start (str or pandas.Timestamp): start date of the new phase
            end (str or pandas.Timestamp): end date of the new phase

        Returns:
            covsirphy.PhaseTracker: self

        Note:
            When today is in the range of (start, end), a past phase and a future phase will be created.
        """
        start = Validator(start, name="start").date()
        end = Validator(end, name="end").date()
        track_df = self._track_df.copy()
        # Start date must be over the first date of records
        self._ensure_date_order(track_df.index.min(), start, name="start")
        # Add a past phase (start -> min(end, today))
        if start <= self._today:
            track_df.loc[start:min(self._today, end), self.ID] = track_df[self.ID].abs().max() + 1
        # Add a future phase (tomorrow -> end)
        if self._today < end:
            phase_start = max(self._today + timedelta(days=1), start)
            df = pd.DataFrame(index=pd.date_range(phase_start, end), columns=track_df.columns)
            df.index.name = self.DATE
            df[self.ID] = track_df[self.ID].abs().max() + 1
            track_df = pd.concat([track_df, df], axis=0).resample("D").last()
        # Fill in skipped dates
        series = track_df[self.ID].copy()
        track_df.loc[(series.index <= end) & (series == 0), self.ID] = series.abs().max() + 1
        # Update self
        self._track_df = track_df.copy()
        return self

    @classmethod
    def _ensure_date_order(cls, previous_date, following_date, name="following_date"):
        """
        Ensure that the order of dates.

        Args:
            previous_date (str or pandas.Timestamp): previous date
            following_date (str or pandas.Timestamp): following date
            name (str): name of @following_date

        Raises:
            ValueError: @previous_date > @following_date
        """
        previous_date = cls._ensure_date(previous_date)
        following_date = cls._ensure_date(following_date)
        p_str = previous_date.strftime(cls.DATE_FORMAT)
        f_str = following_date.strftime(cls.DATE_FORMAT)
        if previous_date <= following_date:
            return None
        raise ValueError(f"@{name} must be the same as/over {p_str}, but {f_str} was applied.")

    def deactivate(self, start, end):
        """
        The status (enabled/disabled) of the phases between start and end date will be reversed.

        Args:
            start (str or pandas.Timestamp): start date of the phase to remove
            end (str or pandas.Timestamp): end date of the phase to remove

        Returns:
            covsirphy.PhaseTracker: self
        """
        start = Validator(start, "start").date()
        end = Validator(end, "end").date()
        self._track_df.loc[start:end, self.ID] *= -1
        return self

    def remove_phase(self, start, end):
        """
        Remove phase information from the date range.

        Args:
            start (str or pandas.Timestamp): start date of the phase to remove
            end (str or pandas.Timestamp): end date of the phase to remove

        Returns:
            covsirphy.PhaseTracker: self
        """
        start = Validator(start, "start").date()
        end = Validator(end, "end").date()
        self._track_df.loc[start:end, self.ID] = 0
        return self

    def track(self):
        """
        Track data with all dates.

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
                        - (int or float): day parameters, including 1/beta [days]
                        - {metric}: score with the estimated parameter values
                        - Trials (int): the number of trials
                        - Runtime (str): runtime of optimization

        Note:
            C/I/F/R/S/I is simulated values if parameter values are available.
        """
        df = self._track_df.copy()
        df = df.loc[df[self.ID] != 0]
        # Use simulated data for tracking
        with contextlib.suppress(UnExecutedError):
            df.update(self.simulate().set_index(self.DATE))
        return df.drop(self.ID, axis=1).reset_index()

    def summary(self):
        """
        Summarize phase information.

        Returns:
            pandas.DataFrame
                Index
                    str: phase names
                Columns
                    - Type: 'Past' or 'Future'
                    - Start: start date of the phase
                    - End: end date of the phase
                    - Population: population value of the start date
                    - If available,
                        - ODE (str): ODE model names
                        - Rt (float): phase-dependent reproduction number
                        - (str, float): estimated parameter values, including rho
                        - tau (int): tau value [min]
                        - (int or float): day parameters, including 1/beta [days]
                        - {metric}: score with the estimated parameter values
                        - Trials (int): the number of trials
                        - Runtime (str): runtime of optimization
        """
        # Remove un-registered phase
        track_df = self._track_df.reset_index()
        track_df["ID_ordered"], _ = track_df[self.ID].factorize()
        track_df = track_df.loc[track_df[self.ID] > 0].drop(self.ID, axis=1)
        # -> index=phase names, columns=Start/variables,.../End
        first_df = track_df.groupby("ID_ordered").first()
        df = first_df.join(track_df.groupby("ID_ordered").last(), rsuffix="_last")
        df = df.rename(columns={self.DATE: self.START, f"{self.DATE}_last": self.END})
        df.index.name = None
        df.index = [self.num2str(num) for num in df.index]
        df = df.loc[:, [col for col in df.columns if "_last" not in col]]
        # Calculate phase types: Past or Future
        df[self.TENSE] = (df[self.START] <= self._today).map({True: self.PAST, False: self.FUTURE})
        # Calculate population values
        df[self.N] = df[[self.S, self.C]].sum(axis=1).replace(0.0, np.nan).ffill().astype(np.int64)
        # Fill in blanks of ODE model name and tau
        if self.ODE in df:
            df[self.ODE] = df[self.ODE].ffill()
        if self.TAU in df:
            df[self.TAU] = self._tau
        # Set the order of columns
        df = df.drop([self.C, self.CI, self.F, self.R, self.S], axis=1)
        fixed_cols = self.TENSE, self.START, self.END, self.N
        others = [col for col in df.columns if col not in set(fixed_cols)]
        return df.loc[:, [*fixed_cols, *others]]

    def trend(self, force, show_figure, **kwargs):
        """
        Define past phases with S-R trend analysis.

        Args:
            force (bool): if True, change points will be over-written
            show_figure (bool): if True, show the result as a figure
            kwargs: keyword arguments of covsirphy.TrendDetector(), .TrendDetector.sr() and .trend_plot()

        Returns:
            covsirphy.PhaseTracker: self
        """
        df = self._track_df.loc[:self._today].reset_index()[self.SUB_COLUMNS]
        detector = TrendDetector(data=df, area=self._area, **find_args(TrendDetector, **kwargs))
        # Perform S-R trend analysis
        detector.sr(**find_args(TrendDetector.sr, **kwargs))
        # Register phases
        if force:
            start_dates, end_dates = detector.dates()
            _ = [self.define_phase(start, end) for (start, end) in zip(start_dates, end_dates)]
        # Show S-R plane
        if show_figure:
            detector.show(**find_args(trend_plot, **kwargs))
        return self

    def estimate(self, model, tau=None, **kwargs):
        """
        Perform parameter estimation for each phases and update parameter values.

        Args:
            model (covsirphy.ModelBase): ODE model
            tau (int or None): tau value [min] or None (to be estimated)
            kwargs: keyword arguments of ODEHandler(), ODEHandler.estimate_tau() and .estimate_param()

        Returns:
            int: applied or estimated tau value [min]

        Note:
            ODE parameter estimation will be done for all active phases.
        """
        Validator(model, "model").subclass(ModelBase)
        Validator(tau, "tau").tau(default=None)
        # Set-up ODEHandler
        data_df = self._track_df.reset_index()
        data_df = data_df.loc[data_df[self.ID] > 0].dropna(how="all", axis=0)
        handler = ODEHandler(model, data_df[self.DATE].min(), tau=tau, **find_args(ODEHandler, **kwargs))
        start_dates = data_df.groupby(self.ID).first()[self.DATE].sort_values()
        end_dates = data_df.groupby(self.ID).last()[self.DATE].sort_values()
        for (start, end) in zip(start_dates, end_dates):
            y0_series = model.convert(data_df.loc[data_df[self.DATE] >= start], tau=None).iloc[0]
            _ = handler.add(end, y0_dict=y0_series.to_dict())
        # Estimate tau value if necessary
        if tau is None:
            tau = handler.estimate_tau(data_df, **find_args(ODEHandler.estimate_tau, **kwargs))
        # Estimate ODE parameter values
        est_dict = handler.estimate_params(data_df, **kwargs)
        # Register phase information to self
        df = pd.DataFrame.from_dict(est_dict, orient="index")
        df[self.DATE] = df[[self.START, self.END]].apply(lambda x: pd.date_range(x[0], x[1]), axis=1)
        df = df.explode(self.DATE).drop([self.START, self.END], axis=1).set_index(self.DATE)
        df.insert(0, self.ODE, model.NAME)
        df.insert(6, self.TAU, tau)
        all_columns = [*self._track_df.columns.tolist(), *df.columns.tolist()]
        sorted_columns = sorted(set(all_columns), key=all_columns.index)
        self._track_df = self._track_df.combine_first(df)
        self._track_df = self._track_df.reindex(columns=sorted_columns)
        # Set model and tau to self
        self._model = model
        self._tau = tau
        return tau

    def set_ode(self, model, param_df, tau):
        """
        Set ODE model, parameter values manually, not using parameter estimation.

        Args:
            model (covsirphy.ModelBase): ODE model
            param_df (pandas.DataFrame):
                Index
                    Date (pandas.Timestamp): dates to update parameter values
                Columns
                    (float): parameter values
            tau (int): tau value [min] (must not be None)

        Raises:
            ValueError: some model parameters are not included in @param_df

        Note:
            Parameters are defined by model.PARAMETERS.

        Returns:
            int: applied tau value [min]

        Note:
            ODE model for simulation will be overwritten.
        """
        self._model = Validator(model, "model").subclass(ModelBase)
        self._ensure_dataframe(param_df, name="param_df", time_index=True, columns=model.PARAMETERS)
        self._tau = Validator(tau, "tau").tau(default=None)
        if self._tau is None:
            raise NAFoundError("tau")
        new_df = param_df.copy()
        # Add model name
        new_df.insert(0, self.ODE, model.NAME)
        # Calculate reproduction number
        new_df.insert(1, self.RT, None)
        new_df[self.RT] = new_df[model.PARAMETERS].apply(
            lambda x: model(1, **x.to_dict()).calc_r0(), axis=1)
        # Add tau
        new_df[self.TAU] = self._tau
        # Calculate days parameters
        days_df = new_df[model.PARAMETERS].apply(
            lambda x: pd.Series(model(1, **x.to_dict()).calc_days_dict(self._tau)), axis=1)
        new_df = pd.concat([new_df, days_df], axis=1)
        # update tracker
        columns_include_dup = [*self._track_df.columns.tolist(), *new_df.columns.tolist()]
        track_df = self._track_df.reindex(
            columns=sorted(set(columns_include_dup), key=columns_include_dup.index))
        track_df.update(new_df)
        self._track_df = track_df.copy()
        return self._tau

    def simulate(self):
        """
        Perform simulation with the multi-phased ODE model.

        Raises:
            covsirphy.UnExecutedError: either tau value or phase information was not set

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases

        Note:
            Deactivated phases will be included.

        Note:
            Un-registered phases will not be included.

        Note:
            If parameter set is not registered for the current phase and
            the previous phase has parameter set, this set will be used for the current phase.
        """
        # Model and tau must be set
        if self._model is None:
            raise UnExecutedError("PhaseTracker.estimate() or PhaseTracker.set_ode()")
        # Get parameter sets and initial values
        record_df = self._track_df.copy()
        record_df = record_df.loc[record_df[self.ID] != 0].ffill().dropna(subset=self._model.PARAMETERS)
        start_dates = record_df.reset_index().groupby(self.ID).first()[self.DATE].sort_values()
        end_dates = record_df.reset_index().groupby(self.ID).last()[self.DATE].sort_values()
        # Set-up ODEHandler
        handler = ODEHandler(self._model, record_df.index.min(), tau=self._tau)
        parameters = self._model.PARAMETERS[:]
        for (start, end) in zip(start_dates, end_dates):
            param_dict = record_df.loc[end, parameters].to_dict()
            if end <= self._today:
                ph_df = record_df.loc[start:, [self.S, self.CI, self.F, self.R]].reset_index()
                y0_dict = self._model.convert(ph_df, self._tau).iloc[0].to_dict()
            else:
                y0_dict = None
            _ = handler.add(end, param_dict=param_dict, y0_dict=y0_dict)
        # Perform simulation
        sim_df = handler.simulate()
        sim_df[self.C] = sim_df[[self.CI, self.F, self.R]].sum(axis=1)
        return sim_df.loc[:, self.SUB_COLUMNS]

    def parse_range(self, dates=None, past_days=None, phases=None):
        """
        Parse date range and return the minimum date and maximum date.

        Args:
            dates (tuple(str or pandas.Timestamp or None, ) or None): start date and end date
            past_days (int or None): how many past days to use in calculation from today (property)
            phases (list[str] or None): phase names to use in calculation

        Raises:
            covsirphy.UnExecutedError: no phases were registered
            ValueError: @dates argument does not have exact two elements

        Returns:
            tuple(pandas.Timestamp, pandas.Timestamp): the minimum date and maximum date

        Notes:
            When not specified (i.e. None was applied),
            the start date of the 0th phase will be used as the minimum date.

        Notes:
            When not specified (i.e. None was applied),
            the end date of the last phase phase will be used as the maximum date.

        Note:
            When @past_days was specified, (today - @past_days, today) will be returned.

        Note:
            In @phases, 'last' means the last registered phase.

        Note:
            Priority is given in the order of @dates, @past_days, @phases.
        """
        if not self:
            raise UnExecutedError("PhaseTracker.define_phase()")
        # Get list of phases: index=phase names, columns=Start/End
        track_df = self._track_df.reset_index()
        track_df = track_df.loc[track_df[self.ID] != 0]
        track_df[self.ID], _ = track_df[self.ID].factorize()
        first_df = track_df.groupby(self.ID).first()
        df = first_df.join(track_df.groupby(self.ID).last(), rsuffix="_last")
        df = df.rename(columns={self.DATE: self.START, f"{self.DATE}_last": self.END})
        df.index = [self.num2str(num) for num in df.index]
        # Get default values
        start_default, end_default = df[self.START].min(), df[self.END].max()
        # Read @dates
        if dates is not None:
            if len(dates) != 2:
                raise ValueError(f"@dates must be a tuple which has two elements, but {dates} was applied.")
            start = Validator(dates[0], name="the first element of 'dates' argument").date(default=start_default)
            end = Validator(dates[1], name="the second element of 'dates' argument").date(default=end_default)
            self._ensure_date_order(start, end, name="the second element of 'dates' argument")
            return (start, end)
        # Read @past_days
        if past_days is not None:
            past_days = Validator(past_days, "past_days").int(value_range=(1, None))
            return (self._today - timedelta(days=past_days), self._today)
        # No arguments were specified
        if phases is None:
            return (start_default, end_default)
        # Read @phases
        self._ensure_list(phases, name="phases")
        dates = []
        for phase in phases:
            phase_replaced = df.index[-1] if phase == "last" else phase
            self._ensure_selectable(phase_replaced, df.index.tolist(), name="phase")
            start = df.loc[phase_replaced, self.START]
            end = df.loc[phase_replaced, self.END]
            dates.extend(pd.date_range(start, end).tolist())
        return (min(dates), max(dates))

    def _ensure_selectable(self, target, candidates, name="target"):
        """
        Ensure that the target can be selectable.

        Args:
            target (object): target to check
            candidates (list[object]): list of candidates
            name (str): name of the target
        """
        Validator(candidates, "candidates").sequence()
        if target in candidates:
            return target
        raise UnExpectedValueError(name=name, value=target, candidates=candidates)
