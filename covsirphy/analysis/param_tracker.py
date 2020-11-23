#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.phase_unit import PhaseUnit
from covsirphy.phase.phase_estimator import MPEstimator
from covsirphy.phase.phase_series import PhaseSeries


class ParamTracker(Term):
    """
    Split records with S-R trend analysis and estimate parameter values of the phases.

    Args:
        record_df (pandas.DataFrame): records
            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - Susceptible (int): the number of susceptible cases
        phase_series (covsirphy.PhaseSeries): phase series object with first/last dates and population
        area (str or None): area name, like Japan/Tokyo, or empty string
        tau (int or None): tau value [min]
    """
    METRICS_DICT = {
        "MAE": sklearn.metrics.mean_absolute_error,
        "MSE": sklearn.metrics.mean_squared_error,
        "MSLE": sklearn.metrics.mean_squared_log_error,
        "RMSE": lambda x1, x2: sklearn.metrics.mean_squared_error(x1, x2, squared=False),
        "RMSLE": lambda x1, x2: np.sqrt(sklearn.metrics.mean_squared_log_error(x1, x2)),
    }

    def __init__(self, record_df, phase_series, area=None, tau=None):
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.SUB_COLUMNS)
        self.series = self.ensure_instance(
            phase_series, PhaseSeries, name="phase_seres")
        self.area = area or ""
        self.tau = self.ensure_tau(tau)

    def trend(self, force=True, show_figure=False, filename=None, **kwargs):
        """
        Split the records with trend analysis.

        Args:
            force (bool): if True, change points will be over-written
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
            kwargs: keyword arguments of ChangeFinder()

        Returns:
            covsirphy.PhaseSeries
        """
        sr_df = self.record_df.set_index(self.DATE).loc[:, [self.R, self.S]]
        if force or not self.series:
            self.series.trend(sr_df=sr_df, **kwargs)
        if show_figure:
            self.series.trend_show(
                sr_df=sr_df, area=self.area, filename=filename)
        return self.series

    def _ensure_phase_setting(self):
        """
        Ensure that phases were set.
        """
        if not self.series:
            raise ValueError(
                "Phases should be registered with .trend() or .add() in advance.")

    def find_phase(self, date):
        """
        Find the name of the phase which has the date.

        Args:
            date (str): date, like 01Jan2020

        Returns:
            tuple(str, covsirphy.PhaseUnit):
                str: phase name, like 1st, 2nd,...
                covsirphy.PhaseUnit: phase unit
        """
        self._ensure_phase_setting()
        self.ensure_date(date)
        phase_nest = [
            (self.num2str(i), unit) for (i, unit) in enumerate(self.series) if date in unit]
        try:
            return phase_nest[0]
        except IndexError:
            raise IndexError(f"Phase on {date} is not registered.") from None

    def change_dates(self):
        """
        Return the list of changed dates (start dates of phases since 1st phase).

        Returns:
            list[str]: list of change dates
        """
        return [unit.start_date for unit in self.series][1:]

    def near_change_dates(self):
        """
        Show the list of dates which are yesterday/tomorrow of the start/end dates.

        Returns:
            list[str]: list of dates
        """
        base_dates = [
            date for ph in self.series for date in [ph.start_date, ph.end_date]]
        return [
            date for base_date in base_dates
            for date in [self.yesterday(base_date), base_date, self.tomorrow(base_date)]]

    def disable(self, phases):
        """
        The phases will be disabled.

        Args:
            phase (list[str] or None): phase names

        Returns:
            covsirphy.PhaseSeries
        """
        phases = self.ensure_list(phases, candidates=None, name="phases")
        for phase in phases:
            self.series.disable(phase)
        return self.series

    def enable(self, phases):
        """
        The phases will be enabled.

        Args:
            phase (list[str] or None): phase names

        Returns:
            covsirphy.PhaseSeries
        """
        phases = self.ensure_list(phases, candidates=None, name="phases")
        for phase in phases:
            self.series.enable(phase)
        return self.series

    def separate(self, date, population=None, **kwargs):
        """
        Create a new phase with the change point.
        New phase name will be automatically determined.

        Args:
            date (str): change point, i.e. start date of the new phase
            population (int): population value of the change point
            kwargs: keyword arguments of PhaseUnit.set_ode() if update is necessary

        Returns:
            covsirphy.PhaseSeries
        """
        phase, old = self.find_phase(date)
        if date in self.near_change_dates():
            raise ValueError(
                f"Cannot be separated on {date} because this date is too close to registered change dates.")
        new_pre = PhaseUnit(
            old.start_date, self.yesterday(date), old.population)
        setting_dict = old.to_dict()
        setting_dict.update(kwargs)
        new_pre.set_ode(**setting_dict)
        new_fol = PhaseUnit(date, old.end_date, population or old.population)
        new_fol.set_ode(model=old.model, **setting_dict)
        self.series.replaces(phase, [new_pre, new_fol])
        return self.series

    def past_phases(self, phases=None):
        """
        Return names and phase units of the past phases.

        Args:
            phases (tuple/list[str]): list of phase names, like 1st, 2nd...

        Returns:
            tuple(list[str], list[covsirphy.PhaseUnit]):
                list[str]: list of phase names
                list[covsirphy.PhaseUnit]: list of phase units

        Notes:
            If @phases is None, return the all past phases.
            If @phases is not None, intersection will be selected.
        """
        self._ensure_phase_setting()
        # List of past phases
        last_date = self.record_df[self.DATE].max().strftime(self.DATE_FORMAT)
        past_nest = [
            [self.num2str(num), unit]
            for (num, unit) in enumerate(self.series)
            if unit and unit <= last_date
        ]
        past_phases, past_units = zip(*past_nest)
        # Select phases to use
        selected_phases = self.ensure_list(
            phases or past_phases, candidates=past_phases, name="phases")
        final_phases = list(set(selected_phases) & set(past_phases))
        # Convert phase names to phase units
        selected_units = [self.series.unit(ph) for ph in selected_phases]
        final_units = list(set(selected_units) & set(past_units))
        return (final_phases, final_units)

    def estimate(self, model, phases=None, n_jobs=-1, **kwargs):
        """
        Perform parameter estimation for each phases.

        Args:
            model (covsirphy.ModelBase): ODE model
            phases (list[str]): list of phase names, like 1st, 2nd...
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()

        Returns:
            tuple(int, covsirphy.PhaseSeries): tau value [min] and phase series

        Notes:
            - If @phases is None, all past phase will be used.
            - Phases with estimated parameter values will be ignored.
            - In kwargs, tau value cannot be included.
        """
        self._ensure_phase_setting()
        model = self.ensure_subclass(model, ModelBase, "model")
        units = [
            unit.set_id(phase=phase)
            for (phase, unit) in zip(*self.past_phases(phases=phases))
            if unit.id_dict is None
        ]
        if not units:
            raise IndexError("All phases have completed parameter estimation.")
        # Parameter estimation
        mp_estimator = MPEstimator(
            record_df=self.record_df, model=model, tau=self.tau, **kwargs
        )
        mp_estimator.add(units)
        results = mp_estimator.run(n_jobs=n_jobs, **kwargs)
        self.tau = mp_estimator.tau
        # Register the results
        self.series.replaces(phase=None, new_list=results, keep_old=True)
        return (self.tau, self.series)

    def simulate(self, y0_dict=None):
        """
        Simulate ODE models with set/estimated parameter values.

        Args:
            y0_dict(dict[str, float] or None): dictionary of initial values of variables

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - Variables of the model and dataset (int): Confirmed etc.
        """
        self._ensure_phase_setting()
        try:
            return self.series.simulate(record_df=self.record_df, y0_dict=y0_dict)
        except NameError:
            raise NameError(
                "Parameter estimation should be done with .estimate() in advance.") from None

    def _compare_with_actual(self, variables, y0_dict=None):
        """
        Compare actual/simulated number of cases.

        Args:
            variables (list[str]): variables to use in calculation
            y0_dict(dict[str, float] or None): dictionary of initial values of variables

        Returns:
            tuple(pandas.DataFrame, pandas.DataFrame):
                - actual (pandas.DataFrame):
                    Index: Date (pd.TimeStamp)
                    Columns: variables defined by @variables
                - simulated (pandas.DataFrame):
                    Index: Date (pd.TimeStamp)
                    Columns: variables defined by @variables
        """
        record_df = self.record_df.copy().set_index(self.DATE)
        simulated_df = self.simulate(y0_dict=y0_dict).set_index(self.DATE)
        df = record_df.join(simulated_df, how="inner", rsuffix="_sim").dropna()
        rec_df = df.loc[:, variables]
        sim_df = df.loc[:, [f"{col}_sim" for col in variables]]
        sim_df.columns = variables
        return (rec_df, sim_df)

    def score(self, metrics="RMSLE", variables=None, phases=None, y0_dict=None):
        """
        Evaluate accuracy of phase setting and parameter estimation of selected enabled phases.

        Args:
            metrics (str): "MAE", "MSE", "MSLE", "RMSE" or "RMSLE"
            variables (list[str] or None): variables to use in calculation
            phases (list[str] or None): phases to use in calculation
            y0_dict(dict[str, float] or None): dictionary of initial values of variables

        Returns:
            float: score with the specified metrics

        Notes:
            If @variables is None, ["Infected", "Fatal", "Recovered"] will be used.
            "Confirmed", "Infected", "Fatal" and "Recovered" can be used in @variables.
            If @phases is None, all phases will be used.
        """
        # Arguments
        if metrics not in self.METRICS_DICT:
            metrics_str = ", ".join(list(self.METRICS_DICT.keys()))
            raise ValueError(
                f"@metrics must be selected from {metrics_str}, but {metrics} was applied.")
        variables = variables or [self.CI, self.F, self.R]
        variables = self.ensure_list(
            variables, self.VALUE_COLUMNS, name="variables")
        # Disable the non-target phases
        all_phases, _ = self.past_phases(phases=None)
        target_phases, _ = self.past_phases(phases=phases)
        ignored_phases = list(set(all_phases) - set(target_phases))
        if ignored_phases:
            self.disable(ignored_phases)
        # Get the number of cases
        rec_df, sim_df = self._compare_with_actual(
            variables=variables, y0_dict=y0_dict)
        # Calculate score
        score = self.METRICS_DICT[metrics.upper()](rec_df, sim_df)
        # Enable the disabled non-target phases
        if ignored_phases:
            self.enable(ignored_phases)
        return score
