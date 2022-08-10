#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
from multiprocessing import cpu_count, Pool
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy._deprecated.jhu_data import JHUData
from covsirphy._deprecated.population import PopulationData
from covsirphy._deprecated._mbase import ModelBase
from covsirphy._deprecated.phase_unit import PhaseUnit


class MPEstimator(Term):
    """
    Deprecated.
    Perform multiprocessing of Phaseunit.estimate()

    Args:
        model (covsirphy.ModelBase or None): ODE model
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        record_df (pandas.DataFrame)
            Index
                reset index
            Columns
                - Date (pd.Timestamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - any other columns will be ignored
        tau (int or None): tau value [min], a divisor of 1440
        kwargs: keyword arguments of model parameters

    Note:
        When @record_df is None, @jhu_data and @population_data must be specified.
    """

    @deprecate("MPEstimator", new="ODEHandler", version="2.19.1-zeta-fu1")
    def __init__(self, model, jhu_data=None, population_data=None,
                 record_df=None, tau=None, **kwargs):
        # Records
        if jhu_data is not None and population_data is not None:
            self.jhu_data = self._ensure_instance(
                jhu_data, JHUData, name="jhu_data")
            # Population
            self.population_data = self._ensure_instance(
                population_data, PopulationData, name="population_data")
            self.from_dataset = True
        else:
            self._ensure_dataframe(record_df, name="record_df", columns=self.NLOC_COLUMNS)
            self.record_df = record_df.copy()
            self.from_dataset = False
        # Arguments
        self.model = Validator(model, "model").subclass(ModelBase)
        self._tau = Validator(tau, "tau").tau(default=None)
        self.param_dict = {
            k: v for (k, v) in kwargs.items() if k in model.PARAMETERS}
        self._units = []

    @staticmethod
    def _ensure_instance(target, class_obj, name="target"):
        """
        Ensure the target is an instance of the class object.

        Args:
            target (instance): target to ensure
            parent (class): class object
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        s = f"@{name} must be an instance of {class_obj}, but {type(target)} was applied."
        if not isinstance(target, class_obj):
            raise TypeError(s)
        return target

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

    @property
    def tau(self):
        """
        int or None: tau value [min]
        """
        return self._tau

    def add(self, units):
        """
        Register PhaseUnits.

        Args:
            units (list[covsirphy.PhaseUnit]): list of phases

        Returns:
            covsirphy.MPEstimator: self
        """
        if not isinstance(units, list):
            raise TypeError("@units must be a list of PhaseUnits.")
        type_ok = all(isinstance(unit, PhaseUnit) for unit in units)
        if not type_ok:
            raise TypeError("@units must be a list of PhaseUnits.")
        units = [
            unit.set_ode(model=self.model, **self.param_dict) for unit in units
        ]
        self._units.extend(units)
        return self

    def _run(self, unit, tau, auto_complement=False, **kwargs):
        """
        Run estimation for one phase.

        Args:
            unit (covsirphy.PhaseUnit): unit of one phase
            tau (int or None): tau value [min], a divisor of 1440
            auto_complement (bool): if True and necessary, the number of cases will be complemented
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()
        """
        # Set tau
        unit.set_ode(tau=tau)
        # Parameter estimation
        if self.from_dataset:
            id_dict = unit.id_dict.copy()
            try:
                country = id_dict["country"]
            except KeyError:
                raise KeyError("PhaseUnit.id_dict['country'] must have country name.") from None
            province = id_dict["province"] if "province" in id_dict else None
            population = self.population_data.value(
                country=country, province=province)
            record_df, _ = self.jhu_data.records(
                country=country, province=province, population=population,
                auto_complement=auto_complement)
        else:
            record_df = self.record_df.copy()
        unit.estimate(record_df=record_df, **kwargs)
        # Show the number of trials and runtime
        unit_dict = unit.to_dict()
        trials, runtime = unit_dict[self.TRIALS], unit_dict[self.RUNTIME]
        print(f"\t{unit}: finished {trials:>4} trials in {runtime}")
        return unit

    def run(self, n_jobs=-1, **kwargs):
        """
        Run estimation.

        Args:
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()

        Returns:
            list[covsirphy.PhaseUnit]
        """
        units = self._units[:]
        results = []
        # The number of parallel jobs
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        # Start optimization
        print(f"\n<{self.model.NAME} model: parameter estimation>")
        print(f"Running optimization with {n_jobs} CPUs...")
        stopwatch = StopWatch()
        # Estimation of the last phase will be done to determine tau value
        if self._tau is None:
            unit_sel, units = units[-1], units[:-1]
            unit_est = self._run(unit=unit_sel, tau=None, **kwargs)
            self._tau = unit_est.tau
            results = [unit_est]
        # Estimation of each phase
        est_f = functools.partial(self._run, tau=self._tau, **kwargs)
        if n_jobs == 1:
            results = [est_f(unit) for unit in units]
        else:
            with Pool(n_jobs) as p:
                units_est = p.map(est_f, units)
            results.extend(units_est)
        # Completion
        stopwatch.stop()
        print(f"Completed optimization. Total: {stopwatch.stop_show()}")
        return results
