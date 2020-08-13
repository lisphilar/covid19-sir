#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
from multiprocessing import cpu_count, Pool
from covsirphy.util.stopwatch import StopWatch
from covsirphy.cleaning.term import Term
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.population import PopulationData
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.phase_unit import PhaseUnit


class MPEstimator(Term):
    """
    Perform multiprocessing of Phaseunit.estimate()

    Args:
        model (covsirphy.ModelBase or None): ODE model
        jhu_data (covsirphy.JHUData): object of records
        population_data (covsirphy.PopulationData): PopulationData object
        record_df (pandas.DataFrame)
            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - any other columns will be ignored
        tau (int or None): tau value [min], a divisor of 1440
        kwargs: keyword arguments of model parameters

    Notes:
        When @record_df is None, @jhu_data and @population_data must be specified.
    """

    def __init__(self, model, jhu_data=None, population_data=None,
                 record_df=None, tau=None, **kwargs):
        # Records
        if jhu_data is not None and population_data is not None:
            self.jhu_data = self.ensure_instance(
                jhu_data, JHUData, name="jhu_data")
            # Population
            self.population_data = self.ensure_instance(
                population_data, PopulationData, name="population_data")
            self.from_dataset = True
        else:
            self.record_df = self.ensure_dataframe(
                record_df, name="record_df", columns=self.NLOC_COLUMNS)
            self.from_dataset = False
        # Arguments
        self.model = self.ensure_subclass(model, ModelBase, "model")
        self._tau = self.ensure_tau(tau)
        self.param_dict = {
            k: v for (k, v) in kwargs.items() if k in model.PARAMETERS}
        self._units = []

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

    def _run(self, unit, tau, **kwargs):
        """
        Run estimation for one phase.

        Args:
            unit (covsirphy.PhaseUnit): unit of one phase
            tau (int or None): tau value [min], a divisor of 1440
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
                raise KeyError(
                    "PhaseUnit.id_dict['country'] must have country name.")
            province = id_dict["province"] if "province" in id_dict else None
            population = self.population_data.value(
                country=country, province=province)
            record_df = self.jhu_data.subset(
                country=country, province=province, population=population
            )
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
        with Pool(n_jobs) as p:
            units_est = p.map(est_f, units)
        results.extend(units_est)
        # Completion
        stopwatch.stop()
        print(f"Completed optimization. Total: {stopwatch.show()}")
        return results
