#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
from multiprocessing import cpu_count, Pool
from covsirphy.util.stopwatch import StopWatch
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.phase.phase_unit import PhaseUnit


class MPEstimator(Term):
    """
    Perform parallel jobs of Phaseunit.estimate()

    Args:
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
        model (covsirphy.ModelBase or None): ODE model
        tau (int or None): tau value [min], a divisor of 1440
        kwargs: keyword arguments of model parameters
    """

    def __int__(self, record_df, model, tau=None, **kwargs):
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
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
        self._units.append(units)
        return self

    def _run(self, unit, record_df, tau, **kwargs):
        """
        Run estimation for one phase.

        Args:
            unit (covsirphy.PhaseUnit): unit of one phase
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
            kwargs: keyword arguments of model parameters and covsirphy.Estimator.run()
        """
        # Set tau and estimation
        unit.set_ode(tau=tau).estimate(record_df=record_df, **kwargs)
        # Show the number of trials and runtime
        unit_dict = unit.to_dict()
        trials, runtime = unit_dict[self.TRIALS], unit_dict[self.RUNTIME]
        print(
            f"\t{unit}: finished {trials} trials in {runtime}"
        )
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
            unit_est = self._run(
                unit=unit_sel, record_df=self.record_df, tau=None, **kwargs)
            self._tau = unit_est.tau
            results = [unit_est]
        # Estimation of each phase
        est_f = functools.partial(
            self._run, record_df=self.record_df, tau=self._tau, **kwargs)
        with Pool(n_jobs) as p:
            units_est = p.map(est_f, units)
        results.extend(units_est)
        # Completion
        stopwatch.stop()
        print(f"Completed optimization. Total: {stopwatch.show()}")
        return results
