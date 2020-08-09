#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.estimator import Estimator
from covsirphy.simulation.simulator import ODESimulator


class PhaseUnit(Term):
    """
    Save information of  a phase.

    Args:
        start_date (str): start date of the phase
        end_date (str): end date of the phase
        population (int): population value
    """

    def __init__(self, start_date, end_date, population):
        self.ensure_date_order(start_date, end_date, name="end_date")
        self._start_date = start_date
        self._end_date = end_date
        self._population = self.ensure_population(population)
        # Summary of information
        self.info_dict = {
            self.START: start_date,
            self.END: end_date,
            self.N: population,
            self.ODE: None,
            self.RT: None
        }
        self._ode_dict = {self.TAU: None}
        self.day_param_dict = {}
        self.est_dict = {
            self.RMSLE: None,
            self.TRIALS: None,
            self.RUNTIME: None
        }
        # Init
        self._model = None
        self._record_df = pd.DataFrame()
        self.y0_dict = {}
        self._estimator = None

    def __str__(self):
        return f"Phase ({self._start_date} - {self._end_date})"

    def __eq__(self, other):
        if not isinstance(other, PhaseUnit):
            return NotImplemented
        if self._start_date != other.start_date:
            return False
        if self._end_date != other.end_date:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def start_date(self):
        """
        str: start date
        """
        return self._start_date

    @property
    def end_date(self):
        """
        str: end date
        """
        return self._end_date

    @property
    def population(self):
        """
        str: population value
        """
        return self._population

    @property
    def tau(self):
        """
        int or None: tau value [min]
        """
        return self._ode_dict[self.TAU]

    @tau.setter
    def tau(self, value):
        self._ode_dict[self.TAU] = self.ensure_tau(value)

    @property
    def model(self):
        """
        covsirphy.ModelBase: model description
        """
        return self._model

    @property
    def estimator(self):
        """
        covsirphy.Estimator or None: estimator object
        """
        return self._estimator

    @property
    def ode_dict(self):
        """
        dict: parameter values and tau value
        """
        return self._ode_dict

    def to_dict(self):
        """
        Summarize phase information and return as a dictionary.

        Returns:
            dict:
                - Start: start date of the phase
                - End: end date of the phase
                - Population: population value of the start date
                - if available:
                    - ODE: model name
                    - Rt: (basic) reproduction number
                    - parameter values if available
                    - day parameter values if available
                    - tau: tau value [min]
                    - RMSLE: RMSLE value of estimation
                    - Trials: the number of trials in estimation
                    - Runtime: runtime of estimation
        """
        return {
            **self.info_dict,
            **self._ode_dict,
            **self.day_param_dict,
            **self.est_dict
        }

    def summary(self):
        """
        Summarize information.

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns:
                    - Start: start date of the phase
                    - End: end date of the phase
                    - Population: population value of the start date
                    - if available:
                        - ODE: model name
                        - Rt: (basic) reproduction number
                        - parameter values if available
                        - tau: tau value [min]
                        - day parameter values if available
                        - RMSLE: RMSLE value of estimation
                        - Trials: the number of trials in estimation
                        - Runtime: runtime of estimation
        """
        summary_dict = self.to_dict()
        df = pd.DataFrame.from_dict(summary_dict, orient="index").T
        return df.dropna(how="all", axis=1).fillna(self.UNKNOWN)

    def set_ode(self, model, tau=None, **kwargs):
        """
        Set ODE model, tau value and parameter values, if necessary.

        Args:
            model (covsirphy.ModelBase): ODE model
            tau (int or None): tau value [min], a divisor of 1440
            kwargs: keyword arguments of model parameters
        """
        tau = self.ensure_tau(tau) or self._ode_dict[self.TAU]
        # Model
        self._model = self.ensure_subclass(model, ModelBase, name="model")
        self.info_dict[self.ODE] = model.NAME
        # Parameter values
        param_dict = self._ode_dict.copy()
        param_dict.update(kwargs)
        param_dict = {
            k: v for (k, v) in param_dict.items() if k in model.PARAMETERS}
        self._ode_dict = param_dict.copy()
        self._ode_dict[self.TAU] = tau
        # Reproduction number and day parameters
        if param_dict.keys() == set(model.PARAMETERS):
            model_instance = model(population=self._population, **param_dict)
            self.info_dict[self.RT] = model_instance.calc_r0()
            if tau is not None:
                self.day_param_dict = model_instance.calc_days_dict(tau)

    @property
    def record_df(self):
        """
        pandas.DataFrame: records of the phase
            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - any other columns will be ignored
        """
        return self._record_df

    @record_df.setter
    def record_df(self, df):
        self._record_df = self.ensure_dataframe(
            df, name="df", columns=self.NLOC_COLUMNS)
        self.set_y0(df)

    def _model_is_registered(self):
        """
        Ensure that model was set.

        Raises:
            NameError: ODE model is not registered
        """
        if self._model is None:
            raise ValueError(
                "PhaseUnit.set_ode(model) must be done in advance.")

    def estimate(self, record_df=None, **kwargs):
        """
        Perform parameter estimation.

        Args:
            record_df (pandas.DataFrame or None)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any other columns will be ignored
            **kwargs: keyword arguments of Estimator.run()

        Raises:
            NameError: PhaseUnit.set_ode(model) was not done in advance.
            ValueError: @record_df is None and PhaseUnit.record_df = ... was not done in advance.

        Notes:
            If @record_df is None, registered records will be used.
        """
        self._model_is_registered()
        # Records
        if record_df is None:
            record_df = self._record_df.copy()
        if record_df.empty:
            raise ValueError(
                "@record_df must be specified or PhaseUnit.record_df = ... must be done in advance.")
        self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # Check dates
        sta = self.date_obj(self.start_date)
        end = self.date_obj(self.end_date)
        series = record_df[self.DATE]
        record_df = record_df.loc[(series >= sta) & (series <= end), :]
        # Parameter estimation of ODE model
        estimator = Estimator(
            record_df, self._model, self._population, **self._ode_dict, **kwargs)
        estimator.run(**kwargs)
        self._read_estimator(estimator, record_df)
        # Set estimator
        self._estimator = estimator

    def _read_estimator(self, estimator, record_df):
        """
        Read the result of parameter estimation and update the summary of phase.

        Args:
            estimator (covsirphy.Estimator): estimator which finished estimation
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
        """
        # Reproduction number
        est_dict = estimator.to_dict()
        self.info_dict[self.RT] = est_dict.pop(self.RT)
        # Get parameter values and tau value
        ode_set = set([*self._model.PARAMETERS, self.TAU])
        ode_dict = {
            k: v for (k, v) in est_dict.items() if k in ode_set}
        self._ode_dict.update(ode_dict)
        # Other information of estimation
        other_dict = dict(est_dict.items() - ode_dict.items())
        self.est_dict.update(other_dict)
        # Initial values
        self.set_y0(record_df=record_df)

    def set_y0(self, record_df):
        """
        Set initial values.

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
        """
        self._model_is_registered()
        df = record_df.loc[
            record_df[self.DATE] >= self.date_obj(self.start_date), :]
        df = self._model.tau_free(df, self._population, tau=None)
        y0_dict = df.iloc[0, :].to_dict()
        self.y0_dict = {
            k: v for (k, v) in y0_dict.items() if k in set(self._model.VARIABLES)
        }

    def simulate(self, y0_dict=None):
        """
        Perform simulation with the set/estimated parameter values.

        Args:
            y0_dict (dict or None):
                - key (str): variable name
                - value (float): initial value

        Returns:
            pandas.DataFrame
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Notes:
            Simulation starts at the start date of the phase.
            Simulation end at the next date of the end date of the phase.
        """
        # Initial values
        y0_dict = y0_dict or {}
        y0_dict.update(self.y0_dict)
        diff_set = set(self._model.VARIABLES) - y0_dict.keys()
        if diff_set:
            diff_str = ", ".join(list(diff_set))
            s = "s" if len(diff_set) > 1 else ""
            raise KeyError(
                f"Initial value{s} of {diff_str} must be specified by @y0_dict or PhaseUnit.set_y0()")
        # Conditions
        param_dict = self._ode_dict.copy()
        tau = param_dict.pop(self.TAU)
        last_date = self.tomorrow(self._end_date)
        # Simulation
        simulator = ODESimulator()
        simulator.add(
            model=self._model,
            step_n=self.steps(self._start_date, last_date, tau),
            population=self._population,
            param_dict=param_dict,
            y0_dict=y0_dict
        )
        simulator.run()
        # Dimensionalized values
        df = simulator.dim(tau=tau, start_date=self._start_date)
        df = self._model.restore(df)
        # Return day-level data
        df = df.set_index(self.DATE).resample("D").first()
        df = df.loc[df.index <= self.date_obj(last_date), :]
        return df.reset_index().loc[:, self.NLOC_COLUMNS]
