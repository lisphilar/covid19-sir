#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from covsirphy.util.error import deprecate
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase


class ODESimulator(Term):
    """
    Simulation of an ODE model for one phase.

    Args:
        country (str or None): country name
        province (str or None): province name
    """

    def __init__(self, country=None, province=None):
        self.country = country or self.UNKNOWN
        self.province = province or self.UNKNOWN
        # keys: model, step_n, population, param_dict, y0_dict
        self.setting = {}
        # key: non-dim variable name, value: dimensional variable name
        self.var_dict = {}

    def add(self, model, step_n, population, param_dict=None, y0_dict=None):
        """
        Add models to the simulator.

        Args:
            model (subclass of cs.ModelBase): the first ODE model
            step_n (int): the number of steps
            population (int): population in the place
            param_dict (dict):
                - key (str): parameter name
                - value (float): parameter value
                - dictionary of parameter values or None
                - if not include some params, the last values will be used
                    - NameError when the model is the first model
                    - NameError if new params are included
            y0_dict (dict):
                - key (str): variable name
                - value (float): initial value
                - dictionary of dimensional initial values or None
                - None or if not include some variables, the last values will be used
                    - NameError when the model is the first model
                    - NameError if new variable are included
        """
        if self.setting:
            raise ValueError(
                "Simulation for two phases is not supported from version 2.7.0")
        # Register the setting
        self.setting = {
            "model": self.ensure_subclass(model, ModelBase, name="model"),
            "step_n": self.ensure_natural_int(step_n, name="step_n"),
            "population": self.ensure_population(population),
            "param_dict": self._ensure_parameters(model, param_dict),
            "y0_dict": self._ensure_initial_values(model, y0_dict),
        }
        # Update variable dictionary
        self.var_dict.update(model.VAR_DICT)

    def _ensure_parameters(self, model, param_dict):
        """
        Validate the dictionary of parameters.

        Args:
            model (subclass of cs.ModelBase): the ODE model
            param_dict (dict):
                - key (str): parameter name
                - value (float): parameter value

        Returns:
            dict(str, str): dictionary of parameters

        Notes:
            If a parameter value is not registered, None will be registered.
        """
        param_dict = param_dict or {}
        usable_dict = {
            p: param_dict[p] if p in param_dict else None for p in model.PARAMETERS}
        if None not in usable_dict.values():
            return usable_dict
        none_params = [k for (k, v) in usable_dict.items() if v is None]
        s = "s" if len(none_params) > 1 else ""
        raise NameError(
            f"Parameter value{s} of {', '.join(none_params)} must be specified by @param_dict."
        )

    def _ensure_initial_values(self, model, y0_dict):
        """
        Validate the dictionary of initial values.

        Args:
            model (subclass of cs.ModelBase): the ODE model
            y0_dict (dict): dictionary of initial values
                - key (str): dimensional variable name
                - value (int):initial value of the variable

        Returns:
            dict(str, str): dictionary of initial values

        Notes:
            If initial value of a variable is not registered, None will be registered.
        """
        y0_dict = y0_dict or {}
        usable_dict = {
            v: y0_dict[v] if v in y0_dict else None for v in model.VARIABLES}
        if None not in usable_dict.values():
            return usable_dict
        none_vars = [k for (k, v) in usable_dict.items() if v is None]
        s = "s" if len(none_vars) > 1 else ""
        raise NameError(
            f"Initial value{s} of {', '.join(none_vars)} must be specified by @y0_dict."
        )

    def _solve_ode(self, model, step_n, param_dict, y0_dict, population):
        """
        Solve ODE of the model.

        Args:
            model (subclass of cs.ModelBase): the ODE model
            step_n (int): the number of steps
            param_dict (dict): dictionary of parameter values
                - key (str): parameter name
                - value (float): parameter value
            y0_dict (dict): dictionary of initial values
                - key (str): dimensional variable name
                - value (int):initial value of the variable
            population (int): total population

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - t (int): Elapsed time divided by tau value [-]
                    - columns with dimensional variables
        """
        tstart, dt, tend = 0, 1, step_n
        variables = model.VARIABLES[:]
        initials = [y0_dict[var] for var in variables]
        sol = solve_ivp(
            fun=model(population=population, **param_dict),
            t_span=[tstart, tend],
            y0=np.array(initials, dtype=np.int64),
            t_eval=np.arange(tstart, tend + dt, dt),
            dense_output=False
        )
        t_df = pd.Series(data=sol["t"], name=self.TS)
        y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=variables)
        y_df = y_df.round()
        return pd.concat([t_df, y_df], axis=1)

    @deprecate(
        old="ODESimulator.run()",
        new="ODESimulator.taufree(), .non_dim() or .dim(tau, start_date) directly")
    def run(self):
        """
        From version 2.7.0, it is not necessary to perform ODESimulator.run().
        Please directory use ODESimulator.taufree(), .non_dim() or .dim(tau, start_date)
        """
        return self.taufree()

    def taufree(self):
        """
        Return tau-free results.

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - t (int): Elapsed time divided by tau value [-]
                    - columns with dimensionalized variables
        """
        df = self._solve_ode(**self.setting)
        df[self.TS] = df.index
        return df.reset_index(drop=True)

    def non_dim(self):
        """
        Return the non-dimensionalized results.

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - t (int): Elapsed time divided by tau value [-]
                    - non-dimensionalized variables of Susceptible etc.
        """
        df = self.taufree()
        df = df.set_index(self.TS)
        df = df.apply(lambda x: x / sum(x), axis=1)
        var_dict_rev = {v: k for (k, v) in self.var_dict.items()}
        df.columns = [var_dict_rev[col] for col in df.columns]
        df = df.reset_index()
        return df

    def dim(self, tau, start_date):
        """
        Return the dimensionalized results.

        Args:
            tau (int): tau value [min]
            start_date (str): start date of the records, like 22Jan2020

        Returns:
            (pandas.DataFrame)
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/state name
                    - variables of the models (int)
        """
        df = self.taufree()
        df = df.drop(self.TS, axis=1).reset_index(drop=True)
        var_cols = df.columns.tolist()
        df = df.astype(np.int64)
        # Date
        start_obj = datetime.strptime(start_date, self.DATE_FORMAT)
        elapsed = pd.Series(df.index * tau)
        df[self.DATE] = start_obj + elapsed.apply(
            lambda x: timedelta(minutes=x)
        )
        # Place
        df[self.COUNTRY] = self.country
        df[self.PROVINCE] = self.province
        # Return the dataframe
        df = df.loc[:, [self.DATE, self.COUNTRY, self.PROVINCE, *var_cols]]
        return df
