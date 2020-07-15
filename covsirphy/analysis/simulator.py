#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase


class ODESimulator(Term):
    """
    Simulation of an ODE model.

    Args:
        country (str): country name
        province (str): province name
    """

    def __init__(self, country, province="-"):
        self.country = country
        self.province = province
        # list of dictionary
        # keys: model, step_n, population, param_dict, y0_dict
        self.settings = list()
        # tau-free data: reset index, 't', columns with dimensional variables
        self._taufree_df = pd.DataFrame()
        # key: non-dim variable name, value: dimensional variable name
        self.var_dict = dict()

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

        Returns:
            self
        """
        # Validate the arguments
        model = self.validate_subclass(model, ModelBase, name="model")
        step_n = self.validate_natural_int(step_n, name="step_n")
        population = self.validate_natural_int(population, name="population")
        # Check param values
        param_dict = param_dict or dict()
        for param in model.PARAMETERS:
            if param in param_dict.keys():
                continue
            if (not self.settings) or (param not in self.settings[-1]["model"].PARAMETERS):
                s = f"{param} value must be specified in @param_dict."
                raise NameError(s)
            param_dict[param] = self.settings[-1]["param_dict"][param]
        # Check initial values
        y0_dict = self._validate_initial_values(model, y0_dict)
        # Register the setting
        self.settings.append(
            {
                "model": model,
                "step_n": step_n,
                "population": population,
                "param_dict": param_dict.copy(),
                "y0_dict": y0_dict.copy(),
            }
        )
        # Update variable dictionary
        self.var_dict.update(model.VAR_DICT)
        return self

    def _validate_initial_values(self, model, y0_dict):
        """
        Validate the dictionary of initial values.

        Args:
            model (subclass of cs.ModelBase): the ODE model
            y0_dict (dict): dictionary of initial values
                - key (str): dimensional variable name
                - value (int):initial value of the variable

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        y0_dict = y0_dict or dict()
        for var in model.VARIABLES:
            if var in y0_dict.keys():
                continue
            if (not self.settings) or (var not in self.settings[-1]["model"].VARIABLES):
                s = f"Initial value of {var} must be specified in @y0_dict."
                raise NameError(s)
            # Will use the last values of the last phase
            y0_dict[var] = None
        return y0_dict

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
        step_n = self.validate_natural_int(step_n, name="step_n")
        population = self.validate_natural_int(population, name="population")
        model = self.validate_subclass(model, ModelBase, name="model")
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
        sim_df = pd.concat([t_df, y_df], axis=1)
        return sim_df

    def run(self):
        """
        Run the simulator.
        """
        self._taufree_df = pd.DataFrame()
        for setting in self.settings:
            # Initial values
            y0_dict = setting["y0_dict"].copy()
            if None in y0_dict.values():
                keys_with_none = [k for (k, v) in y0_dict.items() if v is None]
                if keys_with_none and self._taufree_df.empty:
                    raise NameError(
                        "Initial values of simulation must be specified in advance."
                    )
                last_value_dict = {
                    k: self._taufree_df.loc[self._taufree_df.index[-1], k]
                    for k in keys_with_none
                }
                y0_dict.update(last_value_dict)
            setting["y0_dict"] = y0_dict.copy()
            # Solve ODEs
            new_df = self._solve_ode(**setting)
            taufree_df = pd.concat(
                [self._taufree_df.iloc[:-1, :], new_df],
                axis=0, ignore_index=True
            )
            taufree_df = taufree_df.fillna(0)
            taufree_df[self.TS] = taufree_df.index
            self._taufree_df = taufree_df.copy()
        return self

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
        df = self._taufree_df.copy()
        if df.empty:
            raise Exception("ODESimulator.run() must be done in advance.")
        df = df.reset_index(drop=True)
        return df

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
        df = df.drop(self.TS, axis=1)
        df = df.reset_index(drop=True)
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
