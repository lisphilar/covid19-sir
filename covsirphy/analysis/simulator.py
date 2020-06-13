#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from covsirphy.cleaning.word import Word
from covsirphy.ode.mbase import ModelBase


class ODESimulator(Word):
    """
    Simulation of an ODE model.
    """

    def __init__(self, country, province="-"):
        """
        @country <str>: country name
        @province <str>: province name
        """
        self.country = country
        self.province = province
        # list of dictionary
        # key: model, step_n, population, param_dict, y0_dict
        self.settings = list()
        # TODO: not use non-dimensional data
        self._nondim_df = pd.DataFrame()
        self._dim_df = pd.DataFrame()

    def add(self, model, step_n, population, param_dict=None, y0_dict=None):
        """
        Add models to the simulator.
        @model <subclass of cs.ModelBase>: the first ODE model
        @step_n <int>: the number of steps
        @population <int>: population in the place
        @param_dict <dict[str]=float>:
            - dictionary of parameter values or None
            - if not include some params, the last values will be used
                - NameError when the model is the first model
                - NameError if new params are included
        @y0_dict <dict[str]=float>:
            - dictionary of initial values or None (non-dimensional)
            - if not include some variables, the last values will be used
                - NameError when the model is the first model
                - NameError if new variable are included
        @return self
        """
        # Check model
        if not issubclass(model, ModelBase):
            raise TypeError(
                "@model must be an ODE model <sub-class of cs.ModelBase>."
            )
        # Check step number
        if not isinstance(step_n, int) or step_n < 1:
            raise TypeError(
                f"@step_n must be a non-negative integer, but {step_n} was applied."
            )
        # Check population value
        if not isinstance(step_n, int) or step_n < 1:
            raise TypeError("@population must be a non-negative integer.")
        # Check param values
        param_dict = dict() if param_dict is None else param_dict
        for param in model.PARAMETERS:
            if param in param_dict.keys():
                continue
            if (not self.settings) or (param not in self.settings[-1]["model"].PARAMETERS):
                s = f"{param} value must be specified in @param_dict."
                raise NameError(s)
            param_dict[param] = self.settings[-1]["param_dict"][param]
        # Check initial values
        y0_dict = dict() if y0_dict is None else y0_dict
        for var in model.VARIABLES:
            if var in y0_dict.keys():
                continue
            if (not self.settings) or (var not in self.settings[-1]["model"].VARIABLES):
                s = f"Initial value of {var} must be specified in @y0_dict."
                raise NameError(s)
            # Will use the last values the last phase
            y0_dict[var] = None
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
        return self

    def _solve_ode(self, model, step_n, param_dict, y0_dict):
        """
        Solve ODE of the model.
        @model <subclass of cs.ModelBase>: the ODE model
        @step_n <int>: the number of steps
        @param_dict <dict[str]=float>: dictionary of parameter values
        @y0_dict <dict[str]=float>: dictionary of initial values
        @return <pd.DataFrame>:
            - index: reset index
            - t: time steps, 0, 1, 2, 3...
            - x, y, z, w etc.
                - calculated in child classes.
                - non-dimensionalized variables of Susceptible etc.
        """
        # TODO: not use non-dimensional data
        tstart, dt, tend = 0, 1, step_n
        variables = model.VARIABLES[:]
        initials = [y0_dict[var] for var in variables]
        sol = solve_ivp(
            fun=model(**param_dict),
            t_span=[tstart, tend],
            y0=np.array(initials, dtype=np.float64),
            t_eval=np.arange(tstart, tend + dt, dt),
            dense_output=False
        )
        t_df = pd.Series(data=sol["t"], name="t")
        y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=variables)
        sim_df = pd.concat([t_df, y_df], axis=1)
        # y in non-dimensional model must be over 0
        sim_df.loc[sim_df["y"] < 0, "y"] = 0
        return sim_df

    def run(self):
        """
        Run the simulator.
        """
        # TODO: not use non-dimensional data
        self._nondim_df = pd.DataFrame()
        for setting in self.settings:
            model = setting["model"]
            population = setting["population"]
            # Initial values
            y0_dict = setting["y0_dict"]
            if None in y0_dict.values():
                for (k, v) in y0_dict.items():
                    if v is not None:
                        continue
                    y0_dict[k] = self._nondim_df.loc[
                        self._nondim_df.index[-1], k
                    ]
            # Non-dimensional
            nondim_df = self._solve_ode(
                model=model,
                step_n=setting["step_n"],
                param_dict=setting["param_dict"],
                y0_dict=y0_dict
            )
            nondim_df.fillna(0)
            self._nondim_df = pd.concat(
                [self._nondim_df.iloc[:-1, :], nondim_df],
                axis=0, ignore_index=True
            )
            self._nondim_df["t"] = self._nondim_df.index
            # Dimensional
            dim_df = model.calc_variables_reverse(nondim_df, population)
            self._dim_df = pd.concat(
                [self._dim_df, dim_df], axis=0, ignore_index=True, sort=True
            )
        return self

    def non_dim(self):
        """
        Return the non-dimensionalized results.
        @return <pd.DataFrame>:
            - index: reset index
            - t: time steps, 0, 1, 2, 3...
            - x, y, z, w etc.
                - calculated in child classes.
                - non-dimensionalized variables of Susceptible etc.
        """
        # TODO: not use non-dimensional data
        df = self._nondim_df.copy()
        df = df.reset_index(drop=True)
        return df

    def dim(self, tau, start_date):
        """
        Return the dimensionalized results.
        @tau <int>: tau value [min]
        @start_date <str>: start date of the records, like 22Jan2020
        @return <pd.DataFrame>
            - index <int>: reset index
            - Date <pd.TimeStamp>: Observation date
            - Country <str>: country/region name
            - Province <str>: province/prefecture/state name
            - variables of the models <int>: Confirmed <int> etc.
        """
        df = self._dim_df.copy()
        cols = df.columns.tolist()
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
        df = df.loc[:, [self.DATE, self.COUNTRY, self.PROVINCE, *cols]]
        df = df.reset_index(drop=True)
        return df
