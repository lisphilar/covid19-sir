#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.util.error import deprecate
from covsirphy.ode.mbase import ModelBase


class SIR(ModelBase):
    """
    SIR model.

    Args:
        population (int): total population
        rho (float)
        sigma (float)
    """
    # Model name
    NAME = "SIR"
    # names of parameters
    PARAMETERS = ["rho", "sigma"]
    DAY_PARAMETERS = ["1/beta [day]", "1/gamma [day]"]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.FR
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([1, 1, 1])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.FR]
    # Example set of parameters and initial values
    EXAMPLE = {
        ModelBase.STEP_N: 180,
        ModelBase.N.lower(): 1_000_000,
        ModelBase.PARAM_DICT: {
            "rho": 0.2, "sigma": 0.075,
        },
        ModelBase.Y0_DICT: {
            ModelBase.S: 999_000, ModelBase.CI: 1000, ModelBase.FR: 0,
        },
    }

    def __init__(self, population, rho, sigma):
        # Total population
        self.population = self._ensure_population(population)
        # Non-dim parameters
        self.rho = rho
        self.sigma = sigma
        self.non_param_dict = {"rho": rho, "sigma": sigma}

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, i, *_ = X
        dsdt = 0 - self.rho * s * i / n
        drdt = self.sigma * i
        didt = 0 - dsdt - drdt
        return np.array([dsdt, didt, drdt])

    @classmethod
    @deprecate(".param_range()", new=".guess()", version="2.19.1-zeta-fu1")
    def param_range(cls, taufree_df, population, quantiles=(0.1, 0.9)):
        """
        Deprecated. Define the value range of ODE parameters using (X, dX/dt) points.
        In SIR model, X is S, I and R here.

        Args:
            taufree_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - t (int): time steps (tau-free)
                    - columns with dimensional variables
            population (int): total population
            quantiles (tuple(int, int)): quantiles to cut, like confidence interval

        Returns:
            dict(str, tuple(float, float)): minimum/maximum values
        """
        cls._ensure_dataframe(taufree_df, name="taufree_df", columns=[cls.TS, *cls.VARIABLES])
        df = taufree_df.copy()
        df = df.loc[(df[cls.S] > 0) & (df[cls.CI] > 0)]
        n, t, s, i, r = population, df[cls.TS], df[cls.S], df[cls.CI], df[cls.FR]
        # rho = - n * (dS/dt) / S / I
        rho_series = 0 - n * s.diff() / t.diff() / s / i
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # Calculate quantile
        _dict = {
            k: tuple(v.quantile(quantiles).clip(0, 1))
            for (k, v) in zip(["rho", "sigma"], [rho_series, sigma_series])
        }
        return _dict

    @classmethod
    @deprecate(".specialize()", new=".convert()", version="2.19.1-zeta-fu1")
    def specialize(cls, data_df, population):
        """
        Deprecated. Specialize the dataset for this model.

        Args:
            data_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): total population in the place

        Returns:
            (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - any columns @data_df has
                        - Susceptible (int): the number of susceptible cases
                        - Fatal or Recovered (int): total number of fatal/recovered cases
        """
        cls._ensure_dataframe(data_df, name="data_df", columns=cls.VALUE_COLUMNS)
        df = data_df.copy()
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        df[cls.FR] = df[cls.F] + df[cls.R]
        return df

    @classmethod
    @deprecate(".restore()", new=".convert_reverse()", version="2.19.1-zeta-fu1")
    def restore(cls, specialized_df):
        """
        Deprecated. Restore Confirmed/Infected/Recovered/Fatal.
         using a dataframe with the variables of the model.

        Args:
            specialized_df (pandas.DataFrame): dataframe with the variables

                Index
                    reset index
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal or Recovered (int): the number of fatal/recovered cases
                    - any columns

        Returns:
            (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - the other columns @specialzed_df has
        """
        df = specialized_df.copy()
        other_cols = list(set(df.columns) - set(cls.VALUE_COLUMNS))
        df[cls.C] = df[cls.CI] + df[cls.FR]
        df[cls.F] = 0
        df[cls.R] = df[cls.FR]
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.

        Returns:
            float
        """
        try:
            rt = self.rho / self.sigma
        except ZeroDivisionError:
            return None
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.

        Args:
            param tau (int): tau value [min]

        Returns:
            dict[str, int]
        """
        try:
            return {
                "1/beta [day]": int(tau / 24 / 60 / self.rho),
                "1/gamma [day]": int(tau / 24 / 60 / self.sigma)
            }
        except ZeroDivisionError:
            return {p: None for p in self.DAY_PARAMETERS}

    @classmethod
    def convert(cls, data, tau):
        """
        Divide dates by tau value [min] and convert variables to model-specialized variables.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int): tau value [min] or None (skip division by tau values)

        Returns:
            pandas.DataFrame:
                Index
                    - Date (pd.Timestamp): Observation date (available when @tau is None)
                    - t (int): time steps (available when @tau is not None)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal or Recovered (int): the number of fatal/recovered cases
        """
        # Convert to tau-free if tau was specified
        df = cls._convert(data, tau)
        # Conversion of variables
        df[cls.FR] = df[cls.F] + df[cls.R]
        return df.loc[:, [cls.S, cls.CI, cls.FR]]

    @classmethod
    def convert_reverse(cls, converted_df, start, tau):
        """
        Calculate date with tau and start date, and restore Susceptible/Infected/"Fatal or Recovered".

        Args:
            converted_df (pandas.DataFrame):
                Index
                    t: Dates divided by tau value (time steps)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal or Recovered (int): the number of fatal/recovered cases
            start (pd.Timestamp): start date of simulation, like 14Apr2021
            tau (int): tau value [min]

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        # Calculate date with tau and start date
        df = cls._convert_reverse(converted_df, start, tau)
        # Conversion of variables
        df[cls.F] = 0
        df[cls.R] = df.loc[:, cls.FR]
        return df.loc[:, [cls.DATE, cls.S, cls.CI, cls.F, cls.R]]

    @classmethod
    def guess(cls, data, tau, q=0.5):
        """
        With (X, dX/dt) for X=S, I, R, guess parameter values.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int): tau value [min]
            q (float or tuple(float,)): the quantile(s) to compute, value(s) between (0, 1)

        Returns:
            dict(str, float or pandas.Series): guessed parameter values with the quantile(s)

        Note:
            We can guess parameter values with difference equations as follows.
            - rho = - n * (dS/dt) / S / I
            - sigma = (dR/dt) / I
        """
        # Convert to tau-free and model-specialized dataset
        df = cls.convert(data=data, tau=tau)
        # Remove negative values and set variables
        df = df.loc[(df[cls.S] > 0) & (df[cls.CI] > 0)]
        n = df.loc[df.index[0], [cls.S, cls.CI, cls.FR]].sum()
        # Calculate parameter values with difference equation and tau-free data
        rho_series = 0 - n * df[cls.S].diff() / df[cls.S] / df[cls.CI]
        sigma_series = df[cls.FR].diff() / df[cls.CI]
        # Guess representative values
        return {
            "rho": cls._clip(rho_series.quantile(q=q), 0, 1),
            "sigma": cls._clip(sigma_series.quantile(q=q), 0, 1),
        }
