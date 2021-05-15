#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.util.error import deprecate
from covsirphy.ode.mbase import ModelBase


class SIRD(ModelBase):
    """
    SIR-D model.

    Args:
        population (int): total population
        kappa (float)
        rho (float)
        sigma (float)
    """
    # Model name
    NAME = "SIR-D"
    # names of parameters
    PARAMETERS = ["kappa", "rho", "sigma"]
    DAY_PARAMETERS = ["1/alpha2 [day]", "1/beta [day]", "1/gamma [day]"]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([1, 10, 10, 2])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.R, ModelBase.F]
    # Example set of parameters and initial values
    EXAMPLE = {
        ModelBase.STEP_N: 180,
        ModelBase.N.lower(): 1_000_000,
        ModelBase.PARAM_DICT: {
            "kappa": 0.005, "rho": 0.2, "sigma": 0.075,
        },
        ModelBase.Y0_DICT: {
            ModelBase.S: 999_000, ModelBase.CI: 1000, ModelBase.R: 0, ModelBase.F: 0,
        },
    }

    def __init__(self, population, kappa, rho, sigma):
        # Total population
        self.population = self._ensure_population(population)
        # Non-dim parameters
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma
        self.non_param_dict = {"kappa": kappa, "rho": rho, "sigma": sigma}

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
        dfdt = self.kappa * i
        didt = 0 - dsdt - drdt - dfdt
        return np.array([dsdt, didt, drdt, dfdt])

    @classmethod
    @deprecate(".param_range()", new=".guess()", version="2.19.1-zeta-fu1")
    def param_range(cls, taufree_df, population, quantiles=(0.1, 0.9)):
        """
        Deprecated. Define the value range of ODE parameters using (X, dX/dt) points.
        In SIR model, X is S, I, R, D here.

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
        n, t = population, df[cls.TS]
        s, i, r, d = df[cls.S], df[cls.CI], df[cls.R], df[cls.F]
        # kappa = (dD/dt) / I
        kappa_series = d.diff() / t.diff() / i
        # rho = - n * (dS/dt) / S / I
        rho_series = 0 - n * s.diff() / t.diff() / s / i
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # Calculate quantile
        _dict = {
            k: tuple(v.quantile(quantiles).clip(0, 1))
            for (k, v) in zip(
                ["kappa", "rho", "sigma"],
                [kappa_series, rho_series, sigma_series]
            )
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
        """
        cls._ensure_dataframe(data_df, name="data_df", columns=cls.VALUE_COLUMNS)
        df = data_df.copy()
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        return df

    @classmethod
    @deprecate(".restore()", new=".convert_reverse()", version="2.19.1-zeta-fu1")
    def restore(cls, specialized_df):
        """
        Deprecated. Restore Confirmed/Infected/Recovered/Fatal using a dataframe with the variables of the model.

        Args:
        @specialized_df (pandas.DataFrame): dataframe with the variables

            Index
                reset index
            Columns
                - Susceptible (int): the number of susceptible cases
                - Infected (int): the number of currently infected cases
                - Recovered (int): the number of recovered cases
                - Fatal (int): the number of fatal cases
                - any columns

        Returns:
            (pandas.DataFrame)
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
        df[cls.C] = df[cls.CI] + df[cls.R] + df[cls.F]
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.

        Returns:
            float
        """
        try:
            rt = self.rho / (self.sigma + self.kappa)
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
                "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
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
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
        """
        # Convert to tau-free if tau was specified
        df = cls._convert(data, tau)
        # Conversion of variables: un-necessary for SIR-D model
        return df.loc[:, [cls.S, cls.CI, cls.R, cls.F]]

    @classmethod
    def convert_reverse(cls, converted_df, start, tau):
        """
        Calculate date with tau and start date, and restore Susceptible/Infected/Fatal/Recovered.

        Args:
            converted_df (pandas.DataFrame):
                Index
                    t: Dates divided by tau value (time steps)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
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
        # Conversion of variables: un-necessary for SIR-F model
        return df.loc[:, [cls.DATE, cls.S, cls.CI, cls.F, cls.R]]

    @classmethod
    def guess(cls, data, tau, q=0.5):
        """
        With (X, dX/dt) for X=S, I, R, D, guess parameter values.

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
            - kappa = (dF/dt) / I
            - rho = - n * (dS/dt) / S / I
            - sigma = (dR/dt) / I
        """
        # Convert to tau-free and model-specialized dataset
        df = cls.convert(data=data, tau=tau)
        # Remove negative values and set variables
        df = df.loc[(df[cls.S] > 0) & (df[cls.CI] > 0)]
        n = df.loc[df.index[0], [cls.S, cls.CI, cls.F, cls.R]].sum()
        # Calculate parameter values with difference equation and tau-free data
        kappa_series = df[cls.F].diff() / df[cls.CI]
        rho_series = 0 - n * df[cls.S].diff() / df[cls.S] / df[cls.CI]
        sigma_series = df[cls.R].diff() / df[cls.CI]
        # Guess representative values
        return {
            "kappa": cls._clip(kappa_series.quantile(q=q), 0, 1),
            "rho": cls._clip(rho_series.quantile(q=q), 0, 1),
            "sigma": cls._clip(sigma_series.quantile(q=q), 0, 1),
        }
