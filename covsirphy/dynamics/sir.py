#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.util.validator import Validator
from covsirphy.dynamics.ode import ODEModel


class SIRModel(ODEModel):
    """Class of SIR model.

    Args:
        date_range (tuple(str, str)): start date and end date of simulation
        tau (int): tau value [min]
        initial_dict (dict of {str: int}): initial values
            - Susceptible: the number of susceptible cases
            - Infected: the number of infected cases
            - Fatal or Recovered: the number of fatal or recovered cases
        param_dict (dict of {str: float}): non-dimensional parameter values
            - rho: non-dimensional effective contact rate
            - sigma: non-dimensional recovery plus mortality rate
    """
    # Name of ODE model
    _NAME = "SIR Model"
    # Variables
    _VARIABLES = [ODEModel.S, ODEModel.CI, ODEModel.FR]
    # Non-dimensional parameters
    _PARAMETERS = ["rho", "sigma"]
    # Dimensional parameters
    _DAY_PARAMETERS = ["1/beta [day]", "1/gamma [day]"]
    # Weights of variables in parameter estimation error function
    _WEIGHTS = np.array([1, 1, 1])
    # Variables that increases monotonically
    _VARS_INCREASE = [ODEModel.FR]
    # Sample data
    _SAMPLE_DICT = {
        "initial_dict": {ODEModel.S: 999_000, ODEModel.CI: 1000, ODEModel.FR: 0},
        "param_dict": {"rho": 0.2, "sigma": 0.075}
    }

    def __init__(self, date_range, tau, initial_dict, param_dict):
        super().__init__(date_range, tau, initial_dict, param_dict)
        self._rho, self._sigma = self._param_dict["rho"], self._param_dict["sigma"]

    def _discretize(self, t, X):
        """Discretize the ODE.

        Args:
            t (int): discrete time-steps
            X (numpy.array): the current values of the model

        Returns:
            numpy.array: the next values of the model
        """
        n = self._population
        s, i, *_ = X
        dsdt = 0 - self._rho * s * i / n
        drdt = self._sigma * i
        didt = 0 - dsdt - drdt
        return np.array([dsdt, didt, drdt])

    @classmethod
    def transform(cls, data):
        """Transform a dataframe, converting Susceptible/Infected/Fatal/Recovered to model-specific variables.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Returns:
            pandas.DataFrame:
                Index
                    as the same as index if @data
                Columns
                    - Susceptible: the number of susceptible cases
                    - Infected: the number of infected cases
                    - Fatal or Recovered: the number of fatal or recovered cases

        Note:
            This method must be defined by child classes.
        """
        df = Validator(data, "data").dataframe(columns=cls._SIFR)
        df[cls.FR] = df[cls.F] + df[cls.R]
        return df.loc[:, cls._VARIABLES]

    @classmethod
    def inverse_transform(cls, data):
        """Transform a dataframe, converting model-specific variables to Susceptible/Infected/Fatal/Recovered.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    - Susceptible: the number of susceptible cases
                    - Infected: the number of infected cases
                    - Fatal or Recovered: the number of fatal or recovered cases

        Returns:
            pandas.DataFrame:
                Index
                    as the same as index if @data
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Note:
            This method must be defined by child classes.
        """
        df = Validator(data, "data").dataframe(columns=cls._VARIABLES)
        df[cls.F] = 0
        df[cls.R] = df.loc[:, cls.FR]
        return df.loc[:, cls._SIFR]

    def r0(self):
        """Calculate basic reproduction number.

        Raises:
            ZeroDivisionError: Sigma value was over 0

        Returns:
            float: reproduction number of the ODE model and parameters
        """
        try:
            return round(self._rho / self._sigma, 2)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Sigma must be over 0 to calculate reproduction number with {self._NAME}.") from None

    def dimensional_parameters(self):
        """Calculate dimensional parameter values.

        Raises:
            ZeroDivisionError: either rho or sigma value was over 0

        Returns:
            dict of {str: int or float}: dictionary of dimensional parameter values
        """
        try:
            return {
                "1/beta [day]": round(self._tau / 24 / 60 / self._rho),
                "1/gamma [day]": round(self._tau / 24 / 60 / self._sigma)
            }
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Rho and sigma must be over 0 to calculate dimensional parameters with {self._NAME}.") from None
