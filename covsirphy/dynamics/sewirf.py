#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.util.validator import Validator
from covsirphy.dynamics.ode import ODEModel


class SEWIRFModel(ODEModel):
    """Class of SEWIR-F model.

    Args:
        date_range (tuple of (str, str)): start date and end date of simulation
        tau (int): tau value [min]
        initial_dict (dict of {str: int}): initial values
            - Susceptible (int): the number of susceptible cases
            - Exposed (int): the number of cases who are exposed and in latent period without infectivity
            - Waiting (int): the number of cases who are waiting for confirmation diagnosis with infectivity
            - Infected (int): the number of infected cases
            - Fatal (int): the number of fatal cases
            - Recovered (int): the number of recovered cases
        param_dict (dict of {str: float}): non-dimensional parameter values
            - theta: direct fatality probability of un-categorized confirmed cases
            - kappa: non-dimensional mortality rate
            - rho1: non-dimensional exposure rate (the number of encounter with the virus in a minute)
            - rho2: non-dimensional inverse value of latent period
            - rho3: non-dimensional inverse value of waiting time for confirmation
            - sigma: non-dimensional recovery rate
    """
    # Name of ODE model
    _NAME = "SEWIR-F Model"
    # Variables
    _VARIABLES = [ODEModel.S, ODEModel.E, ODEModel.W, ODEModel.CI, ODEModel.R, ODEModel.F]
    # Non-dimensional parameters
    _PARAMETERS = ["theta", "kappa", "rho1", "rho2", "rho3", "sigma"]
    # Dimensional parameters
    _DAY_PARAMETERS = [
        "alpha1 [-]", "1/alpha2 [day]", "1/beta1 [day]", "1/beta2 [day]", "1/beta3 [day]", "1/gamma [day]"]
    # Variables that increases monotonically
    _VARS_INCREASE = [ODEModel.R, ODEModel.F]
    # Sample data
    _SAMPLE_DICT = {
        "initial_dict": {
            ODEModel.S: 994_000, ODEModel.E: 3_000, ODEModel.W: 2_000, ODEModel.CI: 1000, ODEModel.R: 0, ODEModel.F: 0},
        "param_dict": {
            "theta": 0.002, "kappa": 0.005, "rho1": 0.2, "sigma": 0.075, "rho2": 0.167, "rho3": 0.167}
    }

    def __init__(self, date_range, tau, initial_dict, param_dict):
        super().__init__(date_range, tau, initial_dict, param_dict)
        self._theta = Validator(self._param_dict["theta"], "theta", accept_none=False).float(value_range=(0, 1))
        self._kappa = Validator(self._param_dict["kappa"], "kappa", accept_none=False).float(value_range=(0, 1))
        self._rho1 = Validator(self._param_dict["rho1"], "rho1", accept_none=False).float(value_range=(0, 1))
        self._rho2 = Validator(self._param_dict["rho2"], "rho2", accept_none=False).float(value_range=(0, 1))
        self._rho3 = Validator(self._param_dict["rho3"], "rho3", accept_none=False).float(value_range=(0, 1))
        self._sigma = Validator(self._param_dict["sigma"], "sigma", accept_none=False).float(value_range=(0, 1))

    def _discretize(self, t, X):
        """Discretize the ODE.

        Args:
            t (int): discrete time-steps
            X (numpy.array): the current values of the model

        Returns:
            numpy.array: the next values of the model
        """
        n = self._population
        s, i, *_, e, w = X
        beta_swi = self._rho1 * s * (w + i) / n
        dsdt = 0 - beta_swi
        dedt = beta_swi - self._rho2 * e
        dwdt = self._rho2 * e - self._rho3 * w
        drdt = self._sigma * i
        dfdt = self._kappa * i + self._theta * self._rho3 * w
        didt = 0 - dsdt - drdt - dfdt - dedt - dwdt
        return np.array([dsdt, didt, drdt, dfdt, dedt, dwdt])

    @ classmethod
    def transform(cls, data, tau=None):
        """Transform a dataframe, converting Susceptible/Infected/Fatal/Recovered to model-specific variables.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index or pandas.DatetimeIndex (when tau is not None)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int or None): tau value [min]

        Returns:
            pandas.DataFrame:
                Index
                    as the same as index if @data when @tau is None else converted to time(x) = (TIME(x) - TIME(0)) / tau
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Exposed (int): the number of cases who are exposed and in latent period without infectivity
                    - Waiting (int): the number of cases who are waiting for confirmation diagnosis with infectivity
                    - Infected (int): the number of infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
        """
        df = Validator(data, "data").dataframe(columns=cls._SIRF)
        df.index = cls._date_to_non_dim(df.index, tau=tau)
        df[cls.E] = 0
        df[cls.W] = 0
        return df.loc[:, cls._VARIABLES].convert_dtypes()

    @ classmethod
    def inverse_transform(cls, data, tau=None, start_date=None):
        """Transform a dataframe, converting model-specific variables to Susceptible/Infected/Fatal/Recovered.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Exposed (int): the number of cases who are exposed and in latent period without infectivity
                    - Waiting (int): the number of cases who are waiting for confirmation diagnosis with infectivity
                    - Infected (int): the number of infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
            tau (int or None): tau value [min]
            start_date (str or pandas.Timestamp or None): start date of records ie. TIME(0)

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.DatetimeIndex) or as-is @data (when either @tau or @start_date are None the index @data is date)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
        """
        df = Validator(data, "data").dataframe(columns=cls._VARIABLES)
        df = cls._non_dim_to_date(data=df, tau=tau, start_date=start_date)
        df[cls.S] = df[cls.S] + df[cls.E] + df[cls.W]
        return df.loc[:, cls._SIRF].convert_dtypes()

    def r0(self):
        """Calculate basic reproduction number.

        Raises:
            ZeroDivisionError: rho2 or sigma + kappa value was over 0

        Returns:
            float: reproduction number of the ODE model and parameters
        """
        try:
            rho = self._rho1 / self._rho2 * self._rho3
            rt = rho * (1 - self._theta) / (self._sigma + self._kappa)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Both of 'rho2' and 'sigma + kappa' must be over 0 to calculate reproduction number with {self._NAME}.") from None
        return round(rt, 2)

    def dimensional_parameters(self):
        """Calculate dimensional parameter values.

        Raises:
            ZeroDivisionError: either kappa or rho_i for i=1,2,3 or sigma value was over 0

        Returns:
            dict of {str: int}: dictionary of dimensional parameter values
                - "alpha1 [-]" (float): direct fatality probability of un-categorized confirmed cases
                - "1/alpha2 [day]" (int): mortality period of infected cases
                - "1/beta1 [day]" (int): period for susceptible people to encounter with the virus
                - "1/beta2 [day]" (int): latent period
                - "1/beta3 [day]" (int): waiting time for confirmation
                - "1/gamma [day]" (int): recovery period
        """
        try:
            return {
                "alpha1 [-]": round(self._theta, 3),
                "1/alpha2 [day]": round(self._tau / 24 / 60 / self._kappa),
                "1/beta1 [day]": round(self._tau / 24 / 60 / self._rho1),
                "1/beta2 [day]": round(self._tau / 24 / 60 / self._rho2),
                "1/beta3 [day]": round(self._tau / 24 / 60 / self._rho3),
                "1/gamma [day]": round(self._tau / 24 / 60 / self._sigma)
            }
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Kappa, rho_i for i=1,2,3 and sigma must be over 0 to calculate dimensional parameters with {self._NAME}.") from None

    @classmethod
    def from_data_with_quantile(cls, *args, **kwargs):
        """Initialize model with data, estimating ODE parameters with quantiles.

        Raises:
            NotImplementedError: this model cannot be used for parameter estimation because Exposed/Waiting data is un-available
        """
        raise NotImplementedError(
            "SEWIR-F model cannot be used for parameter estimation because we do not have records "
            "of Exposed and Waiting. Please use SIR-F model with `covsirphy.SIRFModel` class."
        )

    @classmethod
    def from_data_with_optimization(cls, *args, **kwargs):
        """Initialize model with data, estimating ODE parameters hyperparameter optimization using Optuna.

        Raises:
            NotImplementedError: this model cannot be used for parameter estimation because Exposed/Waiting data is un-available
        """
        raise NotImplementedError(
            "SEWIR-F model cannot be used for parameter estimation because we do not have records "
            "of Exposed and Waiting. Please use SIR-F model with `covsirphy.SIRFModel` class."
        )

    @classmethod
    def sr(cls, data):
        """Return log10(S) and R of model-specific variables for S-R trend analysis.

        Args:
            data (pandas.DataFrame):
                Index
                    - Date (pd.Timestamp): Observation date
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): date
                Columns
                    log10(S) (np.float64): common logarithm of Susceptible
                    R (np.int64): Recovered
        """
        Validator(data, "data", accept_none=False).dataframe(time_index=True, columns=cls._SIRF)
        df = data.rename(columns={cls.R: cls._r})
        df[cls._logS] = np.log10(df[cls.S])
        return df.loc[:, [cls._logS, cls._r]].astype({cls._logS: np.float64, cls._r: np.int64})
