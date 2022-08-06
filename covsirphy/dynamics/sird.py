#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.util.validator import Validator
from covsirphy.dynamics.ode import ODEModel


class SIRDModel(ODEModel):
    """Class of SIR-D model.

    Args:
        date_range (tuple of (str, str)): start date and end date of simulation
        tau (int): tau value [min]
        initial_dict (dict of {str: int}): initial values
            - Susceptible (int): the number of susceptible cases
            - Infected (int): the number of infected cases
            - Fatal (int): the number of fatal cases
            - Recovered (int): the number of recovered cases
        param_dict (dict of {str: float}): non-dimensional parameter values
            - kappa: non-dimensional mortality rate
            - rho: non-dimensional effective contact rate
            - sigma: non-dimensional recovery rate
    """
    # Name of ODE model
    _NAME = "SIR-D Model"
    # Variables
    _VARIABLES = [ODEModel.S, ODEModel.CI, ODEModel.R, ODEModel.F]
    # Non-dimensional parameters
    _PARAMETERS = ["kappa", "rho", "sigma"]
    # Dimensional parameters
    _DAY_PARAMETERS = ["1/alpha2 [day]", "1/beta [day]", "1/gamma [day]"]
    # Variables that increases monotonically
    _VARS_INCREASE = [ODEModel.R, ODEModel.F]
    # Sample data
    _SAMPLE_DICT = {
        "initial_dict": {ODEModel.S: 999_000, ODEModel.CI: 1000, ODEModel.R: 0, ODEModel.F: 0},
        "param_dict": {"kappa": 0.005, "rho": 0.2, "sigma": 0.075}
    }

    def __init__(self, date_range, tau, initial_dict, param_dict):
        super().__init__(date_range, tau, initial_dict, param_dict)
        self._kappa = Validator(self._param_dict["kappa"], "kappa", accept_none=False).float(value_range=(0, 1))
        self._rho = Validator(self._param_dict["rho"], "rho", accept_none=False).float(value_range=(0, 1))
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
        s, i, *_ = X
        dsdt = 0 - self._rho * s * i / n
        drdt = self._sigma * i
        dfdt = self._kappa * i
        didt = 0 - dsdt - drdt - dfdt
        return np.array([dsdt, didt, drdt, dfdt])

    @classmethod
    def transform(cls, data, tau=None):
        """Transform a dataframe, converting Susceptible/Infected/Fatal/Recovered to model-specific variables.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index or pandas.DatetimeIndex (when tau is not None)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
            tau (int or None): tau value [min]

        Returns:
            pandas.DataFrame:
                Index
                    as the same as index if @data when @tau is None else converted to time(x) = (TIME(x) - TIME(0)) / tau
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = Validator(data, "data").dataframe(columns=cls._SIRF)
        df.index = cls._date_to_non_dim(df.index, tau=tau)
        return df.loc[:, cls._VARIABLES].convert_dtypes()

    @classmethod
    def inverse_transform(cls, data, tau=None, start_date=None):
        """Transform a dataframe, converting model-specific variables to Susceptible/Infected/Fatal/Recovered.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    - Susceptible (int): the number of susceptible cases
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
        return df.loc[:, cls._SIRF].convert_dtypes()

    def r0(self):
        """Calculate basic reproduction number.

        Raises:
            ZeroDivisionError: sigma + kappa value was over 0

        Returns:
            float: reproduction number of the ODE model and parameters
        """
        try:
            return round(self._rho / (self._sigma + self._kappa), 2)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Sigma + kappa must be over 0 to calculate reproduction number with {self._NAME}.") from None

    def dimensional_parameters(self):
        """Calculate dimensional parameter values.

        Raises:
            ZeroDivisionError: either kappa or rho or sigma value was over 0

        Returns:
            dict of {str: int}: dictionary of dimensional parameter values
                - "1/alpha2 [day]" (int): mortality period
                - "1/beta [day]" (int): infection period
                - "1/gamma [day]" (int): recovery period
        """
        try:
            return {
                "1/alpha2 [day]": round(self._tau / 24 / 60 / self._kappa),
                "1/beta [day]": round(self._tau / 24 / 60 / self._rho),
                "1/gamma [day]": round(self._tau / 24 / 60 / self._sigma)
            }
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Kappa, rho and sigma must be over 0 to calculate dimensional parameters with {self._NAME}.") from None

    @classmethod
    def _param_quantile(cls, data, q=0.5):
        """With combinations (X, dX/dt) for X=S, I, R, D, calculate quantile values of ODE parameters.

        Args:
            data (pandas.DataFrame): transformed data with covsirphy.SIRDModel.transform(data=data, tau=tau)
            q (float or array-like): the quantile(s) to compute, value(s) between (0, 1)

        Returns:
            dict of {str: float or pandas.Series}: parameter values at the quantile(s)

        Note:
            We can get approximate parameter values with difference equations as follows.
                - kappa = (dF/dt) / I
                - rho = - n * (dS/dt) / S / I
                - sigma = (dR/dt) / I
        """
        df = data.copy()
        periods = round((df.index.max() - df.index.min()) / len(df))
        # Remove negative values and set variables
        df = df.loc[(df[cls.S] > 0) & (df[cls.CI] > 0)]
        n = df.loc[df.index[0], cls._VARIABLES].sum()
        # Calculate parameter values with non-dimensional difference equation
        kappa_series = df[cls.F].diff() / periods / df[cls.CI]
        rho_series = 0 - n * df[cls.S].diff() / periods / df[cls.S] / df[cls.CI]
        sigma_series = df[cls.R].diff() / periods / df[cls.CI]
        # Guess representative values
        return {
            "kappa": cls._clip(kappa_series.quantile(q=q), 0, 1),
            "rho": cls._clip(rho_series.quantile(q=q), 0, 1),
            "sigma": cls._clip(sigma_series.quantile(q=q), 0, 1),
        }

    @classmethod
    def sr(cls, data):
        """Return log10(S) and R of model-specific variables for S-R trend analysis.

        Args:
            data (pandas.DataFrame):
                Index
                    Date (pd.Timestamp): Observation date
                Columns
                    Susceptible (int): the number of susceptible cases
                    Infected (int): the number of currently infected cases
                    Recovered (int): the number of recovered cases
                    Fatal (int): the number of fatal cases

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
