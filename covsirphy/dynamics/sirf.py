from __future__ import annotations
import numpy as np
import pandas as pd
from covsirphy.util.validator import Validator
from covsirphy.dynamics.sird import SIRDModel


class SIRFModel(SIRDModel):
    """Class of SIR-F model.

    Args:
        date_range: start date and end date of simulation
        tau: tau value [min]
        initial_dict: initial values

            - Susceptible (int): the number of susceptible cases
            - Infected (int): the number of infected cases
            - Fatal (int): the number of fatal cases
            - Recovered (int): the number of recovered cases
        param_dict: non-dimensional parameter values

            - theta: direct fatality probability of un-categorized confirmed cases
            - kappa: non-dimensional mortality rate of infected cases
            - rho: non-dimensional effective contact rate
            - sigma: non-dimensional recovery rate

    Note:
        SIR-F model is original to Covsirphy, https://www.kaggle.com/code/lisphilar/covid-19-data-with-sir-model/notebook
    """
    # Name of ODE model
    _NAME = "SIR-F Model"
    # Non-dimensional parameters
    _PARAMETERS = ["theta", "kappa", "rho", "sigma"]
    # Dimensional parameters
    _DAY_PARAMETERS = ["alpha1 [-]", "1/alpha2 [day]", "1/beta [day]", "1/gamma [day]"]
    # Sample data
    _SAMPLE_DICT = {
        "initial_dict": {SIRDModel.S: 999_000, SIRDModel.CI: 1000, SIRDModel.R: 0, SIRDModel.F: 0},
        "param_dict": {"theta": 0.002, "kappa": 0.005, "rho": 0.2, "sigma": 0.075}
    }

    def __init__(self, date_range: tuple[str, str], tau: int, initial_dict: dict[str, int], param_dict: dict[str, float]) -> None:
        super().__init__(date_range, tau, initial_dict, param_dict)
        self._theta = Validator(self._param_dict["theta"], "theta", accept_none=False).float(value_range=(0, 1))

    def _discretize(self, t: int, X: np.ndarray) -> np.ndarray:
        """Discretize the ODE.

        Args:
            t: discrete time-steps
            X: the current values of the model

        Returns:
            numpy.array: the next values of the model
        """
        n = self._population
        s, i, *_ = X
        dsdt = 0 - self._rho * s * i / n
        drdt = self._sigma * i
        dfdt = self._kappa * i + (0 - dsdt) * self._theta
        didt = 0 - dsdt - drdt - dfdt
        return np.array([dsdt, didt, drdt, dfdt])

    def r0(self) -> float:
        """Calculate basic reproduction number.

        Raises:
            ZeroDivisionError: sigma + kappa value was over 0

        Returns:
            reproduction number of the ODE model and parameters
        """
        try:
            return round(self._rho * (1 - self._theta) / (self._sigma + self._kappa), 2)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Sigma + kappa must be over 0 to calculate reproduction number with {self._NAME}.") from None

    def dimensional_parameters(self) -> dict[str, float | int]:
        """Calculate dimensional parameter values.

        Raises:
            ZeroDivisionError: either kappa or rho or sigma value was over 0

        Returns:
            dictionary of dimensional parameter values
                - "alpha1 [-]" (float): direct fatality probability of un-categorized confirmed cases
                - "1/alpha2 [day]" (int): mortality period of infected cases
                - "1/beta [day]" (int): infection period
                - "1/gamma [day]" (int): recovery period
        """
        try:
            return {
                "alpha1 [-]": round(self._theta, 3),
                "1/alpha2 [day]": round(self._tau / 24 / 60 / self._kappa),
                "1/beta [day]": round(self._tau / 24 / 60 / self._rho),
                "1/gamma [day]": round(self._tau / 24 / 60 / self._sigma)
            }
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Kappa, rho and sigma must be over 0 to calculate dimensional parameters with {self._NAME}.") from None

    @classmethod
    def _param_quantile(cls, data: pd.DataFrame, q: float | pd.Series = 0.5) -> dict[str, float | pd.Series]:
        """With combinations (X, dX/dt) for X=S, I, R, F, calculate quantile values of ODE parameters.

        Args:
            data: transformed data with covsirphy.SIRFModel.transform(data=data, tau=tau)
            q: the quantile(s) to compute, value(s) between (0, 1)

        Returns:
            parameter values at the quantile(s)

        Note:
            We can get approximate parameter values with difference equations as follows.

                - theta -> +0 (i.e. around 0 and not negative)
                - kappa -> (dF/dt) / I when theta -> +0
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
            "theta": 0.0 if isinstance(q, float) else pd.Series([0.0, 0.5]).repeat([1, len(q) - 1]),
            "kappa": cls._clip(kappa_series.quantile(q=q), 0, 1),
            "rho": cls._clip(rho_series.quantile(q=q), 0, 1),
            "sigma": cls._clip(sigma_series.quantile(q=q), 0, 1),
        }

    @classmethod
    def sr(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Return log10(S) and R of model-specific variables for S-R trend analysis.

        Args:
            data:
                Index
                    Date (pd.Timestamp): Observation date
                Columns
                    Susceptible (int): the number of susceptible cases
                    Infected (int): the number of currently infected cases
                    Recovered (int): the number of recovered cases
                    Fatal (int): the number of fatal cases

        Returns:
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
