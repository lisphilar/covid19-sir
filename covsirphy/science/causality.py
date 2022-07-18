#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.validator import Validator
from covsirphy.science.ode_scenario import ODEScenario


class Causality(ODEScenario):
    """Perform scenario analysis, considering causality of ODE parameters and indicators.

    Args:
        data (pandas.DataFrame): actual data of the number of cases
            Index
                Date (pandas.Timestamp): observation dates
            Columns
                Population (int): total population
                Confirmed (int): the number of confirmed cases
                Fatal (int): the number of fatal cases
                Recovered (int): the number of recovered cases, must be over 0
                Susceptible (int): the number of susceptible cases, will be ignored because overwritten
                Infected (int): the number of currently infected cases, will be ignored because overwritten
                the other columns will be ignored
        location_name (str): name to identify the location to show in figure titles
        complement (bool): perform data complement with covsirphy.DataEngineer().subset(complement=True) or not

    Note:
        Data cleaning will be performed with covsirphy.DataEngineer().clean() automatically.
    """

    def predict(self, days, name, X=None, **kwargs):
        """Create scenarios and append a phase, performing multivariate/univariate prediction of ODE parameters.

        Args:
            days (int): days to predict
            name (str): scenario name
            X (pandas.DataFrame or None): data of indicators or None (univariate prediction)
                Index
                    pandas.Timestamp: Observation date
                Columns
                    observed variables (int or float)
            **kwargs: keyword arguments of autots.AutoTS()

        Return:
            covsirphy.ODEScenario: self
        """
        if X is None:
            self.predict(days=days, name=name, **kwargs)
        track_df = self.to_dynamics(name=name).track()
        model = self._snr_alias.find(name=name)[self.ODE]
        Y = track_df.loc[:, model._PARAMETERS]
        X = Validator(X, "X").dataframe(time_index=True, empty_ok=False)
        return self._predict(days=days, name=name, X=X, Y=Y, method="multivariate_regression", **kwargs)
