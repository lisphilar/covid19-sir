#!/usr/bin/env python
# -*- coding: utf-8 -*-


from datetime import timedelta
import pandas as pd
from covsirphy.util.error import NAFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.dynamics.ode import ODEModel


class _Simulator(Term):
    """Perform simulation with phase-dependent ODE model.

    Args:
        model (covsirphy.ODEModel): definition of ODE model
        data (pandas.DataFrame): new data to overwrite the current information
            Index
                Date (pandas.Timestamp): Observation date
            Columns
                Phase_ID (int): identification number of phases
                Susceptible (int): the number of susceptible cases
                Infected (int): the number of currently infected cases
                Recovered (int): the number of recovered cases
                Fatal (int): the number of fatal cases
                (numpy.float64): ODE parameter values defined with model.PARAMETERS
    """

    def __init__(self, model, data):
        self._model = Validator(model, "model", accept_none=False).subclass(ODEModel)
        self._df = Validator(data, "data", accept_none=False).dataframe(
            time_index=True, columns=[self._PH, *self._SIRF, *model._PARAMETERS])
        self._first, self._last = data.index.min(), data.index.max()

    def run(self, tau, model_specific=False):
        """Perform simulation with phase-dependent ODE model.

        Args:
            tau (int): tau value [min]
            model_specific (bool): whether convert S, I, F, R to model-specific variables or not

        Raises:
            NAFoundError: ODE parameter values on the start dates of phases are un-set

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    Date (pd.Timestamp): Observation date
                    if @model_specific is False:
                    Susceptible (int): the number of susceptible cases
                    Infected (int): the number of currently infected cases
                    Recovered (int): the number of recovered cases
                    Fatal (int): the number of fatal cases
                    if @model_specific is True, variables defined by model.VARIABLES of covsirphy.Dynamics(model)
        """
        all_df = self._df.copy()
        all_df[self._PH], _ = all_df[self._PH].factorize()
        date_groupby = all_df.reset_index().groupby(self._PH)
        start_dates = date_groupby.first()[self.DATE].sort_values()
        end_dates = date_groupby.last()[self.DATE].sort_values()
        variables, parameters = self._SIRF, self._model._PARAMETERS[:]
        param_df = all_df.ffill().loc[:, parameters]
        # Simulation
        dataframes = []
        for (start, end) in zip(start_dates, end_dates):
            variable_df = all_df.loc[start: end + timedelta(days=1), variables]
            if variable_df.iloc[0].isna().any():
                variable_df = pd.DataFrame(
                    index=pd.date_range(start, end + timedelta(days=1), freq="D"), columns=self._SIRF)
                variable_df.update(self._model.inverse_transform(dataframes[-1]))
            variable_df.index.name = self.DATE
            if param_df.loc[start].isna().any():
                raise NAFoundError(
                    f"ODE parameter values on {start.strftime(self.DATE_FORMAT)}", value=param_df.loc[start].to_dict(),
                    details="Please set values with .register() or .estimate_params()")
            param_dict = param_df.loc[start].to_dict()
            instance = self._model.from_data(data=variable_df.reset_index(), param_dict=param_dict, tau=tau)
            dataframes.append(instance.solve())
        # Combine results of phases
        df = pd.concat(dataframes, axis=0).groupby(level=0).first().loc[self._first: self._last]
        if model_specific:
            return df.reset_index().convert_dtypes()
        df = self._model.inverse_transform(data=df)
        df.index.name = self.DATE
        return df.reset_index().convert_dtypes()
