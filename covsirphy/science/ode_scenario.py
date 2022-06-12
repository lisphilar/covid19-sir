#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import pandas as pd
from covsirphy.util.error import ScenarioNotFoundError, SubsetNotFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.alias import Alias
from covsirphy.util.term import Term
from covsirphy.gis.gis import GIS
from covsirphy.engineering.engineer import DataEngineer
from covsirphy.dynamics.ode import ODEModel
from covsirphy.dynamics.dynamics import Dynamics


class ODEScenario(Term):
    """Perform scenario analysis, changing ODE parameters.

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
    _PARAM = "param"

    def __init__(self, data, location_name, complement=True):
        self._location_name = str(location_name)
        # Actual records: Date index, S/I/F/R
        df = Validator(data, "data", accept_none=False).dataframe(
            time_index=True, columns=[self.N, self.C, self.F, self.R])
        df.index.name = self.DATE
        df["location"] = self._location_name
        engineer = DataEngineer(layers=["location"])
        engineer.register(data=df.reset_index())
        engineer.clean()
        engineer.transform()
        self._actual_df, *_ = engineer.subset(
            geo=self._location_name, variables=[self.S, self.CI, self.F, self.R], complement=complement)
        self._first, self._last = self._actual_df.index.min(), self._actual_df.index.max()
        # {scenario_name: {"ODE": ODEModel, "tau": int, "param": pd.DataFrame(index: Date, columns: ODE parameters)}}
        self._snl_alias = Alias(target_class=dict)

    def build_with_dynamics(self, name, dynamics):
        """Build a scenario with covsirphy.Dynamics() instance.

        Args:
            name (str): scenario name
            dynamics (covsirphy.Dynamics): covsirphy.Dynamics() instance which has ODE model, tau value and ODE parameter values

        Return:
            covsirphy.ODEScenario: self
        """
        dyn = Validator(dynamics, "dynamics").instance(Dynamics)
        snl_dict = {self.ODE: dyn.model, self.TAU: dyn.tau, self._PARAM: dyn.track().loc[:, dyn.model._PARAMETERS]}
        self._snl_alias.update(name=name, target=snl_dict)
        return self

    def build_with_model(self, name, model, date_range=None, tau=None):
        """Build a scenario with covsirphy.Dynamics() instance created with the actual data automatically.

        Args:
            name (str): scenario name
            model (covsirphy.ODEModel): definition of ODE model
            date_range (tuple of (str, str)): start date and end date of dynamics to analyze
            tau (int or None): tau value [min] or None (set later with data)

        Return:
            covsirphy.ODEScenario: self
        """
        Validator(model, "model").subclass(ODEModel)
        start_date, end_date = Validator(date_range, "date_range").sequence(default=(None, None), length=2)
        start = Validator(start_date, name="the first value of @date_range").date(default=self._first)
        end = Validator(
            end_date, name="the second date of @date_range").date(value_range=(start, None), default=self._last)
        dyn = Dynamics(model=model, date_range=(start, end), tau=tau)
        dyn.register(data=self._actual_df.loc[:, self._SIFR])
        dyn.segment(points=None, overwrite=True, display=False)
        dyn.estimate()
        return self.build_with_dynamics(name=name, dynamics=dyn)

    def build_with_template(self, name, template):
        """Build a scenario with a template scenario.

        Args:
            name (str): new scenario name
            template (str): template name

        Raises:
            SubsetNotFoundError: scenario with the name is not registered

        Return:
            covsirphy.ODEScenario: self
        """
        temp_snl_dict = self._snl_alias.find(name=template)
        if temp_snl_dict is None:
            raise ScenarioNotFoundError(name=template)
        self._snl_alias.update(name=name, target=deepcopy(temp_snl_dict))
        return self

    @classmethod
    def auto_build(cls, geo, model):
        """Prepare cleaned and subset data from recommended dataset, create instance, build baseline scenario.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): country, province, city
            model (covsirphy.ODEModel): definition of ODE model

        Raises:
            SubsetNotFoundError: actual data of the location was not included in the recommended dataset

        Return:
            covsirphy.ODEScenario: created instance

        Note:
            `geo=None` means total values of all countries.

        Note:
            `geo="Japan"` and `geo=("Japan",)` means country level data of Japan, as an example.

        Note:
            `geo=("Japan", "Tokyo")` means prefecture (province) level data of Tokyo/Japan, as an example.

        Note:
            `geo=("USA", "Alabama", "Baldwin")` means country level data of Baldwin/Alabama/USA, as an example.

        Note:
            Complemented data with Recovered > 0 will be analyzed.
        """
        Validator(geo, "geo", accept_none=True).instance(expected=(str, tuple, list))
        Validator(model, "model", accept_none=False).subclass(ODEModel)
        # Prepare data
        engineer = DataEngineer()
        engineer.download(
            country=geo[0] if isinstance(geo, (tuple, list)) and len(geo) > 1 else None,
            province=geo[1] if isinstance(geo, (tuple, list)) and len(geo) > 2 else None)
        try:
            subset_df, *_ = engineer.subset(geo=geo, complement=True)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                geo=geo, details="Please create covsirphy.DataEngineer() instance to prepare data") from None
        # Create instance
        snl = cls(data=subset_df.loc[subset_df[cls.R] > 0], location_name=GIS.area_name(geo=geo), complement=False)
        # Build baseline scenario
        snl.build_with_model(name="Baseline", model=model)
        return snl

    def to_dynamics(self, name):
        """Create covsirphy.Dynamics instance of the scenario.

        Args:
            name (str): scenario name registered

        Raises:
            SubsetNotFoundError: scenario with the name is not registered

        Returns:
            covsirphy.Dynamics: instance which has ODE model, tau value and ODE parameter values
        """
        temp_snl_dict = self._snl_alias.find(name=name)
        if temp_snl_dict is None:
            raise ScenarioNotFoundError(name=name)
        model, tau, param_df = [temp_snl_dict[k] for k in [self.ODE, self.TAU, self._PARAM]]
        df = self._actual_df.join(param_df, how="right")
        return Dynamics.from_data(model=model, data=df, tau=tau, name=name)

    def summary(self):
        """Summarize phase information of all scenarios.

        Returns:
            pandas.DataFrame:
                Index
                    Scenario (str): scenario names
                    Phase (str): phase names, 0th, 1st,...
                Columns
                    Start (pandas.Timestamp): start date of the phase
                    End (pandas.Timestamp): end date of the phase
                    Rt (float): phase-dependent reproduction number (if parameters are available)
                    (float): parameter values, including rho (if available)
                    (int or float): dimensional parameters, including 1/beta [days] (if tau and parameters are available)
        """
        dataframes = []
        for name in self._snl_alias.all().keys():
            dyn = self.to_dynamics(name=name)
            df = dyn.summary().reset_index()
            df.insert(0, self.SERIES, name)
            print(df)
            dataframes.append(df)
        return pd.concat(dataframes, axis=0).set_index([self.SERIES, self.PHASE]).convert_dtypes()
