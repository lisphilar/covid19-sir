#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import re
import pandas as pd
from covsirphy.util.error import ScenarioNotFoundError, SubsetNotFoundError, UnExpectedTypeError, UnExpectedValueRangeError
from covsirphy.util.error import UnExpectedValueError
from covsirphy.util.validator import Validator
from covsirphy.util.alias import Alias
from covsirphy.util.term import Term
from covsirphy.gis.gis import GIS
from covsirphy.visualization.line_plot import line_plot
from covsirphy.engineering.engineer import DataEngineer
from covsirphy.dynamics.ode import ODEModel
from covsirphy.dynamics.sir import SIRModel
from covsirphy.dynamics.sird import SIRDModel
from covsirphy.dynamics.sirf import SIRFModel
from covsirphy.dynamics.sewirf import SEWIRFModel
from covsirphy.dynamics.dynamics import Dynamics
from covsirphy.science.ml import MLEngineer


class ODEScenario(Term):
    """Perform scenario analysis, changing ODE parameters.

    Args:
        data (pandas.DataFrame): actual data of the number of cases
            Index
                Date (pandas.Timestamp): observation dates
            Columns
                Population (int): total population
                Confirmed (int): the number of confirmed cases
                Recovered (int): the number of recovered cases, must be over 0
                Fatal (int): the number of fatal cases
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
        self._complement = complement
        # Actual records: Date index, S/I/F/R
        df = Validator(data, "data", accept_none=False).dataframe(
            time_index=True, columns=[self.N, self.C, self.F, self.R])
        self._data = df.copy()
        df.index.name = self.DATE
        df["location"] = self._location_name
        engineer = DataEngineer(layers=["location"])
        engineer.register(data=df.reset_index())
        engineer.clean()
        engineer.transform()
        self._actual_df, *_ = engineer.subset(
            geo=self._location_name, variables=[self.S, self.CI, self.R, self.F], complement=complement)
        self._first, self._last = self._actual_df.index.min(), self._actual_df.index.max()
        # {scenario_name: {"ODE": ODEModel, "tau": int, "param": pd.DataFrame(index: Date, columns: ODE parameters)}}
        self._snr_alias = Alias(target_class=dict)
        # Aliases of variable names
        self._variable_alias = Alias.for_variables()

    def to_json(self, filename):
        """Write a JSON file which can usable for recreating ODEScenario instance with .from_json()

        Args:
            filename (str or Path): JSON filename

        Return:
            str: filename
        """
        info_dict = {
            "location_name": self._location_name,
            "complement": self._complement,
            "data": self._data.to_json(date_format="iso", force_ascii=False),
            "scenarios": {
                name: {
                    self.ODE: detail_dict[self.ODE].name(),
                    self.TAU: detail_dict[self.TAU],
                    self._PARAM: detail_dict[self._PARAM].to_json(date_format="iso", force_ascii=False),
                }
                for (name, detail_dict) in self._snr_alias.all().items()
            },
        }
        with Path(filename).open("w") as fh:
            json.dump(info_dict, fh, indent=4)
        return str(filename)

    @classmethod
    def from_json(cls, filename):
        """Create ODEScenario instance with a JSON file.

        Args:
            filename (str or Path): JSON filename

        Returns:
            covsirphy.ODEScenario: self
        """
        with Path(filename).open("r") as fh:
            info_dict = json.load(fh)
        # Validation
        Validator(info_dict, name="info_dict", accept_none=False).dict(
            required_keys=["location_name", "complement", "data", "scenarios"], errors="raise")
        Validator(info_dict["scenarios"], "info_dict['scenarios']", accept_none=False).dict()
        # Create instance
        instance = cls(
            data=pd.read_json(info_dict["data"]),
            location_name=info_dict["location_name"],
            complement=info_dict["complement"],
        )
        required_keys = [cls.ODE, cls.TAU, cls._PARAM]
        model_dict = {model.name(): model for model in [SIRModel, SIRDModel, SIRFModel, SEWIRFModel]}
        for (name, detail_dict) in info_dict["scenarios"].items():
            Validator(detail_dict, f"info_dict['scenarios']['{name}']").dict(
                required_keys=required_keys, errors="raise")
            model_name = detail_dict[cls.ODE]
            Validator([model_name], name=f"model name of {name} scenario", accept_none=False).sequence(
                candidates=model_dict.keys(), length=1)
            instance._snr_alias.update(
                name=name,
                target={
                    cls.ODE: model_dict[model_name],
                    cls.TAU: Validator(detail_dict[cls.TAU], name="tau", accept_none=False).tau(),
                    cls._PARAM: pd.read_json(detail_dict[cls._PARAM]),
                }
            )
        return instance

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
        self._snr_alias.update(name=name, target=snl_dict)
        return self

    def build_with_model(self, name, model, date_range=None, tau=None):
        """Build a scenario with covsirphy.Dynamics() instance created with the actual data automatically.

        Args:
            name (str): scenario name
            model (covsirphy.ODEModel): definition of ODE model
            date_range (tuple of (str, str) or None): start date and end date of dynamics to analyze
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
        dyn.register(data=self._actual_df.loc[:, self._SIRF])
        dyn.segment(points=None, overwrite=True, display=False)
        dyn.estimate()
        return self.build_with_dynamics(name=name, dynamics=dyn)

    def build_with_template(self, name, template):
        """Build a scenario with a template scenario.

        Args:
            name (str): new scenario name
            template (str): template name

        Raises:
            SubsetNotFoundError: scenario with the name is un-registered

        Return:
            covsirphy.ODEScenario: self
        """
        temp_snl_dict = self._snr_alias.find(name=template)
        if temp_snl_dict is None:
            raise ScenarioNotFoundError(name=template)
        self._snr_alias.update(name=name, target=deepcopy(temp_snl_dict))
        return self

    @classmethod
    def auto_build(cls, geo, model, complement=True):
        """Prepare cleaned and subset data from recommended dataset, create instance, build baseline scenario.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): country, province, city
            model (covsirphy.ODEModel): definition of ODE model
            complement (bool): whether perform complement or not

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
            Complemented (if @complement is True) data with Recovered > 0 will be analyzed.
        """
        Validator(geo, "geo", accept_none=True).instance(expected=(str, tuple, list))
        Validator(model, "model", accept_none=False).subclass(ODEModel)
        # Prepare data
        engineer = DataEngineer()
        engineer.download(
            country=geo[0] if isinstance(geo, (tuple, list)) and len(geo) > 1 else None,
            province=geo[1] if isinstance(geo, (tuple, list)) and len(geo) > 2 else None,
            databases=["japan", "covid19dh"])
        engineer.clean()
        try:
            subset_df, *_ = engineer.subset(geo=geo, complement=complement)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                geo=geo, details="Please create covsirphy.DataEngineer() instance to prepare data") from None
        # Create instance
        snl = cls(data=subset_df.loc[subset_df[cls.R] > 0], location_name=GIS.area_name(geo=geo), complement=False)
        # Build baseline scenario
        snl.build_with_model(name="Baseline", model=model)
        return snl

    def delete(self, pattern, exact=False):
        """Delete scenario(s).

        Args:
            pattern (str): scenario name or pattern to search
            exact (bool): if False, use regular expressions

        Return:
            covsirphy.ODEScenario: self
        """
        if exact:
            self._snr_alias.delete(name=pattern)
            return self
        p = re.compile(pattern)
        names = [name for name in self._snr_alias.all().keys() if p.search(name)]
        for name in names:
            self._snr_alias.delete(name=name)
        return self

    def rename(self, old, new):
        """Rename the given scenario names with a new one.

        Args:
            old (str): old name
            new (str): new name

        Returns:
            covsirphy.Scenario: self
        """
        self.build_with_template(name=new, template=old)
        self.delete(pattern=old, exact=True)
        return self

    def to_dynamics(self, name):
        """Create covsirphy.Dynamics instance of the scenario.

        Args:
            name (str): scenario name

        Raises:
            SubsetNotFoundError: scenario with the name is un-registered

        Returns:
            covsirphy.Dynamics: instance which has ODE model, tau value and ODE parameter values
        """
        temp_snl_dict = self._snr_alias.find(name=name)
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
                    ODE (str): ODE model name
                    Rt (float): phase-dependent reproduction number (if parameters are available)
                    (float): parameter values, including rho (if available)
                    tau (int): tau value [min]
                    (int or float): dimensional parameters, including 1/beta [days] (if tau and parameters are available)
        """
        dataframes = []
        for name, snl_dict in self._snr_alias.all().items():
            dyn = self.to_dynamics(name=name)
            df = dyn.summary().reset_index()
            df[self.SERIES] = name
            df[self.ODE] = snl_dict[self.ODE].name()
            df[self.TAU] = snl_dict[self.TAU]
            dataframes.append(df)
        return pd.concat(dataframes, axis=0).set_index([self.SERIES, self.PHASE]).convert_dtypes()

    def track(self):
        """Track reproduction number, parameter value and dimensional parameter values.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    Scenario (str): scenario names
                    Phase (str): phase names
                    Date (pandas.Timestamp): dates
                    Rt (float): phase-dependent reproduction number (if parameters are available)
                    (float): parameter values, including rho (if available)
                    (int or float): dimensional parameters, including 1/beta [days] (if tau and parameters are available)
        """
        df = self.summary().reset_index(drop=False)
        df[self.DATE] = df[[self.START, self.END]].apply(
            lambda x: pd.date_range(start=x[0], end=x[1], freq="D"), axis=1)
        return df.explode(self.DATE).set_index(self.DATE).drop([self.START, self.END], axis=1)

    def describe(self):
        """Describe representative values.

        Returns:
            pandas.DataFrame:
                Index
                    str: scenario name
                Columns
                    - max(Infected) (numpy.int64): max value of Infected
                    - argmax(Infected) (pandas.Timestamp): the date when Infected shows max value
                    - Confirmed({date}) (numpy.int64): Confirmed on the last date
                    - Infected({date} (numpy.int64)): Infected on the last date
                    - Fatal({date}) (numpy.int64): Fatal on the last date
        """
        _dict = {}
        for name in self._snr_alias.all().keys():
            dyn = self.to_dynamics(name=name)
            sim_df = dyn.simulate(model_specific=False)
            sim_df[self.SERIES] = name
            engineer = DataEngineer(layers=[self.SERIES])
            engineer.register(data=sim_df.reset_index())
            engineer.inverse_transform()
            sim_df = engineer.all().set_index(self.DATE)
            last_date = sim_df.index[-1]
            last_date_str = last_date.strftime(self.DATE_FORMAT)
            _dict[name] = {
                f"max({self.CI})": sim_df[self.CI].max(),
                f"argmax({self.CI})": sim_df[self.CI].idxmax(),
                f"{self.C} on {last_date_str}": sim_df.loc[last_date, self.C],
                f"{self.CI} on {last_date_str}": sim_df.loc[last_date, self.CI],
                f"{self.F} on {last_date_str}": sim_df.loc[last_date, self.F],
            }
        return pd.DataFrame.from_dict(_dict, orient="index")

    def simulate(self, name=None, variables=None, display=True, **kwargs):
        """Perform simulation with phase-dependent ODE model.

        Args:
            name (str or None): scenario name registered or None (actual data)
            variables (list of [str] or None): variables/alias to return or None (["Confirmed", "Fatal", "Recovered"])
            display (bool): whether display figure of the result or not
            **kwargs: keyword arguments of covsirphy.line_plot() except for @df

        Returns:
            pandas.DataFrame or pandas.Series:
                Index
                    Date (pd.Timestamp): dates
                Columns
                    Population (int): total population (if selected with @variables)
                    Confirmed (int): the number of confirmed cases (if selected with @variables)
                    Recovered (int): the number of recovered cases (if selected with @variables)
                    Fatal (int): the number of fatal cases (if selected with @variables)
                    Susceptible (int): the number of susceptible cases (if selected with @variables)
                    Infected (int): the number of currently infected cases (if selected with @variables)
        """
        if name is None:
            sirf_df = self._actual_df.copy()
            title = f"{self._location_name}: actual number of cases over time"
            v = None
        else:
            dyn = self.to_dynamics(name=name)
            sirf_df = dyn.simulate(model_specific=False)
            title = f"{self._location_name} ({name} scenario): simulated number of cases over time"
            v = dyn.start_dates()[1:]
        sirf_df[self.SERIES] = name or self.ACTUAL
        engineer = DataEngineer(layers=[self.SERIES])
        engineer.register(data=sirf_df.reset_index())
        engineer.inverse_transform()
        v_converted = self._variable_alias.find(name=variables, default=[self.C, self.F, self.R])
        df = engineer.all().set_index(self.DATE).loc[self._first:self._last, v_converted]
        # Show figure
        if display:
            plot_kwargs = {"title": title, "y_integer": True, "v": v, "ylabel": "the number of cases", **kwargs}
            line_plot(df=df, **plot_kwargs)
        return df

    def compare_cases(self, variable, date_range=None, ref=None, display=True, **kwargs):
        """Compare the number of cases of scenarios.

        Args:
            variable (str): variable name or alias
            date_range (tuple of (str, str)): start date and end date to analyze
            ref (str or None): name of reference scenario to specify phases and dates or None (the first scenario)
            display (bool): whether display figure of the result or not
            **kwargs: keyword arguments of covsirphy.line_plot() except for @df

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): dates
                Columns
                    Actual (numpy.int64): actual records
                    {scenario name} (numpy.int64): values of the scenario
        """
        v_converted = Validator(
            self._variable_alias.find(name=variable, default=[variable]), "variable", accept_none=False).sequence(
            length=1, candidates=[self.C, self.N, *self._actual_df.columns.tolist()])[0]
        df = pd.DataFrame(index=self._actual_df.index.tolist())
        if v_converted == self.C:
            df[self.ACTUAL] = self._actual_df[[self.CI, self.F, self.R]].sum(axis=1)
        elif v_converted == self.N:
            df[self.ACTUAL] = self._actual_df[[self.CI, self.F, self.R, self.S]].sum(axis=1)
        else:
            df[self.ACTUAL] = self._actual_df.loc[:, v_converted]
        dataframes = [
            self.simulate(name=name, display=False)[v_converted].rename(name) for name in self._snr_alias.all().keys()]
        df = pd.concat([df, *dataframes], axis=1)
        start_date, end_date = Validator(date_range, "date_range").sequence(default=(None, None), length=2)
        start = Validator(start_date, name="the first value of @date_range").date(default=df.index.min())
        end = Validator(
            end_date, name="the second date of @date_range").date(value_range=(start, None), default=df.index.max())
        df = df.loc[start: end]
        if display:
            ylabel = f"the number of {v_converted.lower()} cases"
            title = f"{self._location_name}: {ylabel} overt time"
            v = self.to_dynamics(name=ref or list(self._snr_alias.all().keys())[0]).start_dates()[1:]
            plot_kwargs = {"title": title, "y_integer": True, "v": v, "ylabel": ylabel, **kwargs}
            line_plot(df=df, **plot_kwargs)
        return df.convert_dtypes()

    def compare_param(self, param, date_range=None, ref=None, display=True, **kwargs):
        """Compare the number of cases of scenarios.

        Args:
            param (str): one of ODE parameters, "Rt", dimensional parameters
            date_range (tuple of (str, str)): start date and end date to analyze
            ref (str or None): name of reference scenario to specify phases and dates or None (the first scenario)
            display (bool): whether display figure of the result or not
            **kwargs: keyword arguments of covsirphy.line_plot() except for @df

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp)
                Columns
                    {scenario name} (str): values of the scenario
        """
        df = self.track()
        Validator([param], "param", accept_none=False).sequence(candidates=df.columns)
        df[param] = df[param].astype("float64")
        df = df.pivot_table(values=param, index=self.DATE, columns=self.SERIES)
        start_date, end_date = Validator(date_range, "date_range").sequence(default=(None, None), length=2)
        start = Validator(start_date, name="the first value of @date_range").date(default=df.index.min())
        end = Validator(
            end_date, name="the second date of @date_range").date(value_range=(start, None), default=df.index.max())
        df = df.loc[start: end]
        if display:
            ylabel, h = (self.RT_FULL, 1.0) if param == self.RT else (param, None)
            title = f"{self._location_name}: {ylabel} overt time"
            v = self.to_dynamics(name=ref or list(self._snr_alias.all().keys())[0]).start_dates()[1:]
            plot_kwargs = {"title": title, "math_scale": False, "v": v, "ylabel": ylabel, "h": h, **kwargs}
            line_plot(df=df, **plot_kwargs)
        return df.convert_dtypes()

    def _append(self, name, end, **kwargs):
        """Append a new phase, specifying ODE parameter values.

        Args:
            name (str): scenario name
            end (pandas.Timestamp or int): end date or the number days of new phase
            **kwargs: keyword arguments of ODE parameter values (default: values of the last phase)

        Raises:
            SubsetNotFoundError: scenario with the name is un-registered
            UnExpectedValueRangeError: end_date - (the last date of the registered phases) < 3 and parameters were changed
        """
        try:
            snr_dict = self._snr_alias.find(name=name).copy()
        except AttributeError:
            raise ScenarioNotFoundError(name=name) from None
        param_df = snr_dict[self._PARAM].copy()
        last_param_dict = param_df.iloc[-1].to_dict()
        start_date = param_df.index[-1] + timedelta(days=1)
        try:
            delta = timedelta(days=Validator(end, "end", accept_none=False).int(value_range=(0, None)))
            end_date = param_df.index[-1] + delta
        except UnExpectedTypeError:
            end_date = Validator(end, "end", accept_none=False).date(value_range=(param_df.index[-1], None))
        new_param_dict = Validator(kwargs, "keyword arguments").dict(
            default=last_param_dict, required_keys=list(last_param_dict.keys()))
        new_df = pd.DataFrame(new_param_dict, index=pd.date_range(start=start_date, end=end_date, freq="D"))
        snr_dict[self._PARAM] = pd.concat([param_df, new_df], axis=0)
        try:
            Dynamics.from_data(
                model=snr_dict[self.ODE],
                data=self._actual_df.join(snr_dict[self._PARAM], how="right"),
                tau=snr_dict[self.TAU])
        except UnExpectedValueError:
            if isinstance(end, int):
                target, _min = end, 3
            else:
                target = end.strftime(self.DATE_FORMAT)
                _min = (param_df.index[-1] + timedelta(days=3)).strftime(self.DATE_FORMAT)
            raise UnExpectedValueRangeError(
                name="end", target=target, value_range=(_min, None)) from None
        self._snr_alias.update(name=name, target=snr_dict.copy())
        self._last = max((self._last, end_date))

    def append(self, name=None, end=None, **kwargs):
        """Append a new phase, specifying ODE parameter values.

        Args:
            name (str or list[str] None): scenario name(s) or None (all scenarios)
            end (pandas.Timestamp or int or None): end date or the number days of new phase or None (the max date of all scenarios and actual data)
            **kwargs: keyword arguments of ODE parameter values (default: values of the last phase)

        Raises:
            SubsetNotFoundError: scenario with the name is un-registered
            UnExpectedValueRangeError: end_date - (the last date of the registered phases) < 3 and parameters were changed

        Return:
            covsirphy.ODEScenario: self
        """
        if end is None:
            last_end = self._last
            for snl_dict in self._snr_alias.all().values():
                end_date = snl_dict[self._PARAM].index[-1]
                last_end = max(last_end, end_date)
        names = [name] if isinstance(name, str) else Validator(name, "name").sequence(
            default=list(self._snr_alias.all().keys()))
        for _name in names:
            try:
                self._append(name=_name, end=end or last_end, **kwargs)
            except UnExpectedValueRangeError as e:
                raise e from None
        return self

    def predict(self, days, name, seed=0, verbose=1, X=None, **kwargs):
        """Create scenarios and append a phase, performing prediction ODE parameter prediction for given days.

        Args:
            days (int): days to predict
            name (str): scenario name
            X (pandas.DataFrame or None): information for regression or None (no information)
                Index
                    pandas.Timestamp: Observation date
                Columns
                    observed and the target variables (int or float)
            seed (int or None): random seed
            verbose (int): verbosity
            **kwargs: keyword arguments of autots.AutoTS() except for verbose, forecast_length (always the same as @days)

        Return:
            covsirphy.ODEScenario: self

        Note:
            AutoTS package is developed at https://github.com/winedarksea/AutoTS

        Note:
            Phases are determined with rounded reproduction number (one decimal place).
        """
        model = self._snr_alias.find(name=name)[self.ODE]
        Y = self.to_dynamics(name=name).track().loc[:, model._PARAMETERS]
        # Parameter prediction
        eng = MLEngineer(seed=seed, verbose=verbose)
        param_df = eng.forecast(Y=Y, days=days, X=X, **kwargs).reset_index()
        # Create phases with Rt values
        param_df[self.RT] = param_df[model._PARAMETERS].apply(
            lambda x: model.from_data(data=self._actual_df.reset_index(), param_dict=x.to_dict(), tau=1440).r0(), axis=1)
        param_df["Phase"] = (param_df[self.RT] != param_df[self.RT].shift()).cumsum()
        phase_df = param_df.groupby("Phase").last().drop(self.RT, axis=1).rename(columns={self.DATE: "end"})
        phase_df = phase_df.loc[
            (phase_df.index[-1]) | (phase_df["end"] < phase_df["end"].shift(periods=-1) - timedelta(days=3))]
        for phase_dict in phase_df.to_dict(orient="records"):
            with contextlib.suppress(UnExpectedValueRangeError):
                # Skip when the first new phase is smaller than 3 days
                self._append(name=name, **phase_dict)
        return self

    def represent(self, q, variable, date=None, included=None, excluded=None):
        """
        Return the names of representative scenarios using quantiles of the variable on on the date.

        Args:
            q (list[float] or float): quantiles
            variable (str): reference variable, Confirmed, Infected, Fatal or Recovered
            date (str or None): reference date or None (the last end date in the all scenarios)
            included (list[str] or None): included scenarios or None (all included)
            excluded (list[str] or None): excluded scenarios or None (no scenarios not excluded)

        Raises:
            ValueError: the end dates of the last phase is not aligned

        Returns:
            list[float] or float: the nearest scenarios which has the values at the given quantiles

        Note:
            Dimension of returned object corresponds to the type of @q.
        """
        quantiles = [q] if isinstance(q, float) else Validator(q, "q").sequence()
        v_converted = self._variable_alias.find(name=variable, default=[variable])
        Validator(v_converted, "variable", accept_none=False).sequence(length=1)
        # Target scenario to included
        all_set = set(self._snr_alias.all().keys())
        in_set = all_set if included is None else set(Validator(included, "included").sequence())
        ex_set = set() if excluded is None else set(Validator(excluded, "excluded").sequence())
        scenarios = list(all_set & (in_set) - ex_set)
        # Get simulation data of the variable of the target scenarios
        sim_dict = {name: self.simulate(name=name, display=False)[variable] for name in scenarios}
        sim_df = pd.DataFrame(sim_dict)
        if sim_df.isna().to_numpy().sum():
            raise ValueError(
                "The end dates of the last phases must be aligned. Scenario.adjust_end() method may fix this issue.")
        # Find representative scenario
        date_obj = Validator(date, "date").date(default=sim_df.index[-1])
        series = sim_df.loc[date_obj].squeeze()
        values = series.quantile(q=quantiles, interpolation="nearest")
        return [series[series == v].index.tolist()[0] for v in values]
