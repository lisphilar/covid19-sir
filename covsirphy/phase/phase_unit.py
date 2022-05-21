#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.error import UnExecutedError, deprecate
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.estimator import Estimator
from covsirphy.simulation.simulator import ODESimulator


class PhaseUnit(Term):
    """
    Save information of  a phase.

    Args:
        start_date (str): start date of the phase
        end_date (str): end date of the phase
        population (int): population value

    Examples:
        >>> unit1 = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        >>> unit2 = PhaseUnit("02Feb2020", "01Mar2020", 1000)
        >>> unit3 = PhaseUnit("02Mar2020", "01Apr2020", 1000)
        >>> unit4 = PhaseUnit("02Mar2020", "01Apr2020", 1000)
        >>> unit5 = PhaseUnit("01Jan2020", "01Apr2020", 1000)
        >>> str(unit1)
        'Phase (01Jan2020 - 01Feb2020)'
        >>> unit4 == unit4
        True
        >>> unit1 != unit2
        True
        >>> unit1 < unit2
        True
        >>> unit3 > unit1
        True
        >>> unit3 < unit4
        False
        >>> unit3 <= unit4
        True
        >>> unit1 < "02Feb2020"
        True
        >>> unit1 <= "01Feb2020"
        True
        >>> unit1 > "31Dec2019"
        True
        >>> unit1 >= "01Jan2020"
        True
        >>> sorted([unit3, unit1, unit2]) == [unit1, unit2, unit3]
        True
        >>> str(unit1 + unit2)
        'Phase (01Jan2020 - 01Mar2020)'
        >>> str(unit5 - unit1)
        'Phase (02Feb2020 - 01Apr2020)'
        >>> str(unit5 - unit4)
        'Phase (01Jan2020 - 01Mar2020)'
        >>> set([unit1, unit3, unit4]) == set([unit1, unit3])
        True
    """

    @deprecate("PhaseUnit", new="ODEHandler", version="2.19.1-zeta-fu1")
    def __init__(self, start_date, end_date, population):
        self._ensure_date_order(start_date, end_date, name="end_date")
        self._start_date = start_date
        self._end_date = end_date
        population = Validator(population, "population").int(value_range=(1, None))
        # Summary of information
        self.info_dict = {
            self.START: start_date,
            self.END: end_date,
            self.N: population,
            self.ODE: None,
            self.RT: None
        }
        self._ode_dict = {self.TAU: None}
        self.day_param_dict = {}
        self.est_dict = {
            **{metric: None for metric in Evaluator.metrics()},
            self.TRIALS: None,
            self.RUNTIME: None
        }
        # Init
        self._id_dict = None
        self._enabled = True
        self._model = None
        self._record_df = pd.DataFrame()
        self.y0_dict = {}
        self._estimator = None

    @classmethod
    def _ensure_date_order(cls, previous_date, following_date, name="following_date"):
        """
        Ensure that the order of dates.

        Args:
            previous_date (str or pandas.Timestamp): previous date
            following_date (str or pandas.Timestamp): following date
            name (str): name of @following_date

        Raises:
            ValueError: @previous_date > @following_date
        """
        previous_date = cls._ensure_date(previous_date)
        following_date = cls._ensure_date(following_date)
        p_str = previous_date.strftime(cls.DATE_FORMAT)
        f_str = following_date.strftime(cls.DATE_FORMAT)
        if previous_date <= following_date:
            return None
        raise ValueError(f"@{name} must be the same as/over {p_str}, but {f_str} was applied.")

    def __str__(self):
        if self._id_dict is None:
            header = "Phase"
        else:
            id_str = ', '.join(list(self._id_dict.values()))
            header = f"{id_str:>4} phase"
        return f"{header} ({self._start_date} - {self._end_date})"

    def __hash__(self):
        return hash((self._start_date, self._end_date))

    def __eq__(self, other):
        if not isinstance(other, PhaseUnit):
            raise NotImplementedError
        return self._start_date == other.start_date and self._end_date == other.end_date

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        # self < other
        end = self._ensure_date(self._end_date)
        if isinstance(other, str):
            sta_other = self._ensure_date(other)
        elif isinstance(other, PhaseUnit):
            sta_other = self._ensure_date(other.start_date)
        else:
            raise NotImplementedError
        return end < sta_other

    def __le__(self, other):
        # self <= other
        end = self._ensure_date(self._end_date)
        if isinstance(other, str):
            sta_other = self._ensure_date(other)
        elif isinstance(other, PhaseUnit):
            if self.__eq__(other):
                return True
            sta_other = self._ensure_date(other.start_date)
        else:
            raise NotImplementedError
        return end <= sta_other

    def __gt__(self, other):
        # self > other
        if isinstance(other, PhaseUnit) and self.__eq__(other):
            return False
        return not self.__le__(other)

    def __ge__(self, other):
        # self >= other
        return not self.__lt__(other)

    def __add__(self, other):
        if self < other:
            return PhaseUnit(self._start_date, other.end_date, self._population)
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        sta = self._ensure_date(self._start_date)
        end = self._ensure_date(self._end_date)
        sta_other = self._ensure_date(other.start_date)
        end_other = self._ensure_date(other.end_date)
        if sta < sta_other and end == end_other:
            end_date = self.yesterday(other.start_date)
            return PhaseUnit(self._start_date, end_date, self._population)
        if sta == sta_other and end > end_other:
            start_date = self.tomorrow(other.end_date)
            return PhaseUnit(start_date, self._end_date, self._population)

    def __isub__(self, other):
        return self.__sub__(other)

    def __contains__(self, date):
        sta = self._ensure_date(self._start_date)
        end = self._ensure_date(self._end_date)
        date = self._ensure_date(date)
        return sta <= date <= end

    @classmethod
    def _ensure_date(cls, target, name="date", default=None):
        """
        Ensure the format of the string.

        Args:
            target (str or pandas.Timestamp): string to ensure
            name (str): argument name of the string
            default (pandas.Timestamp or None): default value to return

        Returns:
            pandas.Timestamp or None: as-is the target or default value
        """
        if target is None:
            return default
        if isinstance(target, pd.Timestamp):
            return target.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            return pd.to_datetime(target).replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError as e:
            raise ValueError(f"{name} was not recognized as a date, {target} was applied.") from e

    @classmethod
    def tomorrow(cls, date_str):
        """
        Tomorrow of the date.

        Args:
            date_str (str): today

        Returns:
            str: tomorrow
        """
        return cls.date_change(date_str, days=1)

    @classmethod
    def yesterday(cls, date_str):
        """
        Yesterday of the date.

        Args:
            date_str (str): today

        Returns:
            str: yesterday
        """
        return cls.date_change(date_str, days=-1)

    @property
    def id_dict(self):
        """
        tuple(str): id_dict of the phase
        """
        return self._id_dict

    @ id_dict.setter
    def id_dict(self, value):
        self.set_id(value)

    def set_id(self, **kwargs):
        """
        Set identifiers.

        Args:
            id_dict (dict[str, str]): dictionary of identifiers

        Returns:
            covsirphy.PhaseUnit: self
        """
        if self._id_dict is not None:
            raise AttributeError("@id_dict cannot be overwritten.")
        self._id_dict = kwargs
        return self

    def del_id(self):
        """
        Delete identifiers.

        Returns:
            covsirphy.PhaseUnit: self
        """
        self._id_dict = None
        return self

    def enable(self):
        """
        Enable the phase.

        Examples:
            >>> unit.enable
            >>> bool(unit)
            True
        """
        self._enabled = True

    def disable(self):
        """
        Disable the phase.

        Examples:
            >>> unit.disable
            >>> bool(unit)
            False
        """
        self._enabled = False

    def __bool__(self):
        return self._enabled

    @property
    def start_date(self):
        """
        str: start date
        """
        return self._start_date

    @property
    def end_date(self):
        """
        str: end date
        """
        return self._end_date

    @property
    def population(self):
        """
        str: population value
        """
        return self._population

    @property
    def tau(self):
        """
        int or None: tau value [min]
        """
        return self._ode_dict[self.TAU]

    @ tau.setter
    def tau(self, value):
        if self._ode_dict[self.TAU] is None:
            self._ode_dict[self.TAU] = Validator(value, "tau").tau(default=None)
            return
        raise AttributeError(
            f"PhaseUnit.tau is not None ({self._ode_dict[self.TAU]}) and cannot be changed.")

    @property
    def model(self):
        """
        covsirphy.ModelBase or None: model description
        """
        return self._model

    @property
    def estimator(self):
        """
        covsirphy.Estimator or None: estimator object
        """
        return self._estimator

    def to_dict(self):
        """
        Summarize phase information and return as a dictionary.

        Returns:
            dict:
                - Start: start date of the phase
                - End: end date of the phase
                - Population: population value of the start date
                - if available:
                    - ODE: model name
                    - Rt: (basic) reproduction number
                    - parameter values if available
                    - day parameter values if available
                    - tau: tau value [min]
                    - {metric name}: score of parameter estimation
                    - Trials: the number of trials in estimation
                    - Runtime: runtime of estimation
        """
        return {
            **self.info_dict,
            **self._ode_dict,
            **self.day_param_dict,
            **self.est_dict
        }

    def summary(self):
        """
        Summarize information.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Start: start date of the phase
                    - End: end date of the phase
                    - Population: population value of the start date
                    - if available:
                        - ODE (str): model name
                        - Rt (float): (basic) reproduction number
                        - rho etc. (float): parameter values if available
                        - tau (int): tau value [min]
                        - (int): day parameter values if available
                        - {metric name} (float): score of parameter estimation
                        - Trials (int): the number of trials in parameter estimation
                        - Runtime (str): runtime of parameter estimation
        """
        summary_dict = self.to_dict()
        df = pd.DataFrame.from_dict(summary_dict, orient="index").T
        return df.dropna(how="all", axis=1)

    def set_ode(self, model=None, tau=None, **kwargs):
        """
        Set ODE model, tau value and parameter values, if necessary.

        Args:
            model (covsirphy.ModelBase or None): ODE model
            tau (int or None): tau value [min], a divisor of 1440
            kwargs: keyword arguments of model parameters

        Returns:
            covsirphy.PhaseUnit: self
        """
        # Tau value
        tau = Validator(tau, "tau").tau(default=None) or self._ode_dict[self.TAU]
        # Model
        model = model or self._model
        if model is None:
            self._ode_dict[self.TAU] = tau
            return self
        self._model = Validator(model, "model").subclass(ModelBase)
        self.info_dict[self.ODE] = model.NAME
        # Parameter values
        param_dict = self._ode_dict.copy()
        param_dict.update(kwargs)
        param_dict = {
            p: param_dict[p] if p in param_dict else None for p in model.PARAMETERS
        }
        self._ode_dict = param_dict.copy()
        self._ode_dict[self.TAU] = tau
        # Day parameters
        if None in param_dict.values():
            return self
        model_instance = model(population=self._population, **param_dict)
        self.info_dict[self.RT] = model_instance.calc_r0()
        # Reproduction number
        if tau is not None:
            self.day_param_dict = model_instance.calc_days_dict(tau)
        return self

    @property
    def record_df(self):
        """
        pandas.DataFrame: records of the phase

            Index
                reset index
            Columns
                - Date (pd.Timestamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - Susceptible (int): the number of susceptible cases
        """
        return self._record_df

    @ record_df.setter
    def record_df(self, df):
        self._ensure_dataframe(df, name="df", columns=self.NLOC_COLUMNS)
        self._record_df = df.copy()
        self.set_y0(df)

    @staticmethod
    def _ensure_dataframe(target, name="df", time_index=False, columns=None, empty_ok=True):
        """
        Ensure the dataframe has the columns.

        Args:
            target (pandas.DataFrame): the dataframe to ensure
            name (str): argument name of the dataframe
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have
            empty_ok (bool): whether give permission to empty dataframe or not

        Returns:
            pandas.DataFrame:
                Index
                    as-is
                Columns:
                    columns specified with @columns or all columns of @target (when @columns is None)
        """
        if not isinstance(target, pd.DataFrame):
            raise TypeError(f"@{name} must be a instance of (pandas.DataFrame).")
        df = target.copy()
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if not empty_ok and target.empty:
            raise ValueError(f"@{name} must not be a empty dataframe.")
        if columns is None:
            return df
        if not set(columns).issubset(df.columns):
            cols_str = ", ".join(col for col in columns if col not in df.columns)
            included = ", ".join(df.columns.tolist())
            s1 = f"Expected columns were not included in {name} with {included}."
            raise KeyError(f"{s1} {cols_str} must be included.")
        return df.loc[:, columns]

    def _model_is_registered(self):
        """
        Ensure that model was set.

        Raises:
            NameError: ODE model is not registered
        """
        if self._model is None:
            raise UnExecutedError("PhaseUnit.set_ode(model)")

    def estimate(self, record_df=None, **kwargs):
        """
        Perform parameter estimation.

        Args:
            record_df (pandas.DataFrame or None)
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any other columns will be ignored
            **kwargs: keyword arguments of Estimator.run()

        Note:
            If @record_df is None, registered records will be used.
        """
        self._model_is_registered()
        # Records
        if record_df is None:
            record_df = self._record_df.copy()
        if record_df.empty:
            raise UnExecutedError(
                "PhaseUnit.record_df = ...", details="or specify @record_df argument")
        self._ensure_dataframe(record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # Check dates
        sta = self._ensure_date(self.start_date)
        end = self._ensure_date(self.end_date)
        series = record_df[self.DATE]
        record_df = record_df.loc[(series >= sta) & (series <= end), :]
        # Parameter estimation of ODE model
        estimator = Estimator(
            record_df, self._model, self._population, **self._ode_dict, **kwargs)
        estimator.run(**kwargs)
        self._read_estimator(estimator, record_df)
        # Set estimator
        self._estimator = estimator

    def _read_estimator(self, estimator, record_df):
        """
        Read the result of parameter estimation and update the summary of phase.

        Args:
            estimator (covsirphy.Estimator): estimator which finished estimation
            record_df (pandas.DataFrame)
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any other columns will be ignored
        """
        # Reproduction number
        est_dict = estimator.to_dict()
        self.info_dict[self.RT] = est_dict.pop(self.RT)
        # Get parameter values and tau value
        ode_set = {*self._model.PARAMETERS, self.TAU}
        ode_dict = {
            k: v for (k, v) in est_dict.items() if k in ode_set}
        self._ode_dict.update(ode_dict)
        # Other information of estimation
        other_dict = dict(est_dict.items() - ode_dict.items())
        self.est_dict.update(other_dict)
        self.est_dict = {k: v for (k, v) in self.est_dict.items() if v is not None}
        # Initial values
        self.set_y0(record_df=record_df)

    def set_y0(self, record_df):
        """
        Set initial values.

        Args:
            record_df (pandas.DataFrame)
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any other columns will be ignored
        """
        self._model_is_registered()
        df = record_df.loc[
            record_df[self.DATE] >= self._ensure_date(self.start_date), :]
        df = self._model.tau_free(df, self._population, tau=None)
        y0_dict = df.iloc[0, :].to_dict()
        self.y0_dict = {
            k: v for (k, v) in y0_dict.items() if k in set(self._model.VARIABLES)
        }

    def simulate(self, y0_dict=None):
        """
        Perform simulation with the set/estimated parameter values.

        Args:
            y0_dict (dict or None): dictionary of initial values or None
                - key (str): variable name
                - value (float): initial value

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Variables of the model (int): Confirmed etc.

        Note:
            Simulation starts at the start date of the phase.
            Simulation end at the next date of the end date of the phase.
        """
        self._model_is_registered()
        # Initial values
        y0_dict = y0_dict or {}
        y0_dict.update(self.y0_dict)
        diff_set = set(self._model.VARIABLES) - y0_dict.keys()
        y0_dict.update({var: 0 for var in diff_set})
        # Conditions
        param_dict = self._ode_dict.copy()
        if None in param_dict.values():
            raise UnExecutedError("PhaseUnit.set_ode()")
        tau = param_dict.pop(self.TAU)
        last_date = self.tomorrow(self._end_date)
        # Simulation
        simulator = ODESimulator()
        simulator.add(
            model=self._model,
            step_n=self.steps(self._start_date, last_date, tau),
            population=self._population,
            param_dict=param_dict,
            y0_dict=y0_dict
        )
        # Dimensionalized values
        df = simulator.dim(tau=tau, start_date=self._start_date)
        df = self._model.restore(df)
        # Return day-level data
        df = df.set_index(self.DATE).resample("D").first()
        df = df.loc[df.index <= self._ensure_date(last_date), :]
        return df.reset_index().loc[:, self.NLOC_COLUMNS]
