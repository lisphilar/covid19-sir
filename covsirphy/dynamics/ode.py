#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from functools import partial
import math
import optuna
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from covsirphy.util.error import UnExpectedNoneError, NotNoneError
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class ODEModel(Term):
    """Basic class of ordinary differential equation (ODE) model.

    Args:
        date_range (tuple of (str, str)): start date and end date of simulation
        tau (int): tau value [min]
        initial_dict (dict of {str: int}): initial values
        param_dict (dict of {str: float}): non-dimensional parameter values
    """
    _logS = "log10(S)"
    _r = "R"
    # Name of ODE model
    _NAME = "ODE Model"
    # Variables
    _VARIABLES = []
    # Non-dimensional parameters
    _PARAMETERS = []
    # Dimensional parameters
    _DAY_PARAMETERS = []
    # Variables that increases monotonically
    _VARS_INCREASE = []
    # Sample data
    _SAMPLE_DICT = {
        "initial_dict": dict.fromkeys(_VARIABLES),
        "param_dict": dict.fromkeys(_PARAMETERS)
    }

    def __init__(self, date_range, tau, initial_dict, param_dict):
        start_date, end_date = Validator(date_range, "date_range", accept_none=False).sequence(length=2)
        self._start = Validator(start_date, name="the first value of @date_range", accept_none=False).date()
        self._end = Validator(
            end_date, name="the second date of @date_range", accept_none=False).date(value_range=(self._start, None))
        self._tau = Validator(tau, "tau", accept_none=False).tau()
        self._initial_dict = Validator(initial_dict, "initial_dict", accept_none=False).dict(
            required_keys=self._VARIABLES, errors="raise")
        self._param_dict = Validator(param_dict, "param_dict", accept_none=False).dict(
            required_keys=self._PARAMETERS, errors="raise")
        # Total population
        self._population = sum(initial_dict.values())
        # Information regarding ODE parameter estimation
        self._estimation_dict = None

    def __str__(self):
        return self._NAME

    def __repr__(self):
        _dict = self.settings()
        return f"{type(self).__name__}({', '.join([f'{k}={v}' for k, v in _dict.items()])})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    @classmethod
    def name(cls):
        """Return name of ODE model.
        """
        return cls._NAME

    @classmethod
    def definitions(cls):
        """Return definitions of ODE model.

        Returns:
            dict of {str: object}:
                - "name" (str): ODE model name
                - "variables" (list of [str]): variable names
                - "parameters" (list of [str]): non-dimensional parameter names
                - "dimensional_parameters" (list of [str]): dimensional parameter names
        """
        return {
            "name": cls._NAME,
            "variables": cls._VARIABLES,
            "parameters": cls._PARAMETERS,
            "dimensional_parameters": cls._DAY_PARAMETERS,
        }

    def settings(self, with_estimation=False):
        """Return settings.

        Args:
            with_estimation (bool): whether includes information regarding ODE parameter estimation or not

        Returns:
            dict of {str: object}:
                - date_range (tuple of (str, str)): start date and end date of simulation
                - tau (int): tau value [min]
                - initial_dict (dict of {str: int}): initial values
                - param_dict (dict of {str: float}): non-dimensional parameter values
                - estimation_dict (dict of {str: str or int}: information regarding ODE parameter estimation, when @with_estimation is True
                    - method (str): method of estimation, "with_quantile" or "with_optimization" or "not_performed"
                    - {metrics} (int): score of hyperparameter optimization, if available
                    - Trials (int) : the number of trials of hyperparameter optimization, if available
                    - Runtime (str): runtime of hyperparameter optimization, if available
                    - keyword arguments set with covsirphy.ODEModel.with_optimization(), if available
                    - keyword arguments set with covsirphy.ODEModel.with_quantile(), if available
        """
        _dict = {
            "date_range": (self._start.strftime(self.DATE_FORMAT), self._end.strftime(self.DATE_FORMAT)),
            "tau": self._tau,
            "initial_dict": self._initial_dict,
            "param_dict": self._param_dict,
        }
        if with_estimation:
            _dict["estimation_dict"] = self._estimation_dict or {"method": "not_performed"}
        return _dict

    @classmethod
    def from_sample(cls, date_range=None, tau=1440):
        """Initialize model with sample data.

        Args:
            date_range (tuple(str or None, str or None) or None): start date and end date of simulation
            tau (int): tau value [min]

        Returns:
            covsirphy.ODEModel: initialized model

        Note:
            When @date_range or the first value of @date_range is None, today when executed will be set as start date.

        Note:
            When @date_range or the second value of @date_range is None, 180 days after start date will be used as end date.
        """
        start_date, end_date = Validator(date_range, "date_range").sequence(default=(None, None), length=2)
        start = Validator(start_date, name="the first value of @date_range").date(default=datetime.now())
        end = Validator(
            end_date, name="the second date of @date_range").date(value_range=(start, None), default=start + timedelta(days=180))
        return cls(date_range=(start, end), tau=tau, **cls._SAMPLE_DICT)

    def _discretize(self, t, X):
        """Discretize the ODE.

        Args:
            t (int): discrete time-steps
            X (numpy.array): the current values of the model

        Returns:
            numpy.array: the next values of the model
        """
        raise NotImplementedError

    def solve(self):
        """Solve an initial value problem.

        Return:
            pandas.DataFrame: analytical solution
                Index
                    Date (pandas.Timestamp): dates from start date to end date
                Columns
                    (pandas.Int64): model-specific dimensional variables of the model
        """
        step_n = math.ceil((self._end - self._start) / timedelta(minutes=self._tau))
        sol = solve_ivp(
            fun=self._discretize,
            t_span=[0, step_n],
            y0=np.array([self._initial_dict[variable] for variable in self._VARIABLES]),
            t_eval=np.arange(0, step_n + 1, 1),
            dense_output=False
        )
        df = pd.DataFrame(data=sol["y"].T.copy(), columns=self._VARIABLES)
        df = self._non_dim_to_date(data=df, tau=self._tau, start_date=self._start)
        return df.round().convert_dtypes()

    @staticmethod
    def _date_to_non_dim(series, tau):
        """Convert date information (TIME) to time(x) = (TIME(x) - TIME(0)) / tau

        Args:
            series (pandas.DatetimeIndex): date information
            tau (int or None): tau value [min]

        Returns:
            pandas.DatetimeIndex: as-is @series when tau is None else converted time information without series name
        """
        Validator(series, "index of data").instance(pd.DatetimeIndex)
        if tau is None:
            return series
        Validator(tau, "tau", accept_none=False).tau()
        converted = (series - series.min()) / np.timedelta64(tau, "m")
        return converted.rename(None).astype("Int64")

    @classmethod
    def _non_dim_to_date(cls, data, tau, start_date):
        """Convert non-dimensional date information (time) to TIME(x) = TIME(0) + tau * time(x) and resample with dates.

        Args:
            data (pandas.DataFrame):
                Index
                    non-dimensional date information pandas.DatetimeIndex
                Columns
                    any columns
            tau (int or None): tau value [min]
            start_date (str or pandas.Timestamp or None): start date of records ie. TIME(0)

        Raises:
            NotNoneError: @tau is None, but start_date is not None
            UnExpectedNoneError: @tau is not None, but start_date is None

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.DatetimeIndex) or as-is @data (when @tau is None)
                Columns
                    any columns of @data
            pandas.DatetimeIndex: as-is @series when tau is None else converted time information without series name

        Note:
            The first values on date will be selected when resampling.
        """
        df = Validator(data, "data").dataframe()
        if (tau is None and start_date is None) or isinstance(df.index, pd.DatetimeIndex):
            return data
        if tau is None:
            raise NotNoneError("start_date", start_date, details="None is required because tau is None")
        if start_date is None:
            raise UnExpectedNoneError("start_date", details="Not None value is required because tau is not None")
        Validator(tau, "tau", accept_none=False).tau()
        start = Validator(start_date, "start_date", accept_none=False).date()
        df[cls.DATE] = (start + pd.Series(df.index * tau).apply(lambda x: timedelta(minutes=x))).tolist()
        return df.set_index(cls.DATE).resample("D").first()

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
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int or None): tau value [min]


        Returns:
            pandas.DataFrame:
                Index
                    as the same as index of @data when @tau is None else converted to time(x) = (TIME(x) - TIME(0)) / tau
                Columns
                    model-specific variables

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    @classmethod
    def inverse_transform(cls, data, tau=None, start_date=None):
        """Transform a dataframe, converting model-specific variables to Susceptible/Infected/Fatal/Recovered.

        Args:
            data (pandas.DataFrame):
                Index
                    any index
                Columns
                    model-specific variables
            tau (int or None): tau value [min]
            start_date (str or pandas.Timestamp or None): start date of records ie. TIME(0)

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.DatetimeIndex) or as-is @data (when either @tau or @start_date are None)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    def r0(self):
        """Calculate basic reproduction number.

        Returns:
            float: reproduction number of the ODE model and parameters

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    def dimensional_parameters(self):
        """Calculate dimensional parameter values.

        Returns:
            dict of {str: int or float}: dictionary of dimensional parameter values

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    @classmethod
    def from_data(cls, data, param_dict, tau=1440, digits=None):
        """Initialize model with data and ODE parameter values.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
            param_dict (dict of {str: float}): non-dimensional parameter values
            tau (int): tau value [min]
            digits (int or None): effective digits of ODE parameter values or None (skip rounding)

        Returns:
            covsirphy.ODEModel: initialized model
        """
        Validator(data, "data", accept_none=False).dataframe(columns=[cls.DATE, *cls._SIRF], empty_ok=False)
        Validator(tau, "tau", accept_none=False).tau()
        Validator(param_dict, "param_dict", accept_none=False).dict(required_keys=cls._PARAMETERS)
        start, end = data[cls.DATE].min(), data[cls.DATE].max()
        trans_df = cls.transform(data=data.set_index(cls.DATE), tau=tau)
        initial_dict = trans_df.iloc[0].to_dict()
        return cls(
            date_range=(start, end), tau=tau, initial_dict=initial_dict,
            param_dict=param_dict if digits is None else {k: Validator(v, k).float(
                value_range=(0, 1), digits=digits) for k, v in param_dict.items()}
        )

    @classmethod
    def from_data_with_quantile(cls, data, tau=1440, q=0.5, digits=None):
        """Initialize model with data, estimating ODE parameters with quantiles.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
            tau (int): tau value [min]
            q (float): the quantiles to compute, values between (0, 1)
            digits (int or None): effective digits of ODE parameter values or None (skip rounding)

        Returns:
            covsirphy.ODEModel: initialized model
        """
        Validator(data, "data", accept_none=False).dataframe(columns=[cls.DATE, *cls._SIRF], empty_ok=False)
        Validator(tau, "tau", accept_none=False).tau()
        Validator(q, "q", accept_none=False).float(value_range=(0, 1))
        trans_df = cls.transform(data=data.set_index(cls.DATE), tau=tau)
        cls_obj = cls.from_data(data=data, param_dict=cls._param_quantile(data=trans_df, q=q), tau=tau, digits=digits)
        cls_obj._estimation_dict = dict(method="with_quantile", tau=tau, q=q, digits=digits)
        return cls_obj

    @classmethod
    def _param_quantile(cls, data, q=0.5):
        """With combinations (X, dX/dt) for variables, calculate quantile values of ODE parameters.

        Args:
            data (pandas.DataFrame): transformed data with covsirphy.ODEModel.transform(data=data, tau=tau)
            q (float or array-like): the quantile(s) to compute, value(s) between (0, 1)

        Returns:
            dict of {str: float or pandas.Series}: parameter values at the quantile(s)

        Note:
            This method must be defined by child classes.
        """
        raise NotImplementedError

    @classmethod
    def _clip(cls, values, lower, upper):
        """
        Trim values at input threshold.

        Args:
            values (float or array-like): values to trim
            lower (float): minimum threshold
            upper (float): maximum threshold

        Returns:
            float or pandas.Series: clipped array
        """
        return min(max(values, lower), upper) if isinstance(values, float) else pd.Series(values).clip()

    @classmethod
    def from_data_with_optimization(cls, data, tau=1440, metric="RMSLE", digits=None, **kwargs):
        """Initialize model with data, estimating ODE parameters hyperparameter optimization using Optuna.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int): tau value [min]
            metric (str): metric to minimize, refer to covsirphy.Evaluator.score()
            digits (int or None): effective digits of ODE parameter values or None (skip rounding)
            **kwargs: keyword arguments of optimization
                - quantiles (tuple(int, int)): quantiles to cut parameter range, like confidence interval, (0.1, 0.9) as default
                - timeout (int): timeout of optimization, 180 as default
                - timeout_iteration (int): timeout of one iteration, 1 as default
                - tail_n (int): the number of iterations to decide whether score did not change for the last iterations, 4 as default
                - allowance (tuple(float, float)): the allowance of the max predicted values, (0.99, 1.01) as default
                - pruner (str): kind of pruner (hyperband, median, threshold or percentile), "threshold" as default
                - upper (float): works for "threshold" pruner, intermediate score is larger than this value, it prunes, 0.5 as default
                - percentile (float): works for "Percentile" pruner, the best intermediate value is in the bottom percentile among trials, it prunes, 50 as default
                - constant_liar (bool): whether use constant liar to reduce search effort or not, False as default

        Returns:
            covsirphy.ODEModel: initialized model
        """
        kwargs_default = {
            "quantiles": (0.1, 0.9),
            "timeout": 180,
            "timeout_iteration": 1,
            "tail_n": 4,
            "allowance": (0.99, 1.01),
            "pruner": "threshold",
            "upper": 0.5,
            "percentile": 50,
            "constant_liar": False,
        }
        kwargs_dict = Validator(kwargs, "kwargs").dict(default=kwargs_default)
        Validator(data, "data", accept_none=False).dataframe(columns=[cls.DATE, *cls._SIRF], empty_ok=False)
        Validator(tau, "tau", accept_none=False).tau()
        Validator([metric], "metric").sequence(candidates=Evaluator.metrics())
        trans_df = cls.transform(data=data.set_index(cls.DATE), tau=tau)
        param_dict, score, n_trials, runtime = cls._estimate_params(
            data=trans_df, tau=tau, metric=metric, **kwargs_dict)
        cls_obj = cls.from_data(data=data, param_dict=param_dict, tau=tau, digits=digits)
        cls_obj._estimation_dict = {
            "method": "with_optimization", metric: score, cls.TRIALS: n_trials, cls.RUNTIME: runtime,
            cls.TAU: tau, "digits": digits, **kwargs_dict, "data": data,
        }
        return cls_obj

    @classmethod
    def _estimate_params(cls, data, tau, metric, **kwargs):
        """Estimate ODE parameter values with hyperparameter optimization.

        Args:
            data (pandas.DataFrame): transformed data with covsirphy.ODEModel.transform(data=data, tau=tau)
            tau (int): tau value [min]
            metric (str): metric to minimize, refer to covsirphy.Evaluator.score()
            **kwargs: keyword arguments of optimization and must includes
                - quantiles (tuple(int, int)): quantiles to cut parameter range, like confidence interval
                - timeout (int): timeout of optimization
                - timeout_iteration (int): timeout of one iteration
                - tail_n (int): the number of iterations to decide whether score did not change for the last iterations
                - allowance (tuple(float, float)): the allowance of the max predicted values
                - pruner (str): kind of pruner (hyperband, median, threshold or percentile)
                - upper (float): works for "threshold" pruner, intermediate score is larger than this value, it prunes
                - percentile (float): works for "Percentile" pruner, the best intermediate value is in the bottom percentile among trials, it prunes
                - constant_liar (bool): whether use constant liar to reduce search effort or not

        Returns:
            tuple of (dict of {str: float}, str, int):
                - dict of {str: float}: dictionary of parameter values
                - float: score with metrics
                - str: runtime of hyperparameter optimization
                - int: the number of trials of hyperparameter optimization
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Create study of optimization
        pruner_dict = {
            "hyperband": optuna.pruners.HyperbandPruner,
            "median": optuna.pruners.MedianPruner,
            "threshold": optuna.pruners.ThresholdPruner,
            "percentile": optuna.pruners.PercentilePruner,
        }
        pruner_name = Validator([kwargs["pruner"]], "pruner", accept_none=False).sequence(
            candidates=pruner_dict.keys(), length=1)[0]
        pruner_class = pruner_dict[pruner_name]
        pruner = pruner_class(**Validator(kwargs, "keyword arguments").kwargs(pruner_class))
        sampler = optuna.samplers.TPESampler(
            **Validator(kwargs, "keyword arguments").kwargs(optuna.samplers.TPESampler))
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        # Create objective function
        value_range_dict = cls._param_quantile(data, q=kwargs["quantiles"])
        objective_func = partial(
            cls._optuna_objective, data=data, tau=tau, value_range_dict=value_range_dict, metric=metric)
        allowance = Validator(kwargs["allowance"], "allowance", accept_none=False).sequence(length=2)
        # Iteration of optimization
        iter_n = math.ceil(kwargs["timeout"] / kwargs["timeout_iteration"])
        stopwatch = StopWatch()
        scores = []
        param_dict = {}
        tail_n = Validator(kwargs["tail_n"], "tail_n").int(value_range=(1, None))
        for _ in range(iter_n):
            # Run iteration
            study.optimize(objective_func, n_jobs=1, timeout=kwargs["timeout_iteration"])
            param_dict = study.best_params.copy()
            # If score did not change in the last iterations, stop running
            scores.append(cls._optuna_score(param_dict, data, tau, metric))
            if len(scores) >= tail_n and len(set(scores[-tail_n:])) == 1:
                break
            # Check max values are in the allowance
            if cls._optuna_is_in_allowance(param_dict, data, tau, allowance):
                break
        return param_dict, scores[-1], len(study.trials), stopwatch.stop_show()

    @classmethod
    def _optuna_objective(cls, trial, data, tau, value_range_dict, metric):
        """Objective function to minimize (evaluation score of the difference of actual data and solved data) by Optuna.

        Args:
            trial (optuna.trial): a trial of the study
            data (pandas.DataFrame): transformed data with covsirphy.ODEModel.transform(data=data, tau=tau)
            tau (int): tau value [min]
            value_range_dict (dict of {str: pandas.Series}): dictionary of value range of ODE parameters
            metric (str): metric to minimize, refer to covsirphy.Evaluator.score()

        Returns:
            float: score to minimize
        """
        param_dict = {}
        for (k, v) in value_range_dict.items():
            try:
                param_dict[k] = trial.suggest_float(k, *v)
            except OverflowError:
                param_dict[k] = trial.suggest_float(k, 0, 1)
        return cls._optuna_score(param_dict, data, tau, metric)

    @classmethod
    def _optuna_score(cls, param_dict, data, tau, metric):
        """Score function to minimize (i.e. evaluation score of the difference of actual data and solved data).

        Args:
            param_dict (dict of {str: float}): non-dimensional parameter values
            data (pandas.DataFrame): transformed data with covsirphy.ODEModel.transform(data=data, tau=tau)
            tau (int): tau value [min]
            metric (str): metric to minimize, refer to covsirphy.Evaluator.score()

        Returns:
            float: score to minimize or positive infinity (when negative values are included in simulated values)
        """
        df = cls._non_dim_to_date(data=data, tau=tau, start_date="01Jan2022")
        model = cls.from_data(data=cls.inverse_transform(df).reset_index(), param_dict=param_dict, tau=tau, digits=None)
        sim_df = model.solve()
        evaluator = Evaluator(df, sim_df, how="inner", on=None)
        return evaluator.score(metric=metric)

    @classmethod
    def _optuna_is_in_allowance(cls, param_dict, data, tau, allowance):
        """
        Return whether all max values of estimated values are in allowance or not.

        Args:
            param_dict (dict of {str: float}): non-dimensional parameter values
            data (pandas.DataFrame): transformed data with covsirphy.ODEModel.transform(data=data, tau=tau)
            tau (int): tau value [min]
            allowance (tuple(float, float)): the allowance of the predicted value

        Returns:
            bool: True when all max values of predicted values are in allowance
        """
        df = cls.inverse_transform(data=data, tau=tau, start_date="01Jan2022").reset_index()
        max_dict = {v: data[v].max() for v in cls._VARIABLES}
        model = cls.from_data(data=df, param_dict=param_dict, tau=tau, digits=None)
        sim_df = model.solve()
        sim_max_dict = {v: sim_df[v].max() for v in cls._VARIABLES}
        # Check all max values are in allowance
        allowance0, allowance1 = allowance
        ok_list = [
            a * allowance0 <= p <= a * allowance1 for (a, p) in zip(max_dict.values(), sim_max_dict.values())]
        return all(ok_list)

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
                    log10(S) (np.float64): common logarithm of S of the ODE model
                    R (np.int64): R of the model
        """
        raise NotImplementedError
