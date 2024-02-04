from __future__ import annotations
from datetime import timedelta
from functools import partial
from multiprocessing import cpu_count, Pool
import warnings
import numpy as np
import pandas as pd
from p_tqdm import p_umap
from typing_extensions import Self
from covsirphy.util.config import config
from covsirphy.util.error import EmptyError, NotEnoughDataError, UnExpectedNoneError, UnExpectedValueRangeError
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.stopwatch import StopWatch
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.visualization.compare_plot import compare_plot
from covsirphy.dynamics.ode import ODEModel
from covsirphy.dynamics._trend import _TrendAnalyzer
from covsirphy.dynamics._simulator import _Simulator


class Dynamics(Term):
    """Class to hand phase-dependent SIR-derived ODE models.

    Args:
        model: definition of ODE model
        date_range: start date and end date of dynamics to analyze
        tau: tau value [min] or None (set later with data)
        name: name of dynamics to show in figures (e.g. "baseline") or None (un-set)
    """

    def __init__(self, model: ODEModel, date_range: tuple[str | None, str | None], tau: int | None = None, name: str | None = None) -> None:
        self._model = Validator(model, "model", accept_none=False).subclass(ODEModel)
        first_date, last_date = Validator(date_range, "date_range", accept_none=False).sequence(length=2)
        self._first = Validator(first_date, name="the first value of @date_range", accept_none=False).date()
        self._last = Validator(
            last_date, name="the second date of @date_range", accept_none=False).date(value_range=(self._first, None))
        self._tau = Validator(tau, "tau", accept_none=True).tau()
        self._name = None if name is None else Validator(name, "name").instance(str)
        # Index: Date, Columns: S, I, F, R, ODE parameters
        self._parameters = self._model._PARAMETERS[:]
        self._df = pd.DataFrame(
            {self._PH: 0}, index=pd.date_range(start=self._first, end=self._last, freq="D"),
            columns=[self._PH, *self._SIRF, *self._parameters])

    def __len__(self) -> int:
        return self._df[self._PH].nunique()

    @property
    def model(self) -> ODEModel:
        """Return model class.
        """
        return self._model

    @property
    def model_name(self) -> str:
        """Return name of ODE model.
        """
        return self._model._NAME

    @property
    def tau(self) -> int | None:
        """Return tau value [min] or None (un-set).
        """
        return self._tau

    @tau.setter
    def tau(self, value: int | None) -> None:
        self._tau = Validator(value, "tau", accept_none=True).tau()

    @tau.deleter
    def tau(self) -> None:
        self._tau = None

    @property
    def name(self) -> str | None:
        """Return name of dynamics to show in figures (e.g. "baseline") or None (un-set).
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        self._name = Validator(name, "name").instance(str)

    @name.deleter
    def name(self) -> None:
        self._name = None

    @classmethod
    def from_sample(cls, model: ODEModel, date_range: tuple[str | None, str | None] | None = None, tau: int = 1440) -> Self:
        """Initialize model with sample data of one-phase ODE model.

        Args:
            model: definition of ODE model
            date_range: start date and end date of simulation
            tau value [min]

        Returns:
            initialized model

        Note:
            Regarding @date_range, refer to covsirphy.ODEModel.from_sample().
        """
        Validator(model, "model", accept_none=False).subclass(ODEModel)
        model_instance = model.from_sample(date_range=date_range, tau=tau)
        settings_dict = model_instance.settings()
        variable_df = model.inverse_transform(model_instance.solve()).iloc[[0]]
        param_df = pd.DataFrame(settings_dict["param_dict"], index=[pd.to_datetime(settings_dict["date_range"][0])])
        param_df.index.name = cls.DATE
        df = pd.concat([variable_df, param_df], axis=1)
        instance = cls(model=model, date_range=settings_dict["date_range"], tau=tau, name="Sample data")
        instance.register(data=df)
        return instance

    @classmethod
    def from_data(cls, model: ODEModel, data: pd.DataFrame, tau: int | None = 1440, name: str | None = None) -> Self:
        """Initialize model with data.

        Args:
            data: new data to overwrite the current information
                Index
                    Date (pandas.Timestamp): Observation dates
                Columns
                    Susceptible (int): the number of susceptible cases
                    Infected (int): the number of currently infected cases
                    Fatal (int): the number of fatal cases
                    Recovered (int): the number of recovered cases
                    (numpy.float64): ODE parameter values defined with the ODE model (optional)
            tau: tau value [min] or None (un-set)
            name: name of dynamics to show in figures (e.g. "baseline") or None (un-set)

        Returns:
            initialized model

        Note:
            Regarding @date_range, refer to covsirphy.ODEModel.from_sample().
        """
        Validator(model, "model", accept_none=False).subclass(ODEModel)
        Validator(data, "data").dataframe(time_index=True)
        instance = cls(model=model, date_range=(data.index.min(), data.index.max()), tau=tau, name=name)
        instance.register(data=data)
        return instance

    def register(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Register data to get initial values and ODE parameter values (if available).

        Args:
            data: new data to overwrite the current information or None (no new records)
                Index
                    Date (pandas.Timestamp): Observation dates
                Columns
                    Susceptible (int): the number of susceptible cases
                    Infected (int): the number of currently infected cases
                    Recovered (int): the number of recovered cases
                    Fatal (int): the number of fatal cases
                    (numpy.float64): ODE parameter values defined with the model

        Returns:
            dataframe of the current information:
                Index
                    Date (pandas.Timestamp): Observation dates
                Columns
                    Susceptible (int): the number of susceptible cases
                    Infected (int): the number of currently infected cases
                    Recovered (int): the number of recovered cases
                    Fatal (int): the number of fatal cases
                    (numpy.float64): ODE parameter values defined with model.PARAMETERS

        Note:
            Change points of ODE parameter values will be recognized as the change points of phases.

        Note:
            NA can used in the newer phases because filled with that of the older phases.
        """
        if data is not None:
            new_df = Validator(data, "data").dataframe(time_index=True)
            new_df.index = pd.to_datetime(new_df.index).round("D")
            all_df = pd.DataFrame(
                np.nan,
                index=self._df.index,
                columns=self._df.columns,
            )
            all_df[self._PH] = 0
            for col in new_df:
                new_df[col] = new_df[col].astype(pd.Float64Dtype())
                all_df[col] = all_df[col].astype(pd.Float64Dtype())
            all_df.update(new_df, overwrite=True)
            if all_df.loc[self._first, self._SIRF].isna().any():
                raise EmptyError(
                    f"records on {self._first.strftime(self.DATE_FORMAT)}", details="Records must be registered for simulation")
            if all_df.min().min() < 0:
                raise UnExpectedValueRangeError("minimum value of the data", all_df.min().min(), (0, None))
            all_df.index.name = self.DATE
            self._df = all_df.convert_dtypes()
            # Find change points with parameter values
            param_df = all_df.loc[:, self._parameters].ffill().drop_duplicates().dropna(axis=0)
            if not param_df.empty:
                self._segment(points=param_df.index.tolist(), overwrite=True)
        return self._df.loc[:, [*self._SIRF, *self._parameters]]

    def _segment(self, points: list[str], overwrite: bool) -> None:
        """Perform time-series segmentation with points.

        Args:
            points: dates of change points
            overwrite: whether remove all phases before segmentation or not

        Note:
            @points can include the first date, but not required.

        Note:
            @points must be selected from the first date to three days before the last date specified covsirphy.Dynamics(date_range).
        """
        point_dates = [Validator(point, "a change point", accept_none=False).date() for point in points]
        candidates = pd.date_range(start=self._first, end=self._last - timedelta(days=2), freq="D")
        change_points = Validator(point_dates, "points", accept_none=False).sequence(unique=True, candidates=candidates)
        df = self._df.copy()
        if overwrite:
            df[self._PH] = 0
        for point in change_points:
            df.loc[point:, self._PH] += 1
        self._df = df.convert_dtypes()

    def segment(self, points: list[str] | None = None, overwrite: bool = False, **kwargs) -> Self:
        """Perform time-series segmentation with points manually selected or found with S-R trend analysis.

        Args:
            points: dates of change points or None (will be found with S-R trend analysis via .detect() method)
            overwrite: whether remove all phases before segmentation or not
            **kwargs: keyword arguments of covsirphy.Dynamics.detect()

        Returns:
            Updated Dynamics object

        Note:
            @points can include the first date, but not required.

        Note:
            @points must be selected from the first date to three days before the last date specified covsirphy.Dynamics(date_range).
        """
        self._segment(points=points or self.detect(**kwargs)[0], overwrite=overwrite)
        return self

    def detect(self, algo: str = "Binseg-normal", min_size: int = 7, display: bool = True, **kwargs) -> tuple[pd.Timestamp, pd.DataFrame]:
        """Perform S-R trend analysis to find change points of log10(S) - R of model-specific variables, not that segmentation requires .segment() method.

        Args:
            algo: detection algorithms and models
            min_size: minimum value of phase length [days], be equal to or over 3
            display: whether display figure of log10(S) - R plane or not
            **kwargs: keyword arguments of algorithm classes (ruptures.Pelt, .Binseg, BottomUp) except for "model",
                covsirphy.VisualizeBase(), matplotlib.legend.Legend()

        Raises:
            NotEnoughDataError: we have not enough records, the length of the records must be equal to or over min_size * 2

        Returns:
            - pandas.Timestamp: date of change points
            - pandas.Dataframe:
                Index
                    R (int): actual R (R of the ODE model) values
                Columns
                    Actual (float): actual log10(S) (common logarithm of S of the ODE model) values
                    Fitted (float): log10(S) values fitted with y = a * R + b
                    0th (float): log10(S) values fitted with y = a * R + b and 0th phase data
                    1st, 2nd... (float): fitted values of 1st, 2nd phases

        Note:
            - Python library `ruptures` will be used for off-line change point detection.
            - Refer to documentation of `ruptures` library, https://centre-borelli.github.io/ruptures-docs/
            - Candidates of @algo are "Pelt-rbf", "Binseg-rbf", "Binseg-normal", "BottomUp-rbf", "BottomUp-normal".

        Note:
            - S-R trend analysis is original to Covsirphy, https://www.kaggle.com/code/lisphilar/covid-19-data-with-sir-model/notebook
            - "Phase" means a sequential dates in which the parameters of SIR-derived models are fixed.
            - "Change points" means the dates when trend was changed.
            - "Change points" is the same as the start dates of phases except for the 0th phase.
        """
        Validator(min_size, "min_size", accept_none=False).int(value_range=(3, None))
        df = self._df.dropna(how="any", subset=self._SIRF)
        if len(df) < min_size * 2:
            raise NotEnoughDataError("the records of the number of cases without NAs", df, required_n=min_size * 2)
        analyzer = _TrendAnalyzer(data=df, model=self._model, min_size=min_size)
        points = analyzer.find_points(algo=algo, **kwargs)
        fit_df = analyzer.fitting(points=points)
        if display:
            analyzer.display(points=points, fit_df=fit_df, name=self._name, **kwargs)
        return points, fit_df

    def summary(self) -> pd.DataFrame:
        """Summarize phase information.

        Returns:
            Summarized information.
                Index
                    Phase (str): phase names, 0th, 1st,...
                Columns
                    Start (pandas.Timestamp): start date of the phase
                    End (pandas.Timestamp): end date of the phase
                    Rt (float): phase-dependent reproduction number (if parameters are available)
                    (float): parameter values, including rho (if available)
                    (int or float): dimensional parameters, including 1/beta [days] (if tau and parameters are available)
        """
        df = self._df.reset_index()
        df[self._PH], _ = df[self._PH].factorize()
        first_df = df.groupby(self._PH).first()
        df = first_df.join(df.groupby(self._PH).last(), rsuffix="_last")
        df = df.rename(columns={self.DATE: self.START, f"{self.DATE}_last": self.END})
        df = df.loc[:, [col for col in df.columns if "_last" not in col]]
        df.index = [self.num2str(num) for num in df.index]
        df.index.name = self.PHASE  # type: ignore
        # Reproduction number
        df[self.RT] = df[self._parameters].apply(
            lambda x: np.nan if x.isna().any() else self._model.from_data(data=self._df.reset_index(), param_dict=x.to_dict(), tau=self._tau).r0(), axis=1)
        # Day parameters
        if self._tau is not None:
            days_df = df[self._parameters].apply(
                lambda x: np.nan if x.isna().any() else self._model.from_data(
                    data=self._df.reset_index(), param_dict=x.to_dict(), tau=self._tau).dimensional_parameters(),
                axis=1, result_type="expand"
            )
            df = pd.concat([df, days_df], axis=1)
        # Set the order of columns
        fixed_cols = [
            self.START, self.END, self.RT, *self._model._PARAMETERS, *self._model._DAY_PARAMETERS]
        others = [col for col in df.columns if col not in set(fixed_cols) | set(self._SIRF)]
        return df.reindex(columns=[*fixed_cols, *others]).dropna(how="all", axis=1).ffill().convert_dtypes()

    def track(self) -> pd.DataFrame:
        """Track reproduction number, parameter value and dimensional parameter values.

        Returns:
            Dataframe of time-series data of the values.
                Index
                    Date (pandas.Timestamp): dates
                Columns
                    Rt (float): phase-dependent reproduction number (if parameters are available)
                    (float): parameter values, including rho (if available)
                    (int or float): dimensional parameters, including 1/beta [days] (if tau and parameters are available)
        """
        df = self.summary()
        df[self.DATE] = df[[self.START, self.END]].apply(
            lambda x: pd.date_range(start=x[self.START], end=x[self.END], freq="D"), axis=1)
        return df.explode(self.DATE).set_index(self.DATE).drop([self.START, self.END], axis=1)

    def simulate(self, model_specific: bool = False) -> pd.DataFrame:
        """Perform simulation with phase-dependent ODE model.

        Args:
            model_specific (bool): whether convert S, I, F, R to model-specific variables or not

        Raises:
            UnExpectedNoneError: tau value is un-set
            NAFoundError: ODE parameter values on the start dates of phases are un-set

        Returns:
            dataframe of time-series simulated data.
                Index
                    Date (pd.Timestamp): dates
                Columns
                    if @model_specific is False:
                    Susceptible (int): the number of susceptible cases
                    Infected (int): the number of currently infected cases
                    Recovered (int): the number of recovered cases
                    Fatal (int): the number of fatal cases
                    if @model_specific is True, variables defined by model.VARIABLES of covsirphy.Dynamics(model)
        """
        if self._tau is None:
            raise UnExpectedNoneError(
                "tau", details="Tau value must be set with covsirphy.Dynamics(tau) or covsirphy.Dynamics.tau or covsirphy.Dynamics.estimate_tau()")
        simulator = _Simulator(model=self._model, data=self._df)
        return simulator.run(tau=self._tau, model_specific=model_specific).set_index(self.DATE)

    def estimate(self, **kwargs) -> Self:
        """Run covsirphy.Dynamics.estimate_tau() and covsirphy.Dynamics.estimate_params().

        Args:
            **kwargs: keyword arguments of covsirphy.Dynamics.estimate_tau() and covsirphy.Dynamics.estimate_params()

        Returns:
            Updated Dynamics object with estimated ODE parameter values.
        """
        self.estimate_tau(**Validator(kwargs).kwargs(self.estimate_tau))
        self.estimate_params(**kwargs)
        return self

    def estimate_tau(self, metric: str = "RMSLE", q: float = 0.5, digits: int | None = None, n_jobs: int | None = None) -> tuple[float, pd.DataFrame]:
        """Set the best tau value for the registered data, estimating ODE parameters with quantiles.

        Args:
            metric: metric name for scoring when selecting best tau value
            q: the quantiles to compute, values between (0, 1)
            digits: effective digits of ODE parameter values or None (skip rounding)
            n_jobs: the number of parallel jobs or None (CPU count)

        Raises:
            NotEnoughDataError: less than three non-NA records are registered

        Returns:
            - float: tau value with best metric score
            - pandas.DataFrame: metric scores of tau candidates
                Index
                    tau (int): candidate of tau values
                Columns
                    {metric}: score of estimation with metric
        """
        all_df = self._df.dropna(how="any", subset=self._SIRF)
        if len(all_df) < 3:
            raise NotEnoughDataError("registered S/I/F/R data except NAs", all_df, 3)
        score_f = partial(self._score_with_tau, metric=metric, q=q, digits=digits)
        divisors = [i for i in range(1, 1441) if 1440 % i == 0]
        n_jobs_validated = Validator(n_jobs, "n_jobs").int(value_range=(1, cpu_count()), default=cpu_count())
        with Pool(n_jobs_validated) as p:
            scores = p.map(score_f, divisors)
        score_dict = dict(zip(divisors, scores))
        comp_f = {True: min, False: max}[Evaluator.smaller_is_better(metric=metric)]
        self._tau = comp_f(score_dict.items(), key=lambda x: x[1])[0]
        return self._tau, pd.DataFrame.from_dict(score_dict, orient="index", columns=[metric])

    def _score_with_tau(self, tau: int, metric: str, q: float, digits: int | None) -> float:
        """Return the metric score with tau.

        Args:
            tau: tau value [min]
            metric: metric name for scoring when selecting best tau value
            q: the quantiles to compute, values between (0, 1)
            digits: effective digits of ODE parameter values or None (skip rounding)

        Returns:
            metric score
        """
        parameters = self._model._PARAMETERS[:]
        all_df = self._df.dropna(how="any", subset=self._SIRF)
        all_df[parameters] = all_df.loc[:, parameters].astype("Float64")
        starts = all_df.reset_index().groupby(self._PH)[self.DATE].first().sort_values()
        ends = all_df.reset_index().groupby(self._PH)[self.DATE].last().sort_values()
        for start, end in zip(starts, ends):
            model_instance = self._model.from_data_with_quantile(
                data=all_df.loc[start: end].reset_index(), tau=tau, q=q, digits=digits)
            all_df.loc[start, parameters] = pd.Series(model_instance.settings()["param_dict"])
        simulator = _Simulator(model=self._model, data=all_df)
        sim_df = simulator.run(tau=tau, model_specific=False).set_index(self.DATE)
        evaluator = Evaluator(all_df[self._SIRF], sim_df[self._SIRF], how="inner")
        return evaluator.score(metric=metric)

    def estimate_params(self, metric: str = "RMSLE", digits: int | None = None, n_jobs: int | None = None, **kwargs) -> pd.DataFrame:
        """Set ODE parameter values optimized for the registered data with hyperparameter optimization using Optuna.

        Args:
            metric: metric name for scoring when optimizing ODE parameter values of phases
            digits: effective digits of ODE parameter values or None (skip rounding)
            n_jobs: the number of parallel jobs or None (CPU count)
            **kwargs: keyword arguments of optimization, refer to covsirphy.ODEModel.from_data_with_optimization()

        Raises:
            UnExpectedNoneError: tau value is un-set
            NotEnoughDataError: less than three non-NA records are registered

        Returns:
            Index
                Date (pandas.Timestamp): dates
            Columns
                (numpy.float64): ODE parameter values defined with model.PARAMETERS
                {metric}: score with the estimated parameter values
                Trials (int): the number of trials
                Runtime (str): runtime of optimization, like 0 min 10 sec
        """
        if self._tau is None:
            raise UnExpectedNoneError(
                "tau", details="Tau value must be set with covsirphy.Dynamics(tau) or covsirphy.Dynamics.tau or covsirphy.Dynamics.estimate_tau()")
        all_df = self._df.loc[:, [self._PH, *self._SIRF]].dropna(how="any")
        if len(all_df) < 3:
            raise NotEnoughDataError("registered S/I/F/R data except NAs", all_df, 3)
        n_jobs_validated = Validator(n_jobs, "n_jobs").int(value_range=(1, cpu_count()), default=cpu_count())
        starts = all_df.reset_index().groupby(self._PH)[self.DATE].first().sort_values()
        ends = all_df.reset_index().groupby(self._PH)[self.DATE].last().sort_values()
        est_f = partial(
            self._optimized_params, model=self._model, tau=self._tau, metric=metric, digits=digits, **kwargs)
        phase_dataframes = [all_df[start: end] for start, end in zip(starts, ends)]
        config.info(f"\n<{self._model._NAME}: parameter estimation>")
        config.info(f"Running optimization with {n_jobs_validated} CPUs...")
        stopwatch = StopWatch()
        # p-tqdm with Python 3.12: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        results = p_umap(est_f, phase_dataframes, num_cpus=n_jobs_validated)
        config.info(f"Completed optimization. Total: {stopwatch.stop_show()}\n")
        est_df = pd.concat(results, sort=True, axis=0)
        est_df = est_df.loc[:, [*self._parameters, metric, self.TRIALS, self.RUNTIME]].ffill().convert_dtypes()
        # Update registered parameter values
        r_df = self.register()
        for col in self._parameters:
            r_df[col] = r_df[col].astype(pd.Float64Dtype())
        r_df.update(est_df, overwrite=True)
        self.register(data=r_df)
        return est_df

    def _optimized_params(self, phase_df: pd.DataFrame, model: ODEModel, tau: int, metric: str, digits: int | None, **kwargs) -> pd.DataFrame:
        """Return ODE parameter values optimized with the registered data, estimating ODE parameters hyperparameter optimization using Optuna.

        Args:
            phase_df: records of a phase
                Index
                    Date (pandas.Timestamp): observation dates
                Columns
                    variables of the model
            model: definition of ODE model
            tau: tau value [min]
            metric: metric name for scoring when optimizing ODE parameter values of phases
            digits: effective digits of ODE parameter values or None (skip rounding)
            n_jobs: the number of parallel jobs or None (CPU count)
            **kwargs: keyword arguments of optimization, refer to covsirphy.ODEModel.from_data_with_optimization()

        Raises:
            UnExpectedNoneError: tau value is un-set

        Returns:
            Index
                Date (pandas.Timestamp): dates
            Columns
                (numpy.float64): ODE parameter values defined with model.PARAMETERS
                {metric}: score with the estimated parameter values
                Trials (int): the number of trials
                Runtime (str): runtime of optimization, like 0 min 10 sec
        """
        df = phase_df.copy()
        # ODE parameter optimization
        model_instance = model.from_data_with_optimization(
            data=df.reset_index(), tau=tau, metric=metric, digits=digits, **kwargs)
        df.loc[df.index[0], model._PARAMETERS] = pd.Series(model_instance.settings()["param_dict"])
        # Get information regarding optimization
        est_dict = model_instance.settings(with_estimation=True)["estimation_dict"]
        est_dict = {k: v for k, v in est_dict.items() if k in {metric, self.TRIALS, self.RUNTIME}}
        warnings.filterwarnings("ignore", category=FutureWarning)
        df.loc[df.index[0], list(est_dict.keys())] = pd.Series(est_dict)
        return df

    def parse_phases(self, phases: list[str] | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return minimum date and maximum date of the phases.

        Args:
            phases: phases (0th, 1st, 2nd,... last) or None (all phases)

        Returns:
            minimum date and maximum date of the phases

        Note:
            "last" can be used to specify the last phase.
        """
        if phases is None:
            return self._first, self._last
        all_df = self._df.copy()
        all_df[self._PH], _ = all_df[self._PH].factorize()
        phase_numbers = [all_df[self._PH].max() if ph == "last" else self.str2num(ph) for ph in phases]
        df = all_df.loc[all_df[self._PH].isin(phase_numbers)]
        # FutureWarning to be fixed by pandas version 3.0.0 release
        warnings.filterwarnings("ignore", category=FutureWarning)
        return df.index.min(), df.index.max()

    def parse_days(self, days: int, ref: pd.Timestamp | str | None = "last") -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return min(ref, ref + days) and max(ref, ref + days).

        Args:
            days: the number of days
            ref: reference date or "first" (the first date of records) or "last"/None (the last date)

        Returns:
            minimum date and maximum date of the selected dates

        Note:
            Note that the days clipped with the first and the last dates of records.
        """
        days_n = Validator(days, "days", accept_none=False).int()
        ref_dict = {"first": self._first, "last": self._last}
        ref_date = Validator(ref_dict.get(ref, ref) if isinstance(ref, str) else ref, name="ref").date(
            value_range=(self._first, self._last), default=self._last)
        min_date = min(ref_date, ref_date + timedelta(days=days_n))
        max_date = max(ref_date, ref_date + timedelta(days=days_n))
        return max(min_date, self._first), min(max_date, self._last)

    def evaluate(self, date_range: tuple[str | pd.Timestamp | None, str | pd.Timestamp | None] | None = None, metric: str = "RMSLE", display: bool = True, **kwargs) -> float:
        """Compare the simulated results and actual records, and evaluate the differences.

        Args:
            date_range: range of dates to evaluate or None (the first and the last date)
            metric: metric to evaluate the difference
            display: whether display figure of comparison or not
            kwargs: keyword arguments of covsirphy.compare_plot()

        Returns:
            evaluation score
        """
        variables = [self.CI, self.F, self.R]
        start_date, end_date = Validator(date_range, name="date_range").sequence(
            default=(self._first, self._last), length=2)
        start = Validator(start_date, "date_range[0]").date(value_range=(self._first, self._last), default=self._first)
        end = Validator(end_date, "date_range[1]").date(value_range=(self._first, self._last), default=self._last)
        actual_df = self._df.loc[start:end, variables].dropna(how="any", axis=0)
        sim_df = self.simulate(model_specific=False).loc[start: end, variables].dropna(how="any", axis=0)
        df = actual_df.join(sim_df, how="inner", lsuffix="_actual", rsuffix="_simulated")
        if display:
            compare_plot(df, variables=variables, groups=["actual", "simulated"], **kwargs)
        return Evaluator(actual_df, sim_df).score(metric=metric)

    def start_dates(self) -> list[str]:
        """Return the start dates of phases.

        Returns:
            start dates
        """
        df = self._df.reset_index()
        df[self._PH], _ = df[self._PH].factorize()
        return df.groupby(self._PH).first()[self.DATE].tolist()
