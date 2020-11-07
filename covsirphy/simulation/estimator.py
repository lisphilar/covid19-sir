#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from covsirphy.util.stopwatch import StopWatch
from covsirphy.cleaning.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.simulation.estimation_study import EstimationStudy


class Estimator(Term):
    """
    Hyperparameter optimization of an ODE model.

    Args:
        record_df (pandas.DataFrame)
            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
                - any other columns will be ignored
        model (covsirphy.ModelBase): ODE model
        population (int): total population in the place
        tau (int): tau value [min], a divisor of 1440
        kwargs: parameter values of the model

    Notes:
        If some columns are not included, they may be calculated with the model.
    """
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", SyntaxWarning)

    def __init__(self, record_df, model, population, tau=None, **kwargs):
        # Arguments
        self.model = self.ensure_subclass(model, ModelBase, name="model")
        self.variables = model.VARIABLES[:]
        self.population = self.ensure_population(population)
        self.tau = self.ensure_tau(tau)
        self.fixed_dict = {
            k: v for (k, v) in kwargs.items()
            if k in set(self.variables) and v is not None}
        # Dataset: complement the columns with the model
        if not set(self.NLOC_COLUMNS).issubset(record_df.columns):
            record_df = model.restore(record_df)
        self.record_df = self.ensure_dataframe(
            record_df, name="record_df", columns=self.NLOC_COLUMNS)
        # For optimization
        self.est_study = None
        self.total_trials = 0
        self.runtime = 0
        # tau value
        # Check quality of estimation
        self.increasing_cols = [
            f"{v}{self.P}" for v in self.model.VARS_INCLEASE]

    def _create_study(self, seed):
        """
        Create a instance for estimation study.

        Args:
            seed (int or None): random seed of hyperparameter optimization

        Returns:
            covsirphy.EstimationStudy
        """
        return EstimationStudy(
            record_df=self.record_df, model=self.model,
            population=self.population, seed=seed)

    def run(self, timeout=60, reset_n_max=3,
            timeout_iteration=5, allowance=(0.98, 1.02), seed=0):
        """
        Run optimization.
        If the result satisfied the following conditions, optimization ends.
        - all values are not under than 0
        - values of monotonic increasing variables increases monotonically
        - predicted values are in the allowance when each actual value shows max value

        Args:
            timeout (int): time-out of optimization
            reset_n_max (int): if study was reset @reset_n_max times, will not be reset anymore
            timeout_iteration (int): time-out of one iteration
            allowance (tuple(float, float)): the allowance of the predicted value
            seed (int or None): random seed of hyperparameter optimization
        """
        reset_n = 0
        iteration_n = math.ceil(timeout / timeout_iteration)
        stopwatch = StopWatch()
        self.est_study = self._create_study(seed=seed)
        for _ in range(iteration_n):
            self.est_study.run(
                timeout=timeout_iteration, tau=self.tau, **self.fixed_dict)
            # Check monotonic variables
            comp_df = self.est_study.compare()
            if not self._is_monotonic(comp_df):
                if reset_n == reset_n_max - 1:
                    break
                # Initialize the study
                self.est_study = self._create_study(seed=seed)
                reset_n += 1
                continue
            # Need additional trials when the values are not in allowance
            if self._is_in_allowance(comp_df, allowance):
                break
        # Get the results
        self.est_dict = self.est_study.estimated()
        self.total_trials = self.est_study.n_trials
        self.runtime = stopwatch.stop()

    def _is_monotonic(self, comp_df):
        """
        Check that simulated number of cases show mononit increasing.

        Args:
            comp_df (pandas.DataFrame):
                Index:
                    t (int): time step, 0, 1, 2,...
                Columns:
                    - columns with suffix "_actual"
                    - columns with suffix "_predicted"
                    - columns are defined by self.variables

        Returns:
            bool: whether all variable show monotonic increasing or not
        """
        # Check monotonic variables
        mono_ok_list = [
            comp_df[col].is_monotonic_increasing for col in self.increasing_cols
        ]
        return all(mono_ok_list)

    def _is_in_allowance(self, comp_df, allowance):
        """
        Return whether all max values of predicted values are in allowance or not.

        Args:
            comp_df (pandas.DataFrame): [description]
            allowance (tuple(float, float)): the allowance of the predicted value

        Returns:
            bool: True when all max values of predicted values are in allowance
        """
        a_max_values = [comp_df[f"{v}{self.A}"].max() for v in self.variables]
        p_max_values = [comp_df[f"{v}{self.P}"].max() for v in self.variables]
        allowance0, allowance1 = allowance
        ok_list = [
            (a * allowance0 <= p) and (p <= a * allowance1)
            for (a, p) in zip(a_max_values, p_max_values)
        ]
        return all(ok_list)

    def to_dict(self):
        """
        Summarize the results of optimization.

        Args:
            name (str or None): index of the dataframe

        Returns:
            pandas.DataFrame:
                Index:
                    name or reset index (when name is None)
                Columns:
                    - (parameters of the model)
                    - tau
                    - Rt: basic or phase-dependent reproduction number
                    - (dimensional parameters [day])
                    - RMSLE: Root Mean Squared Log Error
                    - Trials: the number of trials
                    - Runtime: run time of estimation
        """
        est_dict = self.est_dict.copy()
        model_instance = self.model(
            population=self.population,
            **{k: v for (k, v) in est_dict.items() if k != self.TAU}
        )
        return {
            **est_dict,
            self.RT: model_instance.calc_r0(),
            **model_instance.calc_days_dict(est_dict[self.TAU]),
            self.RMSLE: self.est_study.rmsle(),
            self.TRIALS: self.total_trials,
            self.RUNTIME: StopWatch.show_time(self.runtime)
        }

    def history(self, show_figure=True, filename=None):
        """
        Show the history of optimization as a figure
            and return it as dataframe.

        Args:
            show_figure (bool): if True, show the history as a pair-plot of parameters.
            filename (str): filename of the figure, or None (show figure)

        Returns:
            pandas.DataFrame: the history
        """
        df = self.est_study.history()
        # Show figure
        if not show_figure:
            return df
        fig_df = df.loc[:, df.columns.str.startswith("params_")]
        fig_df.columns = fig_df.columns.str.replace("params_", "")
        sns.pairplot(fig_df, diag_kind="kde", markers="+")
        # Save figure or show figure
        if filename is None:
            plt.show()
            return df
        plt.savefig(filename, bbox_inches="tight", transparent=False, dpi=300)
        plt.clf()
        return df

    def accuracy(self, show_figure=True, filename=None):
        """
        Show the accuracy as a figure.

        Args:
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)

        Returns:
            pandas.DataFrame:
                Index:
                    t (int): time step, 0, 1, 2,...
                Columns:
                    - columns with suffix "_actual"
                    - columns with suffix "_predicted"
                    - columns are defined by self.variables
        """
        # Create a table to compare observed/estimated values
        df = self.est_study.compare()
        if not show_figure:
            return df
        # Variables to show the accuracy
        variables = [
            v for (i, (p, v))
            in enumerate(zip(self.model.WEIGHTS, self.model.VARIABLES))
            if p != 0 and i != 0
        ]
        # Prepare figure object
        val_len = len(variables) + 1
        fig, axes = plt.subplots(
            ncols=1, nrows=val_len, figsize=(9, 6 * val_len / 2)
        )
        # Comparison of each variable
        for (ax, v) in zip(axes.ravel()[1:], variables):
            df[[f"{v}{self.A}", f"{v}{self.P}"]].plot.line(
                ax=ax, ylim=(0, None), sharex=True,
                title=f"{self.model.NAME}: Comparison of observed/estimated {v}(t)"
            )
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax.legend(
                bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0
            )
        # Summarize in a figure
        for v in variables:
            df[f"{v}_diff"] = df[f"{v}{self.A}"] - df[f"{v}{self.P}"]
            df[f"{v}_diff"].plot.line(
                ax=axes.ravel()[0], sharex=True,
                title=f"{self.model.NAME}: observed - estimated"
            )
        axes.ravel()[0].axhline(y=0, color="black", linestyle="--")
        axes.ravel()[0].yaxis.set_major_formatter(
            ScalarFormatter(useMathText=True))
        axes.ravel()[0].ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0))
        axes.ravel()[0].legend(bbox_to_anchor=(1.02, 0),
                               loc="lower left", borderaxespad=0)
        fig.tight_layout()
        # Save figure or show figure
        if filename is None:
            plt.show()
            return df
        plt.savefig(filename, bbox_inches="tight", transparent=False, dpi=300)
        plt.clf()
        return df
