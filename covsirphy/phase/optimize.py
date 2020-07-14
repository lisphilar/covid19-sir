#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from covsirphy.cleaning.term import Term


class Optimizer(Term):
    """
    Hyperparameter optimization with Optuna package.

    Args:
        train_df (pandas.DataFrame): training dataset

            Index:
                reset index
            Columns:
                - Explanatory variable defined by @x
                - Response variables which is not @x
        param (keyword arguments): fixed parameter values
    """
    optuna.logging.disable_default_handler()

    def __init__(self, train_df, x="t", **params):
        self.x = x
        self.y_list = [v for v in train_df.columns if v != x]
        self.train_df = train_df.copy()
        self.y0_dict = train_df.iloc[0, :].to_dict()
        self.step_n = len(train_df)
        self.fixed_dict = params.copy()
        self.study = None
        self.total_trials = 0
        self.run_time = 0

    def _init_study(self, seed=None):
        """
        Initialize Optuna study.

        Args:
            seed (int or None): random seed of hyperparameter optimization

        Notes:
            @seed will effective when the number of CPUs is 1
        """
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )

    def run(self, n_trials, timeout, n_jobs=-1, seed=None):
        """
        Run optimization.
        This method can be overwritten in subclass.

        Args:
            timeout (int): time-out of run
            n_trials (int): the number of trials
            n_jobs (int): the number of parallel jobs or -1 (CPU count)
            seed (int or None): random seed of hyperparameter optimization

        Notes:
            @seed will effective when @n_jobs is 1
        """
        if seed is not None and n_jobs != 1:
            raise ValueError(
                "@seed must be None when @n_jobs is not equal to 1.")
        start_time = datetime.now()
        if self.study is None:
            self._init_study(seed=seed)
        self.study.optimize(
            lambda x: self.objective(x),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs
        )
        end_time = datetime.now()
        self.run_time += (end_time - start_time).total_seconds()
        self.total_trials += n_trials

    def objective(self, trial):
        """
        Objective function of Optuna study.
        This defines the parameter values using Optuna.
        This method should be overwritten in subclass.

        Args:
            trial (optuna.trial): a trial of the study

        Returns:
            (float): score of the error function to minimize
        """
        param_dict = dict()
        return self.error_f(param_dict, self.train_df)

    def error_f(self, param_dict, train_df):
        """
        Definition of error score to minimize in the study.
        This method should be overwritten in subclass.

        Args:
            param_dict (dict): estimated parameter values
                - key (str): parameter name
                - value (int or float): parameter value
            train_df (pandas.DataFrame): actual data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by @variables

        Returns:
            (float): score of the error function to minimize
        """
        sim_df = self.simulate(self.step_n, param_dict)
        comp_df = self.compare(self.train_df, sim_df)
        _ = (sim_df, comp_df)
        return None

    def simulate(self, param_dict):
        """
        Simulate the values with the parameters.
        This method should be overwritten in subclass.

        Args:
            param_dict (dict): estimated parameter values
                - key (str): parameter name
                - value (int or float): parameter value

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Explanatory variable defined by self.x
                    - Response variables which defined by self.y_list
        """
        _ = param_dict.copy()
        df = pd.DataFrame(columns=[self.x, *self.y_list])
        return df

    def compare(self, actual_df, predicted_df):
        """
        Return comparison table.

        Args:
            actual_df (pandas.DataFrame): actual data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by self.y_list

            predicted_df (pandas.DataFrame): predicted data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by self.y_list

        Returns:
            (pandas.DataFrame):
                Index:
                    (str): time step
                Index:
                    reset index
                Columns:
                    - columns with "_actual"
                    - columns with "_predicted:
                    - columns are defined by self.y_list
        """
        # Check the arguments
        if not set(self.y_list).issubset(set(predicted_df.columns)):
            y_str = ", ".join(self.y_list)
            raise KeyError(f"@predicted_df must have {y_str} columns.")
        # Data for comparison
        df = pd.merge(
            actual_df, predicted_df, on=self.x,
            suffixes=(self.A, self.P)
        )
        df = df.set_index(self.x)
        return df

    def param(self):
        """
        Return the estimated parameters as a dictionary.

        Returns:
            (dict)
                - key (str): parameter name
                - value (int or float): parameter value
        """
        param_dict = self.study.best_params.copy()
        param_dict.update(self.fixed_dict)
        return param_dict

    def result(self, name):
        """
        Return the estimated parameters as a dataframe.
        This method should be overwritten in subclass.

        Args:
            name (str): index of the dataframe

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - (estimated parameters)
                    - Trials: the number of trials
                    - Runtime: run time of estimation
        """
        param_dict = self.param()
        # The number of trials
        param_dict["Trials"] = self.total_trials
        # Runtime
        minutes, seconds = divmod(int(self.run_time), 60)
        param_dict["Runtime"] = f"{minutes} min {seconds} sec"
        return param_dict

    def rmsle(self, train_df, dim=1):
        """
        Calculate RMSLE score.
        This method can be overwritten in child class.

        Args:
            train_df (pandas.DataFrame): actual data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by self.y_list
            dim (int or float): dimension where comparison will be performed

        Returns:
            (float): RMSLE score
        """
        predicted_df = self.predict()
        df = self.compare(train_df, predicted_df)
        df = (df * dim + 1).astype(np.int64)
        for v in self.y_list:
            df = df.loc[df[f"{v}{self.A}"] * df[f"{v}{self.P}"] > 0, :]
        a_list = [np.log10(df[f"{v}{self.A}"]) for v in self.y_list]
        p_list = [np.log10(df[f"{v}{self.P}"]) for v in self.y_list]
        diffs = [((a - p) ** 2).sum() for (a, p) in zip(a_list, p_list)]
        score = np.sqrt(sum(diffs) / len(diffs))
        return score

    def predict(self):
        """
        Predict the values with the calculated values.
        This method can be overwritten in subclass.
        """
        param_dict = self.param()
        param_dict.pop(self.TAU)
        return self.simulate(self.step_n, param_dict)

    def history(self, show_figure=True, filename=None):
        """
        Show the history of optimization as a figure
            and return it as dataframe.

        Args:
            show_figure (bool): if True, show the history as a pair-plot of parameters.
            filename (str): filename of the figure, or None (show figure)

        Returns:
            (pandas.DataFrame): the history
        """
        # Create dataframe of the history
        df = self.study.trials_dataframe()
        series = df["datetime_complete"] - df["datetime_start"]
        df["time[s]"] = series.dt.total_seconds()
        df = df.drop(
            ["datetime_complete", "datetime_start"],
            axis=1
        )
        try:
            df = df.drop("system_attrs__number", axis=1)
        except KeyError:
            pass
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

    def accuracy(self, train_df, variables=None, show_figure=True, filename=None):
        """
        Show the accuracy as a figure.
        This method can be overwritten in child class.

        Args:
            train_df (pandas.DataFrame): actual data

                Index:
                    reset index
                Columns:
                    - t: time step, 0, 1, 2,...
                    - includes columns defined by self.y_list

            variables (list[str]): variables to compare or None (all variables)
            show_figure (bool): if True, show the result as a figure
            filename (str): filename of the figure, or None (show figure)
        """
        # Create a table to compare observed/estimated values
        predicted_df = self.predict()
        df = self.compare(train_df, predicted_df)
        if not show_figure:
            return df
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
