#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import ruptures as rpt
from covsirphy.util.term import Term


class _SRChange(Term):
    """
    Perform S-R trend analysis.

    Args:
        sr_df (pandas.DataFrame)
            Index
                Date (pd.TimeStamp): Observation date
            Columns
                - Recovered (int): the number of recovered cases (>0)
                - Susceptible (int): the number of susceptible cases
                - any other columns will be ignored
    """

    def __init__(self, sr_df):
        self._ensure_dataframe(sr_df, name="sr_df", time_index=True, columns=[self.S, self.R])
        # Index: Date, Columns: R, logS
        self._sr_df = pd.DataFrame(
            {
                "R": sr_df[self.R],
                "logS": np.log10(sr_df[self.S].astype(np.float64)),
            }
        )

    def run(self, min_size):
        """
        Run optimization and return the change points.

        Args:
            min_size (int): minimum value of phase length [days]

        Returns:
            list[pandas.Timestamp]: list of change points
        """
        self._ensure_natural_int(min_size, name="min_size")
        # Index: "R", Columns: logS
        df = self._sr_df.pivot_table(index="R", values="logS", aggfunc="last")
        df.index.name = None
        # Detect change points with Ruptures package: reset index + 1 values will be returned
        algorithm = rpt.KernelCPD(kernel="rbf", min_size=min_size)
        results = algorithm.fit_predict(df.iloc[:, 0].to_numpy(), pen=0.5)[:-1]
        # Convert reset index + 1 values to logS
        logs_df = df.iloc[[result - 1 for result in results]]
        # Convert logS to dates
        merge_df = pd.merge_asof(
            logs_df.sort_values("logS"), self._sr_df.reset_index().sort_values("logS"),
            on="logS", direction="nearest")
        return merge_df[self.DATE].sort_values().tolist()
