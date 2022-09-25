#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from covsirphy.util.validator import Validator
from covsirphy.visualization.vbase import VisualizeBase, find_args


class ComparePlot(VisualizeBase):
    """Compare two groups with specified variables.

    Args:
        filename (str or None): filename to save the figure or None (display)
        bbox_inches (str): bounding box in inches when creating the figure
        kwargs: the other arguments of matplotlib.pyplot.savefig()
    """

    def __init__(self, filename=None, bbox_inches="tight", **kwargs):
        self._filename = filename
        self._savefig_dict = {"bbox_inches": bbox_inches, **kwargs}
        # Properties
        self._title = ""
        self._variables = []
        self._ax = None

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        return super().__exit__(*exc_info)

    def plot(self, data, variables, groups):
        """Compare two groups with specified variables.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    x values
                Columns
                    y variables to show, "{variable}_{group}" for all combinations of variables and groups
            variables (list[str]): variables to compare
            groups (list[str]): the first group name and the second group name
        """
        Validator(variables, "variables").sequence()
        group1, group2 = Validator(groups, "groups").sequence()
        col_nest = [[f"{variable}_{group}" for group in groups] for variable in variables]
        Validator(data, "data").dataframe(columns=sum(col_nest, []))
        # Prepare figure object
        fig_len = len(variables) + 1
        _, self._ax = plt.subplots(ncols=1, nrows=fig_len, figsize=(9, 6 * fig_len / 2))
        # Compare each variable
        for (ax, v, columns) in zip(self._ax.ravel()[1:], variables, col_nest):
            data[columns].plot.line(
                ax=ax, ylim=(None, None), sharex=True, title=f"Comparison regarding {v}(t)")
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
        # Show residuals
        for (v, columns) in zip(variables, col_nest):
            data[f"{v}_diff"] = data[columns[0]] - data[columns[1]]
            data[f"{v}_diff"].plot.line(
                ax=self._ax.ravel()[0], sharex=True,
                title=f"{group1.capitalize()} - {group2.capitalize()}")
        self._ax.ravel()[0].axhline(y=0, color="black", linestyle="--")
        self._ax.ravel()[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        self._ax.ravel()[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self._ax.ravel()[0].legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)


def compare_plot(df, variables, groups, filename=None, **kwargs):
    """Wrapper function: show chronological change of the data.

    Args:
        df (pandas.DataFrame): data to show
            Index
                x values
            Columns
                y variables to show, "{variable}_{group}" for all combinations of variables and groups
        variables (list[str]): variables to compare
        groups (list[str]): the first group name and the second group name
        filename (str or None): filename to save the figure or None (display)
        kwargs: keyword arguments of the following classes and methods.
            - matplotlib.pyplot.savefig()
            - matplotlib.pyplot.legend()
    """
    with ComparePlot(filename=filename, **find_args(plt.savefig, **kwargs)) as cp:
        cp.plot(data=df, variables=variables, groups=groups)
