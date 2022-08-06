#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from covsirphy.util.error import deprecate
from covsirphy.util.validator import Validator
from covsirphy.visualization.vbase import find_args
from covsirphy.visualization.line_plot import LinePlot


class TrendPlot(LinePlot):
    """
    Create line plot with actual values for S-R trend analysis.

    Args:
        filename (str or None): filename to save the figure or None (display)
        bbox_inches (str): bounding box in inches when creating the figure
        kwargs: the other arguments of matplotlib.pyplot.savefig()
    """

    @deprecate(old="TrendPlot", new="Dynamics.detect()", version="2.24.0-xi")
    def __init__(self, filename=None, bbox_inches="tight", **kwargs):
        self._filename = filename
        self._savefig_dict = {"bbox_inches": bbox_inches, **kwargs}
        # Properties
        self._title = ""
        self._variables = []
        _, self._ax = plt.subplots(1, 1)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        return super().__exit__(*exc_info)

    def plot(self, data, actual_col="Actual"):
        """
        Plot chronological change of the data with multiple lines.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    x values
                Columns
                    - column defined by @actual_col, actual values for y-axis
                    - the other arguments will be assumed as predicted values for y-axis
            actual_col (str): column name for y-axis
        """
        Validator(data, "data").dataframe(columns=[actual_col])
        predicted_cols = [col for col in data.columns if col != actual_col]
        self._variables = data.columns.tolist()
        # Scatter plot (actual values)
        self._ax.plot(
            data.index, data[actual_col],
            label=actual_col, color="black", marker=".", markeredgewidth=0, linewidth=0)
        # Plot lines
        for col in predicted_cols:
            self._ax.plot(data.index, data[col], label=col)
        plt.tight_layout()

    def x_axis(self, xlabel=None, xlim=(None, None)):
        """
        Set x axis.

        Args:
            xlabel (str or None): x-label
            xlim (tuple(int or float, int or float)): limit of x domain

        Note:
            When xlim[0] is None and lower x-axis limit determined by matplotlib automatically is lower than 0,
            lower x-axis limit will be set to 0.
        """
        # Label
        self._ax.set_xlabel(xlabel)
        # limit
        self._ax.set_xlim(max(self._ax.get_xlim()[0], xlim[0] or 0), xlim[1])
        # Integer scale
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        self._ax.xaxis.set_major_formatter(fmt)

    def y_axis(self, ylabel=None):
        """
        Set x axis.

        Args:
            ylabel (str or None): y-label
        """
        # Label
        self._ax.set_ylabel(ylabel)
        # Scale
        try:
            self._ax.set_yscale("log", base=10)
        except Exception:
            # Matplotlib version < 3.3
            self._ax.set_yscale("log", basey=10)
        # Log scale
        # Delete y-labels of log-scale (minor) axis
        plt.setp(self._ax.get_yticklabels(minor=True), visible=False)
        self._ax.tick_params(left=False, which="minor")
        # Set new y-labels of major axis
        ymin, ymax = self._ax.get_ylim()
        ydiff_scale = int(np.log10(ymax - ymin))
        yticks = np.linspace(
            round(ymin, - ydiff_scale),
            round(ymax, - ydiff_scale), 5, dtype=np.int64)
        self._ax.set_yticks(yticks)
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        self._ax.yaxis.set_major_formatter(fmt)

    def legend(self, bbox_to_anchor=(0.5, -0.5), bbox_loc="lower center", ncol=7, **kwargs):
        """
        Set legend.

        Args:
            bbox_to_anchor (tuple(int or float, int or float)): distance of legend and plot
            bbox_loc (str): location of legend
            ncol (int or None): the number of columns that the legend has
            kwargs: keyword arguments of matplotlib.pyplot.legend()
        """
        super().legend(bbox_to_anchor=bbox_to_anchor, bbox_loc=bbox_loc, ncol=ncol, **kwargs)


@deprecate(old="trend_plot()", new="Dynamics.detect()", version="2.24.0-xi")
def trend_plot(df, title=None, filename=None, show_legend=True, **kwargs):
    """
    Wrapper function: show chronological change of the data.

    Args:
        df (pandas.DataFrame): data to show
            Index
                x values
            Columns
                - column defined by @actual_col, actual values for y-axis
                - columns defined by @predicted_cols, predicted values for y-axis
        actual_col (str): column name for y-axis
        predicted_cols (list[str]): list of columns which have predicted values
        title (str): title of the figure
        filename (str or None): filename to save the figure or None (display)
        show_legend (bool): whether show legend or not
        kwargs: keyword arguments of the following classes and methods.
            - covsirphy.TrendPlot() and its methods,
            - matplotlib.pyplot.savefig() and matplotlib.pyplot.legend()
    """
    with TrendPlot(filename=filename, **find_args(plt.savefig, **kwargs)) as tp:
        tp.title = title
        tp.plot(data=df, **find_args([TrendPlot.plot], **kwargs))
        # Axis
        tp.x_axis(**find_args([TrendPlot.x_axis], **kwargs))
        tp.y_axis(**find_args([TrendPlot.y_axis], **kwargs))
        # Vertical/horizontal lines
        tp.line(**find_args([TrendPlot.line], **kwargs))
        # Legend
        if show_legend:
            tp.legend(**find_args([TrendPlot.legend, plt.legend], **kwargs))
        else:
            tp.legend_hide()


@deprecate(old="covsirphy.line_plot_multiple()", new="covsirphy.trend_plot()")
def line_plot_multiple(df, x_col, actual_col, predicted_cols, **kwargs):
    """
    Show multiple line graph of chronological change with actual plots.
    This function was deprecated. Please use covsirphy.trend_plot() function.

    Args:
        df (pandas.DataFrame): data
            Index
                Date (pandas.TimeStamp): Observation date
            Columns
                - column defined by @x_col, values for x-axis
                - column defined by @actual_col, actual values for y-axis
                - columns defined by @predicted_cols, predicted values for y-axis
        x_col (str): column name for x-axis
        actual_col (str): column name for y-axis
        kwargs: any other arguments of covsirphy.trend_plot()
    """
    return trend_plot(df.set_index(x_col), actual_col=actual_col, **kwargs)
