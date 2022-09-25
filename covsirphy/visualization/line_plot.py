#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from covsirphy.util.validator import Validator
from covsirphy.visualization.vbase import VisualizeBase, find_args


class LinePlot(VisualizeBase):
    """Create a line plot.

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

    def plot(self, data, colormap=None, color_dict=None, **kwargs):
        """Plot chronological change of the data.

        Args:
            data (pandas.DataFrame or pandas.Series): data to show
                Index
                    x values
                Columns
                    y variables to show
            colormap (str, matplotlib colormap object or None): colormap, please refer to https://matplotlib.org/examples/color/colormaps_reference.html
            color_dict (dict[str, str] or None): dictionary of column names (keys) and colors (values)
            kwargs: keyword arguments of pandas.DataFrame.plot()
        """
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        Validator(data, "data").dataframe()
        self._variables = data.columns.tolist()
        # Color
        color_args = self._plot_colors(data.columns, colormap=colormap, color_dict=color_dict)
        # Set plotting
        try:
            self._ax = data.plot(**color_args, **kwargs)
        except KeyError as e:
            raise KeyError(e.args[0]) from None

    def x_axis(self, xlabel=None, x_logscale=False, xlim=(None, None)):
        """Set x axis.

        Args:
            xlabel (str or None): x-label
            x_logscale (bool): whether use log-scale in x-axis or not
            xlim (tuple(int or float, int or float)): limit of x domain

        Note:
            If None is included in xlim, the values will be automatically determined by Matplotlib
        """
        # Label
        self._ax.set_xlabel(xlabel)
        # Log scale
        if x_logscale:
            self._ax.set_xscale("log")
            xlim = (None, None) if xlim[0] == 0 else xlim
        # limit
        self._ax.set_xlim(*xlim)

    def y_axis(self, ylabel="Cases", y_logscale=False, ylim=(0, None), math_scale=True, y_integer=False):
        """Set x axis.

        Args:
            ylabel (str or None): y-label
            y_logscale (bool): whether use log-scale in y-axis or not
            ylim (tuple(int or float, int or float)): limit of y domain
            math_scale (bool): whether use LaTEX or not in y-label
            y_integer (bool): whether force to show the values as integer or not

        Note:
            If None is included in ylim, the values will be automatically determined by Matplotlib
        """
        # Label
        self._ax.set_ylabel(ylabel)
        # Math scale
        if math_scale:
            self._ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            self._ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # Integer scale
        if y_integer:
            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)
            self._ax.yaxis.set_major_formatter(fmt)
        # Log scale
        if y_logscale:
            self._ax.set_yscale("log")
            ylim = (None, None) if ylim[0] == 0 else ylim
        # limit
        self._ax.set_ylim(*ylim)

    @staticmethod
    def _convert_to_list(x):
        """Convert None to empty list, str/float/int etc. to a list.

        Args:
            x (list/tuple[str/int/float] or None): value(s)

        Returns:
            list or tuple
        """
        return x if isinstance(x, (list, tuple)) else [] if x is None else [x]

    def line(self, v=None, h=None, color="black", linestyle=":"):
        """Show vertical/horizontal lines.

        Args:
            v (list/tuple[int/float] or None): list of x values of vertical lines or None
            h (list/tuple[int/float] or None): list of y values of horizontal lines or None
            color (str): color of the line
            linestyle (str): linestyle
        """
        # Horizontal
        h = self._convert_to_list(h)
        for value in h:
            self._ax.axhline(y=value, color=color, linestyle=linestyle)
        # Vertical
        v = self._convert_to_list(v)
        for value in v:
            self._ax.axvline(x=value, color=color, linestyle=linestyle)


def line_plot(df, title=None, filename=None, show_legend=True, **kwargs):
    """Wrapper function: show chronological change of the data.

    Args:
        data (pandas.DataFrame or pandas.Series): data to show
            Index
                Date (pandas.Timestamp)
            Columns
                variables to show
        title (str): title of the figure
        filename (str or None): filename to save the figure or None (display)
        show_legend (bool): whether show legend or not
        kwargs: keyword arguments of the following classes and methods.
            - covsirphy.LinePlot() and its methods,
            - matplotlib.pyplot.savefig(), matplotlib.pyplot.legend(),
            - pandas.DataFrame.plot()
    """
    with LinePlot(filename=filename, **find_args(plt.savefig, **kwargs)) as lp:
        lp.title = title
        lp.plot(data=df, **find_args([LinePlot.plot, pd.DataFrame.plot], **kwargs))
        # Axis
        lp.x_axis(**find_args([LinePlot.x_axis], **kwargs))
        lp.y_axis(**find_args([LinePlot.y_axis], **kwargs))
        # Vertical/horizontal lines
        lp.line(**find_args([LinePlot.line], **kwargs))
        # Legend
        if show_legend:
            lp.legend(**find_args([LinePlot.legend, plt.legend], **kwargs))
        else:
            lp.legend_hide()
