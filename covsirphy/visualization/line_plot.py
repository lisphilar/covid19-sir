#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from covsirphy.util.error import UnExecutedError
from covsirphy.util.argument import find_args
from covsirphy.visualization.vbase import VisualizeBase


class LinePlot(VisualizeBase):
    """
    Create line plot.

    Args:
        filename (str or None): filename to save the figure or None (display)
        kwargs: the other arguments of matplotlib.pyplot.savefig()
    """

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=filename, **kwargs)
        self._variables = []

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        # Tight layout
        plt.tight_layout()
        return super().__exit__(*exc_info)

    def plot(self, data, colormap=None, color_dict=None, **kwargs):
        """
        Plot chronological change of the data.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    Date (pandas.Timestamp)
                Columns
                    variables to show
            colormap (str, matplotlib colormap object or None): colormap, please refer to https://matplotlib.org/examples/color/colormaps_reference.html
            color_dict (dict[str, str] or None): dictionary of column names (keys) and colors (values)
            kwargs: keyword arguments of pandas.DataFrame.plot()
        """
        self._ensure_dataframe(data, name="data", time_index=True)
        self._variables = data.columns.tolist()
        # Color
        if color_dict is None:
            color_args = {"colormap": colormap}
        else:
            colors = [color_dict.get(col) for col in data.columns]
            color_args = {"colormap": colormap, "color": colors}
        # Set plotting
        try:
            self._ax = data.plot(**color_args, **kwargs)
        except ValueError as e:
            raise ValueError(e.args[0]) from None

    def x_axis(self, xlabel=None, x_logscale=False, xlim=(None, None)):
        """
        Set x axis.

        Args:
            xlabel (str or None): x-label
            x_logscale (bool): whether use log-scale in x-axis or not
            xlim (tuple(int or float, int or float)): limit of x dimain

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
        """
        Set x axis.

        Args:
            ylabel (str or None): y-label
            y_logscale (bool): whether use log-scale in y-axis or not
            ylim (tuple(int or float, int or float)): limit of y dimain
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
        # Interger scale
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

    def line(self, v=None, h=None, color="black", linestyle=":"):
        """
        Show vertical/horizontal lines.

        Args:
            v (list[int/float] or None): list of x values of vertical lines or None
            h (list[int/float] or None): list of y values of horizontal lines or None
            color (str): color of the line
            linestyle (str): linestyle
        """
        if h is not None:
            self._ax.axhline(y=h, color="black", linestyle=":")
        if v is not None:
            v = v if isinstance(v, list) else [v]
            for value in v:
                self._ax.axvline(x=value, color=color, linestyle=linestyle)

    def legend(self, bbox_to_anchor=(1.02, 0), bbox_loc="lower left", ncol=None, **kwargs):
        """
        Set legend.

        Args:
            bbox_to_anchor (tuple(int or float, int or float)): distance of legend and plot
            bbox_loc (str): location of legend
            ncol (int): the number of columns that the legend has
            kwargs: keyword arguments of matplotlib.pyplot.legend()
        """
        if not self._variables:
            raise UnExecutedError("LinePlot.plot()")
        ncol = self._ensure_natural_int(
            ncol or 1 if "left" in bbox_loc else len(self._variables), name="ncol")
        self._ax.legend(bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0, ncol=ncol, **kwargs)

    def legend_hide(self):
        """
        Hide legend.
        """
        self._ax.legend().set_visible(False)


def line_plot(df, title=None, filename=None, show_legend=True, **kwargs):
    """
    Wrapper function: show chronological change of the data.

    Args:
        data (pandas.DataFrame): data to show
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
