#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
from covsirphy.util.error import UnExecutedError
from covsirphy.util.validator import Validator
from covsirphy.visualization.vbase import find_args
from covsirphy.visualization.line_plot import LinePlot


class ScatterPlot(LinePlot):
    """Create a scatter plot.

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
        self._ax = None
        self._data = pd.DataFrame(columns=["x", "y"])

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        return super().__exit__(*exc_info)

    def plot(self, data, colormap=None, color_dict=None, **kwargs):
        """Plot chronological change of the data.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    reset index
                Columns
                    x (int or float): x values
                    y (int or float): y values
            colormap (str, matplotlib colormap object or None): colormap, please refer to https://matplotlib.org/examples/color/colormaps_reference.html
            color_dict (dict[str, str] or None): dictionary of column names (keys) and colors (values)
            kwargs: keyword arguments of pandas.DataFrame.plot()
        """
        self._data = Validator(data, "data").dataframe(columns=["x", "y"])
        # Color
        color_args = self._plot_colors(data.columns, colormap=colormap, color_dict=color_dict)
        # Set plotting
        try:
            self._ax = data.plot.scatter(x="x", y="y", **color_args, **kwargs)
        except KeyError as e:
            raise KeyError(e.args[0]) from None

    def line_straight(self, p1=None, p2=None, color="black", linestyle=":"):
        """Connect the points with a straight line.

        Args:
            p1 (tuple(int or float, int or float) or None): (x, y) of the first point or None (min values)
            p2 (tuple(int or float, int or float) or None): (x, y) of the second point or None (max values)
            color (str): color of the line
            linestyle (str): linestyle

        Note:
            The same line will be show when p1 and p2 is reordered.
        """
        if self._data.empty:
            raise UnExecutedError("ScatterPlot.plot()")
        x1, y1 = (self._data["x"].min(), self._data["y"].min()) if p1 is None else p1
        x2, y2 = (self._data["x"].max(), self._data["y"].max()) if p2 is None else p2
        self._ax.plot([x1, x2], [y1, y2], color=color, linestyle=linestyle)

    def legend(self, **kwargs):
        """ScatterPlot.legend() is not implemented.
        """
        raise NotImplementedError

    def legend_hide(self):
        """ScatterPlot.legend_hide() is not implemented.
        """
        raise NotImplementedError


def scatter_plot(df, title=None, filename=None, **kwargs):
    """Wrapper function: show chronological change of the data.

    Args:
        data (pandas.DataFrame): data to show
            Index
                reset index
            Columns
                x (int or float): x values
                y (int or float): y values
        title (str): title of the figure
        filename (str or None): filename to save the figure or None (display)
        kwargs: keyword arguments of the following classes and methods.
            - covsirphy.ScatterPlot() and its methods,
            - matplotlib.pyplot.savefig(), matplotlib.pyplot.legend(),
            - pandas.DataFrame.plot()
    """
    with ScatterPlot(filename=filename, **find_args(plt.savefig, **kwargs)) as sp:
        sp.title = title
        sp.plot(data=df, **find_args([ScatterPlot.plot, pd.DataFrame.plot], **kwargs))
        # Axis
        sp.x_axis(**find_args([ScatterPlot.x_axis], **kwargs))
        sp.y_axis(**find_args([ScatterPlot.y_axis], **kwargs))
        # Vertical/horizontal lines
        sp.line(**find_args([ScatterPlot.line], **kwargs))
        # Straight lines
        sp.line_straight(**find_args([ScatterPlot.line_straight], **kwargs))
