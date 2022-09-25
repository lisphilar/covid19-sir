#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import signature
import sys
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
from matplotlib import pyplot as plt
from covsirphy.util.error import UnExecutedError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term

# Style of Matplotlib
plt.style.use("fast")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["legend.frameon"] = False


def find_args(func_list, **kwargs):
    """Find values of enabled arguments of the function from the keyword arguments.

    Args:
        func_list (list[function] or function): target function
        kwargs: keyword arguments

    Returns:
        dict: dictionary of enabled arguments
    """
    if not isinstance(func_list, list):
        func_list = [func_list]
    enabled_nest = [
        list(signature(func).parameters.keys()) for func in func_list
    ]
    enabled_set = set(sum(enabled_nest, list()))
    enabled_set = enabled_set - {"self", "cls"}
    return {k: v for (k, v) in kwargs.items() if k in enabled_set}


class VisualizeBase(Term):
    """Base class for visualization.

    Args:
        filename (str or None): filename to save the figure or None (display)
        bbox_inches (str): bounding box in inches when creating the figure
        kwargs: the other arguments of matplotlib.pyplot.savefig
    """

    def __init__(self, filename=None, bbox_inches="tight", **kwargs):
        self._filename = filename
        self._savefig_dict = {"bbox_inches": bbox_inches, **kwargs}
        # Properties
        self._title = ""
        _, self._ax = plt.subplots(1, 1)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        # Settings
        if self._title:
            self._ax.title.set_text(self._title)
        # Tight layout
        plt.tight_layout()
        # Display the figure if filename is None after plotting
        if self._filename is None:
            plt.show()
        else:
            # Save the image as a file
            plt.savefig(self._filename, **self._savefig_dict)
            plt.clf()
            plt.close("all")

    @property
    def title(self):
        """str: title of the figure
        """
        return self._title

    @title.setter
    def title(self, title):
        self._title = str(title)

    @property
    def ax(self):
        """matplotlib.axis: axis
        """
        return self._ax

    @ax.setter
    def ax(self, ax):
        self._ax = Validator(ax, "ax").instance(matplotlib.axes.Axes)

    def plot(self):
        """Method for plotting. This will be defined in child classes.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError

    def tick_params(self, **kwargs):
        """Directly calling matplotlib.pyplot.tick_params, change the appearance of ticks, tick labels and grid lines.

        Args:
            kwargs: arguments of matplotlib.pyplot.tick_params
        """
        self._ax.tick_params(**kwargs)

    def legend(self, bbox_to_anchor=(0.5, -0.2), bbox_loc="lower center", ncol=None, **kwargs):
        """Set legend.

        Args:
            bbox_to_anchor (tuple(int or float, int or float)): distance of legend and plot
            bbox_loc (str): location of legend
            ncol (int or None): the number of columns that the legend has
            kwargs: keyword arguments of matplotlib.pyplot.legend()
        """
        if not self._variables:
            raise UnExecutedError(".plot()")
        ncol = Validator(
            ncol or (1 if "left" in bbox_loc else len(self._variables)), "ncol").int(value_range=(1, None))
        self._ax.legend(bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0, ncol=ncol, **kwargs)
        plt.tight_layout()

    def legend_hide(self):
        """Hide legend.
        """
        self._ax.legend().set_visible(False)

    @staticmethod
    def _plot_colors(variables, colormap=None, color_dict=None):
        """Create an argument dictionary of colors for Matplotlib.

        Args:
            variables (list[str] or pandas.Index): list of variables to show
            colormap (str, matplotlib colormap object or None): colormap, please refer to https://matplotlib.org/examples/color/colormaps_reference.html
            color_dict (dict[str, str] or None): dictionary of column names (keys) and colors (values)
        """
        # Color
        if color_dict is None:
            return {"colormap": colormap}
        colors = [color_dict.get(col) for col in variables if col in color_dict]
        return {"colormap": colormap, "color": colors}
