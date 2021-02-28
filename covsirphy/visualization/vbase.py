#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
from matplotlib import pyplot as plt
from covsirphy.util.term import Term

# Style of Matplotlib
plt.style.use("fast")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["legend.frameon"] = False


class VisualizeBase(Term):
    """
    Base class for visualization.

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
        # Display the figure if filename is None after plotting
        try:
            if self._filename is None:
                plt.show()
            else:
                # Save the image as a file
                plt.savefig(self._filename, **self._savefig_dict)
                plt.clf()
                plt.close("all")
        except AttributeError:
            pass

    @property
    def title(self):
        """
        str: title of the figure
        """
        return self._title

    @title.setter
    def title(self, title):
        self._title = str(title)

    @property
    def ax(self):
        """
        matplotlib.axis: axis
        """
        return self._ax

    @ax.setter
    def ax(self, ax):
        self._ax = self._ensure_instance(ax, matplotlib.axes.Axes, name="ax")

    def plot(self):
        """
        Method for plotting. This will be defined in child classes.

        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError

    def tick_params(self, **kwargs):
        """
        Directly calling matplotlib.pyplot.tick_params,
        change the appearance of ticks, tick labels and gridlines.

        Args:
            kwargs: arguments of matplotlib.pyplot.tick_params
        """
        self._ax.tick_params(**kwargs)
