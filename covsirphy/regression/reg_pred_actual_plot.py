#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from covsirphy.visualization.vbase import VisualizeBase


class _PredActualPlot(VisualizeBase):
    """
    Class for a scatter plot (predicted vs. actual parameter values).

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
        self._ax = None
        self._facet_grid = None

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        # Settings
        if self._title:
            self._facet_grid.fig.subplots_adjust(top=0.9)
            self._facet_grid.fig.suptitle(self._title)
        # Display the figure if filename is None after plotting
        if self._filename is None:
            plt.show()
        else:
            # Save the image as a file
            self._facet_grid.savefig(self._filename, **self._savefig_dict)
            plt.clf()
            plt.close("all")

    def plot(self, data, x, y):
        """
        Create a scatter plot (predicted vs. actual parameter values).

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - {x} (float): actual values
                    - {y} (float): predicted values
                    - parameter (float): parameter names
                    - subset (str): "Train" or "Test"
            x (str): x label
            y (str): y label
        """
        facet_grid = sns.relplot(
            data=data, x=x, y=y, col="parameter", hue="subset", style="subset", kind="scatter",
            col_wrap=2, alpha=0.7, facet_kws={"sharex": False, "sharey": False, "legend_out": False})
        # Add y = x lines
        for ax in facet_grid.axes.flat:
            ax.axline((0, 0), slope=1, ls=":")
        # Set limits
        facet_grid.set(xlim=(0, None), ylim=(0, None))
        # Set Legend
        handles, legends = facet_grid.axes[0].get_legend_handles_labels()
        facet_grid.axes[0].legend_.remove()
        facet_grid.fig.legend(
            handles, legends, title="", ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
        # Save grid
        self._facet_grid = facet_grid
