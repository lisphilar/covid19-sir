#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# plt.style.use("seaborn-ticks")
plt.style.use("fast")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["legend.frameon"] = False


def line_plot(df, title, xlabel=None, ylabel="Cases",
              v=None, h=None,
              xlim=(None, None), ylim=(0, None),
              math_scale=True, x_logscale=False, y_logscale=False, y_integer=False,
              show_legend=True, bbox_to_anchor=(1.02, 0), bbox_loc="lower left",
              filename=None):
    """
    Show chronological change of the data.

    Args:
        df (pandas.DataFrame): target data

            Index:
                reset index
            Columns:
                field names
            Values:
                data values
        title (str): title of the figure
        xlabel (str): x-label
        ylabel (str): y-label
        v (list[int/float]): list of x values of vertical lines or None
        h (list[int/float]): list of y values of horizontal lines or None
        xlim (tuple(int or float, int or float)): limit of x dimain
        ylim (tuple(int or float, int or float)): limit of y dimain
        math_scale (bool): whether use LaTEX or not
        x_logscale (bool): whether use log-scale in x-axis or not
        y_logscale (bool): whether use log-scale in y-axis or not
        y_integer (bool): whether force to show the values as integer or not
        show_legend (bool): whether show legend or not
        bbox_to_anchor (tuple(int or float, int or float)): distance of legend and plot
        bbox_loc (str): location of legend
        filename (str): filename of the figure, or None (show figure)

    Notes:
        If None is included in xlim/ylim, the values will be automatically determined by Matplotlib
    """
    ax = df.plot()
    # Scale
    if math_scale:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter(useMathText=True)
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    if x_logscale:
        ax.set_xscale("log")
        if xlim[0] == 0:
            xlim = (None, None)
    if y_logscale:
        ax.set_yscale("log")
        if ylim[0] == 0:
            ylim = (None, None)
    if y_integer:
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    # Set metadata of figure
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0)
    else:
        ax.legend().set_visible(False)
    if h is not None:
        ax.axhline(y=h, color="black", linestyle=":")
    if v is not None:
        if not isinstance(v, list):
            v = [v]
        for value in v:
            ax.axvline(x=value, color="black", linestyle=":")
    plt.tight_layout()
    # Save figure or show figure
    if filename is None:
        plt.show()
        return None
    plt.savefig(
        filename, bbox_inches="tight", transparent=False, dpi=300
    )
    plt.clf()
    return None


def box_plot(df, title, xlabel=None, ylabel=None,
             v=None, h=None,
             show_legend=True, bbox_to_anchor=(1.02, 0), bbox_loc="lower left",
             filename=None):
    """
    Show box plot of the data.

    Args:
        df (pandas.DataFrame): target data

            Index:
                reset index
            Columns:
                field names
            Values:
                data values
        title (str): title of the figure
        xlabel (str): x-label
        ylabel (str): y-label
        v (list[int/float]): list of x values of vertical lines or None
        h (list[int/float]): list of y values of horizontal lines or None
        show_legend (bool): whether show legend or not
        bbox_to_anchor (tuple(int or float, int or float)): distance of legend and plot
        bbox_loc (str): location of legend
        filename (str): filename of the figure, or None (show figure)
    """
    df.plot.bar(title=title)
    plt.xticks(rotation=0)
    if h is not None:
        plt.axhline(y=h, color="black", linestyle=":")
    plt.legend(
        bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0
    )
    plt.tight_layout()
    if filename is None:
        plt.show()
        return None
    plt.savefig(
        filename, bbox_inches="tight", transparent=False, dpi=300
    )
    plt.clf()
    return None
