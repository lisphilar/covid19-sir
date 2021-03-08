#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("seaborn-ticks")
plt.style.use("fast")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["legend.frameon"] = False


def box_plot(df, title, xlabel=None, ylabel=None,
             v=None, h=None,
             show_legend=True, bbox_to_anchor=(1.02, 0), bbox_loc="lower left",
             filename=None):
    """
    Show box plot of the data.

    Args:
        df (pandas.DataFrame): target data

            Index
                reset index
            Columns
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


def line_plot_multiple(df, x_col, actual_col, predicted_cols,
                       title, ylabel, xlim=(None, None), v=None, y_logscale=False, filename=None):
    """
    show multiple line graph of chronological change with actual plots.

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
        predicted_cols (list[str]): list of columns which have predicted values
        title (str): title of the figure
        y_label (str): label for y-axis
        xlim (tuple(int or float, int or float)): limit of x dimain
        y_logscale (bool): whether use log-scale in y-axis or not
        v (list[int]): list of Recovered values to show vertical lines
        filename (str): filename of the figure, or None (show figure)

    Note:
        When xlim[0] is None and lower x-axis limit determined by matplotlib automatically is lower than 0,
        lower x-axis limit will be set to 0.
    """
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", UserWarning)
    x_series = df[x_col]
    actual = df[actual_col]
    # Plot the actual values
    plt.plot(
        x_series, actual,
        label=actual_col, color="black", marker=".", markeredgewidth=0, linewidth=0)
    # Plot the predicted values
    for col in predicted_cols:
        plt.plot(x_series, df[col], label=col)
    # x-axis
    plt.xlabel(x_col)
    plt.xlim(max(plt.xlim()[0], xlim[0] or 0), xlim[1])
    # y-axis
    plt.ylabel(ylabel)
    try:
        plt.yscale("log", base=10)
    except Exception:
        # Matplotlib version < 3.3
        plt.yscale("log", basey=10)
    # Delete y-labels of log-scale (minor) axis
    plt.setp(plt.gca().get_yticklabels(minor=True), visible=False)
    plt.gca().tick_params(left=False, which="minor")
    # Set new y-labels of major axis
    if y_logscale:
        ymin, ymax = plt.ylim()
        ydiff_scale = int(np.log10(ymax - ymin))
        yticks = np.linspace(
            round(ymin, - ydiff_scale),
            round(ymax, - ydiff_scale), 5, dtype=np.int64)
        plt.gca().set_yticks(yticks)
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        plt.gca().yaxis.set_major_formatter(fmt)
    # Title
    plt.title(title)
    # Vertical lines
    if isinstance(v, (list, tuple)):
        for value in v:
            plt.axvline(x=value, color="black", linestyle=":")
    # Legend
    plt.legend(
        bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0
    )
    # Save figure or show figure
    plt.tight_layout()
    if filename is None:
        plt.show()
        return None
    plt.savefig(
        filename, bbox_inches="tight", transparent=False, dpi=300
    )
    plt.clf()
