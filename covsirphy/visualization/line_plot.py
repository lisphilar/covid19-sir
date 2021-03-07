#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        return super().__exit__(*exc_info)

    def plot(self, data, colormap=None, color_dict=None, **kwargs):
        """
        Plot chronological change of the data.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    Date (pandas.TimeStamp)
                Columns
                    variables to show
            colormap (str, matplotlib colormap object or None): colormap, please refer to https://matplotlib.org/examples/color/colormaps_reference.html
            color_dict (dict[str, str] or None): dictionary of column names (keys) and colors (values)
            kwargs: keyword arguments of pandas.DataFrame.plot()
        """
        self._ensure_dataframe(data, name="data", time_index=True)
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
