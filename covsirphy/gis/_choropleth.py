#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import geopandas as gpd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from covsirphy.util.validator import Validator
from covsirphy.visualization.vbase import VisualizeBase


class _ChoroplethMap(VisualizeBase):
    """Create choropleth map.

    Args:
        filename (str or None): filename to save the figure or None (display)
        kwargs: the other arguments of matplotlib.pyplot.savefig()
    """

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=filename, **kwargs)
        self._geo_dirpath = Path("input")

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        return super().__exit__(*exc_info)

    def plot(self, data, logscale, **kwargs):
        """Set geopandas.GeoDataFrame to create a choropleth map.

        Args:
            data (geopandas.GeoDataFrame):
                Index:
                    - reset index
                Columns:
                    - Location (str): location names
                    - Variable (str): variable to show
                    - geometry: geometric information
            logscale (bool): whether convert the value to log10 scale values or not
            kwargs: arguments of geopandas.GeoDataFrame.plot() except for 'column'
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        df = Validator(data, "data").dataframe(columns=["Location", "Variable", "geometry"])
        df["Variable"] = df["Variable"].astype("float64")
        gdf = gpd.GeoDataFrame(df)
        # Color bar
        divider = make_axes_locatable(self._ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        # Arguments of plotting with GeoPandas
        plot_kwargs = {
            "legend": True,
            "cmap": "coolwarm",
            "ax": self._ax,
            "cax": cax,
            "missing_kwds": {
                "color": "lightgrey",
                "edgecolor": "white",
                "hatch": "///",
            }
        }
        plot_kwargs.update(kwargs)
        plot_kwargs["legend_kwds"] = {"orientation": "horizontal"}
        # Convert to log10 scale
        if logscale:
            gdf["Variable"] = np.log10(gdf["Variable"] + 1)
            plot_kwargs["legend_kwds"]["label"] = "in log10 scale"
        # Plotting
        if not gdf["Variable"].isna().sum():
            # Missing values are not included
            plot_kwargs.pop("missing_kwds")
        gdf.plot(column="Variable", **plot_kwargs)
        # Remove all ticks
        self._ax.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
