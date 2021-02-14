#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from pathlib import Path
import warnings
import country_converter as coco
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import requests
from unidecode import unidecode
from covsirphy.visualization.vbase import VisualizeBase


class ColoredMap(VisualizeBase):
    """
    Create global map with pandas.DataFrame.

    Args:
        filename (str or None): filename to save the figure or None (display)
        kwargs: the other arguments of matplotlib.pyplot.savefig()
    """

    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=filename, **kwargs)
        self._to_iso3 = partial(coco.convert, to="ISO3", not_found=None)
        self._geo_dirpath = Path("input")

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        return super().__exit__(*exc_info)

    @property
    def directory(self):
        """
        str: directory to save the downloaded files of geometry information
        """
        return str(self._geo_dirpath)

    @directory.setter
    def directory(self, name):
        self._geo_dirpath = Path(name)

    def plot(self, data, level="Country", included=None, excluded=None, logscale=True, **kwargs):
        """
        Set dataframe and the variable to show in a colored map.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    reset index
                Columns
                    - Country (str or pandas.Category): country name(s)
                    - Province (str or pandas.Category): province names, necessary when @level is 'Province'
                    - Value (int or float or None): values to coloring the map
                    - ISO3 (str): ISO3 codes, optional
            level (str): 'Country' (global map) or 'Province' (country-specific map)
            logscale (bool): whether convert the value to log10 scale values or not
            included (list[str] or None): included countries/provinces or None (all)
            excluded (list[str] or None): excluded countries/provinces or None (all)
            kwargs: arguments of geopandas.GeoDataFrame.plot() except for 'column'

        Raises:
            ValueError: labels for data are not unique
            UnExpectedValueError: some countries' records are included when @level is 'Province'
            SubsetNotFoundError: no geometry information available for the labels
        """
        expected_cols = [self.COUNTRY, "Value"] + [] if level == self.COUNTRY else [self.PROVINCE]
        self._ensure_dataframe(data, name="data", columns=expected_cols)
        if level == self.COUNTRY:
            # Global map with country level data
            if not data[self.COUNTRY].is_unique:
                raise ValueError(f"{self.COUNTRY} column of data should be unique.")
            gdf = self._global_data(data=data, included=included, excluded=excluded)
        else:
            # Country-specific map with province level data
            unique_n = data[self.COUNTRY].nunique()
            if unique_n > 1:
                raise ValueError(
                    f"{self.COUNTRY} column of data should have only one country name when @level is {self.PROVINCE}, but {unique_n} found.")
            country = data[self.COUNTRY].value_counts().index[0]
            if not data[self.PROVINCE].is_unique:
                raise ValueError(f"{self.PROVINCE} column of data should be unique.")
            gdf = self._country_specific_data(data, included=included, excluded=excluded, country=country)
        gdf.loc[gdf["Value"] < 0, "Value"] = 0
        # Colorbar
        divider = make_axes_locatable(self._ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
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
        # Convert to log10 scale
        if logscale:
            gdf["Value"] = np.log10(gdf["Value"] + 1)
            plot_kwargs["legend_kwds"] = {"label": "in log10 scale"}
        # Plotting
        warnings.filterwarnings("ignore", category=UserWarning)
        if not gdf["Value"].isna().sum():
            # Missing values are not included
            plot_kwargs.pop("missing_kwds")
        gdf.plot(column="Value", **plot_kwargs)
        # Remove all ticks
        self._ax.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)

    def _global_data(self, data, included, excluded):
        """
        Create global map data with geometry information.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    reset index
                Columns
                    - Country (str): country names
                    - Value (int or float or None): values to plot
                    - ISO3 (str): ISO3 codes, optional
            included (list[str] or None): included countries or None (all)
            excluded (list[str] or None): excluded countries or None (all)

        Returns:
            geopandas.GeoDataFrame:
                Index
                    reset index
                Columns
                    - Value (int or float or None)
                    - geometry (geopandas.GeoDataFrame.geometry): geometry information
        """
        # data to plot
        df = data.copy()
        df[self.ISO3] = df[self.COUNTRY].apply(self._to_iso3) if self.ISO3 not in df else df[self.ISO3]
        # Geometry
        gdf = self._load_geo_global()
        # Merge them
        gdf = gdf.merge(df, how="left", on=self.ISO3)
        # Select countries
        included_codes = gdf[self.ISO3].tolist() if included is None else [self._to_iso3(c) for c in included]
        excluded_codes = [] if excluded is None else [self._to_iso3(c) for c in excluded]
        sel = set(included_codes) - set(excluded_codes)
        return gdf.loc[gdf[self.ISO3].isin(sel), ["Value", "geometry"]]

    def _country_specific_data(self, data, included, excluded, country):
        """
        Create country-specific map data with geometry information.

        Args:
            data (pandas.DataFrame): data to show
                Index
                    reset index
                Columns
                    - Province (str): province names
                    - Value (int or float or None): values to plot
            included (list[str] or None): included countries or None (all)
            excluded (list[str] or None): excluded countries or None (all)
            country (str): country name
            scale (str): scale of geographic shapes, '10m', '50m' or '110m'

        Returns:
            geopandas.GeoDataFrame:
                Index
                    reset index
                Columns
                    - Value (int or float or None)
                    - geometry (geopandas.GeoDataFrame.geometry): geometry information
        """
        # Get geometry information of the country
        iso3 = self._to_iso3(country)
        scale = "50m" if iso3 == "USA" else "10m"
        gdf = self._load_geo_country_specific(scale=scale)
        gdf = gdf.loc[gdf[self.ISO3] == iso3]
        # Merge the data with geometry information
        gdf = gdf.merge(data, how="left", on=self.PROVINCE)
        # Select countries
        sel = set(included or gdf[self.PROVINCE].unique()) - set(excluded or [])
        return gdf.loc[gdf[self.PROVINCE].isin(sel), ["Value", "geometry"]]

    def _load_geo_global(self):
        """
        Retrieve geometry information for global map.

        Returns:
            geopandas.GeoDataFrame:
                Index
                    reset index
                Columns
                    - ISO3 (str): ISO3 codes
                    - geometry (geopandas.GeoDataFrame.geometry): geometry information
        """
        # Geometry information from Natural Earth
        # pop_est, continent, name, iso_a3, gdp_md_est, geometry
        geopath = gpd.datasets.get_path("naturalearth_lowres")
        gdf = gpd.read_file(geopath)
        # Data cleaning
        gdf["name"] = gdf["name"].apply(unidecode)
        gdf["name"].replace(
            {
                "Fr. S. Antarctic Lands": "French Southern and Antarctic Lands",
                "S. Sudan": "South Sudan"
            }, inplace=True)
        # Get ISO3 codes
        gdf.rename(columns={"iso_a3": self.ISO3}, inplace=True)
        sel = gdf[self.ISO3] == "-99"
        gdf.loc[sel, self.ISO3] = gdf.loc[sel, "name"].apply(self._to_iso3)
        # Remove Antarctica and
        gdf = gdf.loc[gdf[self.ISO3] != "ATA"]
        return gdf.loc[:, [self.ISO3, "geometry"]]

    def _load_geo_country_specific(self, scale):
        """
        Load shape file from 'Natural Earth Vector'.
        https://github.com/nvkelso/natural-earth-vector

        Args:
            scale (str): scale of geographic shapes, '10m', '50m' or '110m'

        Returns:
            geopandas.GeoDataFrame:
                Index
                    reset index
                Columns
                    - ISO3 (str): ISO3 codes
                    - Province (str): province name
                    - geometry (geopandas.GeoDataFrame.geometry): geometry information
        """
        title = f"ne_{scale}_admin_1_states_provinces"
        extensions = [".README.html", ".VERSION.txt", ".cpg", ".dbf", ".sbn", ".sbx", ".shp", ".shx"]
        geo_dirpath = self._geo_dirpath.joinpath(title)
        geo_dirpath.mkdir(parents=True, exist_ok=True)
        # Download files, if necessary
        for ext in extensions:
            basename = f"{title}{ext}"
            if geo_dirpath.joinpath(basename).exists():
                continue
            url = f"https://github.com/nvkelso/natural-earth-vector/blob/master/{scale}_cultural/{basename}?raw=true"
            response = requests.get(url=url)
            with geo_dirpath.joinpath(basename).open("wb") as fh:
                fh.write(response.content)
        # Data cleaning
        gdf = gpd.read_file(geo_dirpath.joinpath(f"{title}.shp"))
        gdf["name"] = gdf["name"].fillna("").apply(unidecode)
        gdf.rename(columns={"name": self.PROVINCE, "adm0_a3": self.ISO3}, inplace=True)
        return gdf.loc[:, [self.ISO3, self.PROVINCE, "geometry"]]
