#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests
from covsirphy.util.error import UnExpectedValueError
from covsirphy.visualization.vbase import VisualizeBase


class ColoredMap(VisualizeBase):
    """
    Create global map with pandas.DataFrame.

    Args:
        filename (str or None): filename to save the figure or None (display)
        kwargs: the other arguments of matplotlib.pyplot.savefig()
    """

    def __init__(self, filename, **kwargs):
        super().__init__(filename=filename, **kwargs)

    def plot(self, series, index_name="ISO3", directory="input", **kwargs):
        """
        Set dataframe and the variable to show in a colored map.

        Args:
            series (pandas.Series): data to show
                Index
                    ISO3 codes, country names or province names
                Values
                    - (int or float): values to color the map
            index_name (str): index name, 'ISO3', 'Country' or 'Province'
            directory (str): directory to save the downloaded files of geometry information
            kwargs: arguments of geopandas.GeoDataFrame.plot() except for 'column'

        Raises:
            UnExpectedValueError: incorrect value was applied as @index_name
        """
        key_dict = {
            self.ISO3: "iso_a3", self.COUNTRY: "name", self.PROVINCE: "name"}
        if index_name not in key_dict:
            raise UnExpectedValueError(
                name="index_name", value=index_name, candidates=list(key_dict.keys()))
        self._ensure_instance(series, pd.Series, name="series")
        df = series.reset_index().dropna()
        df.columns = [index_name, "Value"]
        df.dropna(inplace=True)
        # Values of ISO3 column should be unique
        if not df[index_name].is_unique:
            raise ValueError(
                f"{self.ISO3} column of the dataframe should be unique.")
        # Geometry information from Natural Earth
        if index_name in (self.ISO3, self.COUNTRY):
            # pop_est, continent, name, iso_a3, gdp_md_est, geometry
            geopath = gpd.datasets.get_path("naturalearth_loweres")
        else:
            geopath = self._load_geo_provinces(directory=directory)
        geo_df = gpd.read_file(geopath)
        # Merge the data with geometry information
        df = geo_df.merge(
            df, how="inner", left_on=key_dict[index_name], right_on=index_name)
        # Plotting
        df.plot(column="Value", **kwargs)
        # Remove all ticks
        self.tick_params(
            labelbottom=False, labelleft=False, left=False, bottom=False)

    @staticmethod
    def _load_geo_provinces(directory):
        """
        Load the shape file (1:10 million scale) from 'Natural Earth Vector'.
        https://github.com/nvkelso/natural-earth-vector

        Args:
            directory (str): directory to save the downloaded files of geometry information

        Returns:
            str: filename of the shape file
        """
        title = "ne_10m_admin_1_states_provinces"
        extensions = [
            ".README.html", ".VERSION.txt", ".cpg", ".dbf", ".sbn", ".sbx", ".shp", ".shx"]
        geo_dirpath = Path(directory).joinpath(title)
        geo_dirpath.mkdir(parents=True, exist_ok=True)
        shapefile = geo_dirpath.joinpath(f"{title}.shp")
        # Download files, if necessary
        for ext in extensions:
            basename = f"{title}{ext}"
            if geo_dirpath.joinpath(basename).exists():
                continue
            url = f"https://github.com/nvkelso/natural-earth-vector/blob/master/10m_cultural/{basename}?raw=true"
            response = requests.get(url=url)
            with geo_dirpath.joinpath(basename).open("wb") as fh:
                fh.write(response.content)
        return shapefile
