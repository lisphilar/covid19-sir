#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import country_converter as coco
import geopandas as gpd
import pandas as pd
import requests
from unidecode import unidecode
from covsirphy.util.error import SubsetNotFoundError, UnExpectedValueError
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

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        return super().__exit__(*exc_info)

    def plot(self, series, index_name="ISO3", directory="input", usa=False, **kwargs):
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
            usa (bool): if True, not show islands of USA when @index_name is 'Province'
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
        df = pd.DataFrame(series).reset_index().dropna()
        df.columns = [index_name, "Value"]
        # Values of ISO3 column should be unique
        if not df[index_name].is_unique:
            raise ValueError(
                f"'{index_name}' index of the dataframe should be unique.")
        # Geometry information from Natural Earth
        if index_name in (self.ISO3, self.COUNTRY):
            # pop_est, continent, name, iso_a3, gdp_md_est, geometry
            geopath = gpd.datasets.get_path("naturalearth_lowres")
        else:
            scale = "50m" if usa else "10m"
            geopath = self._load_geo_provinces(
                directory=directory, scale=scale)
        gdf = gpd.read_file(geopath)
        # Merge the data with geometry information
        df[index_name] = df[index_name].apply(unidecode)
        gdf["name"] = gdf["name"].fillna("").apply(unidecode)
        gdf["name"].replace(
            {
                "Fr. S. Antarctic Lands": "French Southern and Antarctic Lands",
                "S. Sudan": "South Sudan"
            }, inplace=True)
        if index_name in (self.ISO3, self.COUNTRY):
            gdf["iso_a3"] = gdf[["name", "iso_a3"]].apply(
                lambda x: coco.convert(x[0], to="ISO3", not_found=x[1]), axis=1)
        gdf = gdf.merge(
            df, how="inner", left_on=key_dict[index_name], right_on=index_name)
        if gdf.empty:
            raise SubsetNotFoundError(
                country="the selected country", message="(geometry data)")
        # Plotting
        gdf.plot(column="Value", **kwargs)
        # Remove all ticks
        self._ax.tick_params(
            labelbottom=False, labelleft=False, left=False, bottom=False)

    @ staticmethod
    def _load_geo_provinces(directory, scale="10m"):
        """
        Load the shape file (1:10 million scale) from 'Natural Earth Vector'.
        https://github.com/nvkelso/natural-earth-vector

        Args:
            directory (str): directory to save the downloaded files of geometry information
            scale (str): scale of geographic shapes, '10m', '50m' or '110m'

        Returns:
            str: filename of the shape file
        """
        title = f"ne_{scale}_admin_1_states_provinces"
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
            url = f"https://github.com/nvkelso/natural-earth-vector/blob/master/{scale}_cultural/{basename}?raw=true"
            response = requests.get(url=url)
            with geo_dirpath.joinpath(basename).open("wb") as fh:
                fh.write(response.content)
        return shapefile
