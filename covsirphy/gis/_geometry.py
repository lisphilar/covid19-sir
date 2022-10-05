#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import geopandas as gpd
from unidecode import unidecode
from covsirphy.util.config import config
from covsirphy.util.validator import Validator
from covsirphy.util.filer import Filer
from covsirphy.util.term import Term


class _Geometry(Term):
    """Class to add geometry information to geo-spatial data.

    Args:
        data (pandas.DataFrame): geo-spatial data
            Index
                reset index
            Columns
                - column defined by @layer
                - the other columns
        layer (str): layer name which has location information
        directory (str): directory to save GeoJSON files
    """

    def __init__(self, data, layer, directory):
        self._df = Validator(data, f"{layer}-level data").dataframe(columns=[layer])
        self._layer = layer
        self._filer = Filer(directory=directory)

    def to_geopandas(self, iso3, natural_earth):
        """Add geometry information with GeoJSON file of "Natural Earth" GitHub repository to data.

        Args:
            iso3 (str or None): ISO3 code (for the province/city-level data) or None (country-level data)
            natural_earth (str or None): title of GeoJSON file (without extension) of "Natural Earth" GitHub repository or None (automatically determined)

        Returns:
            geopandas.GeoDataFrame:
                Index:
                    - reset index
                Columns:
                    - columns included in @data of _Geometry()
                    - geometry: geometric information

        Note:
            GeoJSON files are listed in https://github.com/nvkelso/natural-earth-vector/tree/master/geojson
            https://www.naturalearthdata.com/
            https://github.com/nvkelso/natural-earth-vector
            Natural Earth (Free vector and raster map data at naturalearthdata.com, Public Domain)
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        gdf = self._natural_earth(title=natural_earth or self._natural_earth_parse_title(iso3=iso3))
        right_on = self.ISO3 if iso3 is None else "NAME"
        gdf = self._df.merge(gdf, how="left", left_on=self._layer, right_on=right_on)
        return gdf.rename(columns={f"{self.ISO3}_x": self.ISO3}).loc[:, [*self._df.columns.tolist(), "geometry"]]

    @staticmethod
    def _natural_earth_parse_title(iso3):
        """Find the best file title ("Natural Earth" GitHub repository) for the layer and country.

        Args:
            iso3 (str or None): ISO3 code (for the province/city-level data) or None (country-level data)

        Returns:
            str: parsed title
        """
        if iso3 is None:
            return "ne_110m_admin_0_countries"
        scale = "50m" if iso3 == "USA" else "10m"
        return f"ne_{scale}_admin_1_states_provinces"

    def _natural_earth(self, title):
        """Download GeoJSON files from "Natural Earth" GitHub repository.
        https://www.naturalearthdata.com/
        https://github.com/nvkelso/natural-earth-vector
        Natural Earth (Free vector and raster map data at naturalearthdata.com, Public Domain)

        Args:
            title (str): title of GeoJSON file (without extension) of "Natural Earth" GitHub repository

        Returns:
            geopandas.GeoDataFrame:
                Index:
                    - reset index
                Columns:
                    - ISO3: ISO3 codes
                    - NAME: Country names or province names
                    - geometry: geometric information

        Note:
            GeoJSON files are listed in https://github.com/nvkelso/natural-earth-vector/tree/master/geojson
        """
        file_dict = self._filer.geojson(title, driver="GeoJSON")
        filename = file_dict["filename"]
        if Path(filename).exists():
            gdf = gpd.read_file(filename)
        else:
            config.info("Retrieving GIS data from Natural Earth https://www.naturalearthdata.com/")
            url = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{title}.geojson"
            gdf = gpd.read_file(url)
            gdf.rename(columns={"ISO_A3": self.ISO3, "adm0_a3": self.ISO3, "name": "NAME"}, inplace=True)
            gdf["NAME"] = gdf["NAME"].fillna(self.NA).apply(unidecode)
            gdf["NAME"] = gdf["NAME"].replace(
                {"Xizang": "Tibet", "Inner Mongol": "Inner Mongolia", "S. Sudan": "South Sudan",
                 "Fr. S. Antarctic Lands": "French Southern and Antarctic Lands"})
            gdf.loc[gdf[self.ISO3] == "HKG", "NAME"] = "Hong Kong"
            gdf.loc[gdf[self.ISO3] == "-99", self.ISO3] = self._to_iso3(gdf.loc[gdf[self.ISO3] == "-99", "NAME"])
            gdf[self.ISO3] = gdf[self.ISO3].replace({"MAC": "CHN"})
            gdf.to_file(**file_dict)
        return gdf.reindex(columns=[self.ISO3, "NAME", "geometry"])
