#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from pathlib import Path
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
from covsirphy.util.config import config
from covsirphy.util.error import NotRegisteredError, SubsetNotFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.gis._subset import _SubsetManager
from covsirphy.gis._layer import _LayerAdjuster
from covsirphy.gis._geometry import _Geometry
from covsirphy.gis._choropleth import _ChoroplethMap


class GIS(Term):
    """Class of geographic information system to handle geo-spatial time-series data.

    Args:
        layers (list[str] or None): list of layers of geographic information or None (["ISO3", "Province", "City"])
        country (str or None): layer name of countries or None (countries are not included in the layers)
        date (str): column name of observation dates

    Raises:
        ValueError: @layers has duplicates

    Note:
        Country level data specified with @country will be stored with ISO3 codes.
    """

    def __init__(self, layers=None, country="ISO3", date="Date", **kwargs):
        # Countries will be specified with ISO3 codes and this requires conversion
        self._country = None if country is None else str(country)
        # Location data
        self._layers = Validator(layers or [self._country, self.PROVINCE, self.CITY], "layers").sequence()
        # Date column
        self._date = str(date)
        # Verbosity
        if "verbose" in kwargs:
            verbose = kwargs.get("verbose", 2)
            config.logger(level=verbose)
            config.warning(
                f"Argument verbose was deprecated, please use covsirphy.config.logger(level={verbose}) instead.")
        # Layer adjuster
        self._adjuster = _LayerAdjuster(layers=self._layers, country=self._country, date=self._date)
        self._un_registered = True

    def all(self, variables=None, errors="raise"):
        """Return all available data.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)
            errors (str): 'raise' or 'coerce'

        Raises:
            NotRegisteredError: No records have been registered yet

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - (pandas.Category): columns defined by covsirphy.GIS(layers)
                    - (pandas.Timestamp): observation dates, column defined by covsirphy.GIS(date)
                    - columns defined by @variables
        """
        if self._un_registered and errors == "raise":
            raise NotRegisteredError("No records have been registered yet.")
        df = self._adjuster.all(variables=variables)
        return df.astype(dict.fromkeys(self._layers, "category"))

    def citations(self, variables=None):
        """
        Return citation list of the secondary data sources.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            list[str]: citation list
        """
        return self._adjuster.citations(variables=variables)

    def register(self, data, layers=None, date="Date", variables=None, citations=None, convert_iso3=True, **kwargs):
        """Register new data.

        Args:
            data (pandas.DataFrame): new data
                Index
                    reset index
                Columns
                    - columns defined by @layers
                    - column defined by @date
                    - columns defined by @variables
            layers (list[str] or None): layers of the data or None (as the same as covsirphy.GIS(layer))
            date (str): column name of observation dates of the data
            variables (list[str] or None): list of variables to add or None (all available columns)
            citations (list[str] or str or None): citations of the dataset or None (["my own dataset"])
            convert_iso3 (bool): whether convert country names to ISO3 codes or not
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"

        Raises:
            ValueError: @data_layers has duplicates

        Returns:
            covsirphy.GIS: self
        """
        self._adjuster.register(
            data=data, layers=layers, date=date, variables=variables, citations=citations,
            convert_iso3=convert_iso3, **Validator(kwargs, "keyword arguments").kwargs(pd.to_datetime))
        self._un_registered = False
        return self

    def layer(self, geo=None, start_date=None, end_date=None, variables=None, errors="raise"):
        """Return the data at the selected layer in the date range.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to specify the layer or None (the top level)
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            variables (list[str] or None): list of variables to add or None (all available columns)
            errors (str): whether raise errors or not, 'raise' or 'coerce'

        Raises:
            TypeError: @geo has un-expected types
            ValueError: the length of @geo is larger than the length of layers
            NotRegisteredError: No records have been registered at the layer yet

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns
                    - (str): columns defined by covsirphy.GIS(layers)
                    - (pandas.Timestamp): observation dates, column defined by covsirphy.GIS(date)
                    - columns defined by @variables

        Note:
           Note that records with NAs as country names will be always removed.

        Note:
            When `geo=None` or `geo=(None,)`, returns country-level data, assuming we have country/province/city as layers here.

        Note:
            When `geo=("Japan",)` or `geo="Japan"`, returns province-level data in Japan.

        Note:
            When `geo=(["Japan", "UK"],)`, returns province-level data in Japan and UK.

        Note:
            When `geo=("Japan", "Kanagawa")`, returns city-level data in Kanagawa/Japan.

        Note:
            When `geo=("Japan", ["Tokyo", "Kanagawa"])`, returns city-level data in Tokyo/Japan and Kanagawa/Japan.
        """
        # Get all data
        if self._un_registered and errors == "raise":
            raise NotRegisteredError("GIS.register()", details="No records have been registered yet")
        data = self._adjuster.all(variables=variables)
        # Filter with geo
        geo_converted = self._parse_geo(geo=geo, data=data)
        manager = _SubsetManager(layers=self._layers)
        df = manager.layer(data=data, geo=geo_converted)
        if df.empty and errors == "raise":
            raise NotRegisteredError("GIS.register()", details="No records have been registered at the layer yet")
        # Filter with date
        series = df[self._date].copy()
        start = Validator(start_date).date(default=series.min())
        end = Validator(end_date).date(default=series.max())
        df = df.loc[(df[self._date] >= start) & (df[self._date] <= end)]
        if df.empty and errors == "raise":
            raise NotRegisteredError(
                "GIS.register()", details=f"No records have been registered at the layer yet from {start_date} to {end_date}")
        # Get representative records for dates
        df = df.groupby([*self._layers, self._date], dropna=True).first()
        return df.reset_index().convert_dtypes()

    def to_geopandas(self, geo=None, on=None, variables=None, directory=None, natural_earth=None):
        """Add geometry information with GeoJSON file of "Natural Earth" GitHub repository to data.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to specify the layer or None (the top level)
            on (str or None): the date, like 22Jan2020, or None (the last date of each location)
            variables (list[str] or None): list of variables to add or None (all available columns)
            directory (list[str] or tuple(str) or str): top directory name(s) to save GeoJSON files or None (directory of this this script)
            natural_earth (str or None): title of GeoJSON file (without extension) of "Natural Earth" GitHub repository or None (automatically determined)

        Raises:
            ValueError: country layer is not included in the dataset

        Returns:
            geopandas.GeoDataFrame:
                Index:
                    - reset index
                Columns:
                    - (str): layer focused on with @gis and GIS.layer()
                    - (pandas.Timestamp): observation dates, column defined by covsirphy.GIS(date)
                    - geometry: geometric information

        Note:
            Regarding @geo argument, please refer to covsirphy.GIS.layer().

        Note:
            GeoJSON files are listed in https://github.com/nvkelso/natural-earth-vector/tree/master/geojson
            https://www.naturalearthdata.com/
            https://github.com/nvkelso/natural-earth-vector
            Natural Earth (Free vector and raster map data at naturalearthdata.com, Public Domain)
        """
        if self._country not in self._layers:
            raise ValueError("This cannot be done because country layer is not included in the dataset.")
        df = self.layer(geo=geo, variables=variables)
        if on is None:
            df = df.sort_values(self._date, ascending=True).groupby(self._layers).last().reset_index()
        else:
            df = df.loc[df[self._date] == Validator(on).date()]
        focused_layer = [layer for layer in self._layers if df[layer][df[layer] != self.NA].nunique() > 0][-1]
        geometry = _Geometry(
            data=df, layer=focused_layer, directory=directory or Path(__file__).with_name("Natural_Earth"))
        iso3 = None if focused_layer == self._country else self._to_iso3(list(df[self._country].unique())[0])
        return geometry.to_geopandas(iso3=iso3, natural_earth=natural_earth).drop(set(self._layers) - {focused_layer}, axis=1)

    def choropleth(self, variable, filename, title="Choropleth map", logscale=True, **kwargs):
        """Create choropleth map.

        Args:
            variable (str): variable name to show
            filename (str or None): filename to save the figure or None (display)
            title (str): title of the map
            logscale (bool): whether convert the value to log10 scale values or not
            kwargs: keyword arguments of the following classes and methods.
                - covsirphy.GIS.to_geopandas() except for @variables,
                - matplotlib.pyplot.savefig(), matplotlib.pyplot.legend(), and
                - pandas.DataFrame.plot()
        """
        v = Validator(kwargs, "keyword arguments")
        gdf = self.to_geopandas(variables=[variable], **v.kwargs(functions=GIS.to_geopandas, default=None))
        focused_layer = [layer for layer in self._layers if layer in gdf.columns][0]
        gdf.rename(columns={focused_layer: "Location", variable: "Variable"}, inplace=True)
        with _ChoroplethMap(filename=filename, **v.kwargs(functions=plt.savefig, default=None)) as cm:
            cm.title = str(title)
            cm.plot(data=gdf, logscale=logscale, **v.kwargs(functions=gpd.GeoDataFrame.plot, default=None))

    def subset(self, geo=None, start_date=None, end_date=None, variables=None, errors="raise"):
        """Return subset of the location and date range.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None (total at the top level)
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            variables (list[str] or None): list of variables to add or None (all available columns)
            errors (str): whether raise errors or not, 'raise' or 'coerce'

        Raises:
            TypeError: @geo has un-expected types
            ValueError: the length of @geo is larger than the length of layers
            NotRegisteredError: No records have been registered yet
            SubsetNotFoundError: no records were found for the country and @errors is 'raise'

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns
                    - (pandas.Timestamp): observation dates, column defined by covsirphy.GIS(date)
                    - columns defined by @variables

        Note:
           Note that records with NAs as country names will be always removed.

        Note:
            When `geo=None` or `geo=(None,)`, returns global scale records (total values of all country-level data), assuming we have country/province/city as layers here.

        Note:
            When `geo=("Japan",)` or `geo="Japan"`, returns country-level data in Japan.

        Note:
            When `geo=(["Japan", "UK"],)`, returns country-level data of Japan and UK.

        Note:
            When `geo=("Japan", "Tokyo")`, returns province-level data of Tokyo/Japan.

        Note:
            When `geo=("Japan", ["Tokyo", "Kanagawa"])`, returns total values of province-level data of Tokyo/Japan and Kanagawa/Japan.

        Note:
            When `geo=("Japan", "Kanagawa", "Yokohama")`, returns city-level data of Yokohama/Kanagawa/Japan.

        Note:
            When `geo=(("Japan", "Kanagawa", ["Yokohama", "Kawasaki"])`, returns total values of city-level data of Yokohama/Kanagawa/Japan and Kawasaki/Kanagawa/Japan.
        """
        # Get all data
        if self._un_registered and errors == "raise":
            raise NotRegisteredError("GIS.register()", details="No records have been registered yet.")
        data = self._adjuster.all(variables=variables)
        # Filter with geo
        geo_converted = self._parse_geo(geo=geo, data=data)
        manager = _SubsetManager(layers=self._layers)
        df = manager.filter(data=data, geo=geo_converted)
        if df.empty and errors == "raise":
            raise SubsetNotFoundError(geo=geo)
        # Filter with date
        series = df[self._date].copy()
        start = Validator(start_date).date(default=series.min())
        end = Validator(end_date).date(default=series.max())
        df = df.loc[df[self._date].between(start, end)]
        if df.empty and errors == "raise":
            raise SubsetNotFoundError(geo=geo, start_date=start_date, end_date=end_date)
        # Calculate total value if geo=None
        if geo is None or geo[0] is None:
            variables_agg = list(set(df.columns) - {*self._layers, self._date})
            df = df.pivot_table(values=variables_agg, index=self._date, columns=self._layers[0], aggfunc="last")
            df = df.ffill().fillna(0).stack().reset_index()
        # Get representative records for dates
        with contextlib.suppress(IndexError, KeyError):
            df = df.drop(self._layers[1:], axis=1)
        df = df.groupby([self._layers[0], self._date], dropna=True).first().reset_index(level=self._date)
        return df.groupby(self._date, as_index=False).sum().convert_dtypes()

    @classmethod
    def area_name(cls, geo=None):
        """
        Return area name of the geographic information, like 'Japan', 'Tokyo/Japan', 'Japan_UK', 'the world'.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names

        Returns:
            str: area name
        """
        if geo is None or geo[0] is None:
            return "the world"
        names = [
            info if isinstance(info, str) else "_".join(list(info)) for info in ([geo] if isinstance(geo, str) else geo)]
        return cls.SEP.join(names[:: -1])

    def _parse_geo(self, geo, data):
        """Parse geographic specifier.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - (str): column defined by @country (of covsirphy.GIS) if @country is not None

        Returns:
            geo (tuple(list[str] or tuple(str) or str or None) or str or None): parsed location names
        """
        if geo is None:
            return geo
        return [self._info_to_iso3(info, self._layers[i], data) for i, info in enumerate([geo] if isinstance(geo, str) else geo)]

    def _info_to_iso3(self, geo_info, layer, data):
        """Convert a element of geographic specifier to ISO3 code.

        Args:
            geo_info (list[str] or tuple(str) or str or None): element of geographic specifier
            layer (str): layer of geographic information
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - (str): column defined by @country if @country is not None
        """
        if layer != self._country or geo_info is None or set(geo_info).issubset(data[layer].unique()):
            return geo_info
        return self._to_iso3(geo_info)
