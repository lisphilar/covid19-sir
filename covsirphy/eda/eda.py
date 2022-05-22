#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.gis.gis import GIS
from covsirphy.downloading.downloader import DataDownloader


class EDA(Term):
    """Class for data engineering (via covsirphy.DataEngineer) and explanatory data analysis (EDA) of geo-spatial time-series data.

    Args:
        layers (list[str] or None): list of layers of geographic information or None (["ISO3", "Province", "City"])
        country (str or None): layer name of countries or None (countries are not included in the layers)
        directory (str or pathlib.Path): directory to save downloaded datasets
        verbose (int): level of verbosity of stdout

    Raises:
        ValueError: @layers has duplicates

    Note:
        Country level data specified with @country will be stored with ISO3 codes.

    Note:
        If @verbose is 0, no descriptions will be shown.
        If @verbose is 1 or larger, details of layer adjustment will be shown.
    """

    def __init__(self, layers=None, country="ISO3", directory="input", verbose=1):
        self._layers = Validator(layers, "layers").sequence(default=[self.ISO3, self.PROVINCE, self.CITY])
        self._country = str(country)
        self._gis_kwargs = dict(layers=self._layers, country=self._country, date=self.DATE, verbose=verbose)
        self._gis = GIS(**self._gis_kwargs)
        self._directory = directory
        # Aliases
        variable_preset_dict = {
            "N": [self.N], "T": [self.TESTS], "C": [self.C], "F": [self.F], "R": [self.R],
            "CFR": [self.C, self.F, self.R],
            "CIFR": [self.C, self.CI, self.F, self.R],
            "CR": [self.C, self.R],
        }
        self._alias_dict = {"subset": {}, "variables": variable_preset_dict.copy()}

    def all(self, variables=None):
        """Return all available data.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Raises:
            NotRegisteredError: No records have been registered yet

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - (pandas.Category): columns defined by covsirphy.EDA(layers)
                    - Data (pandas.Timestamp): observation dates, column defined by covsirphy.EDA(date)
                    - columns defined by @variables
        """
        return self._gis.all(variables=variables, errors="raise")

    def citations(self, variables=None):
        """
        Return citation list of the secondary data sources.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            list[str]: citation list
        """
        return self._gis.citations(variables=variables)

    def register(self, data, citations=None, **kwargs):
        """Register new data.

        Args:
            data (pandas.DataFrame): new data
                Index
                    reset index
                Columns
                    - columns defined by covsirphy.EDA(layer)
                    - Date (str): observation dates
                    - Population (str): total population, optional
                    - Tests (str): column of the number of tests, optional
                    - Confirmed (str): the number of confirmed cases, optional
                    - Fatal (str): the number of fatal cases, optional
                    - Recovered (str): the number of recovered cases, optional
            citations (list[str] or str or None): citations of the dataset or None (["my own dataset"])
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"

        Returns:
            covsirphy.EDA: self
        """
        self._gis.register(
            data=data, layers=self._layers, date=self.DATE, variables=None,
            citations=citations or ["my own dataset"], convert_iso3=(self._country in self._layers), **kwargs)
        return self

    def download(self, country=None, province=None, databases=None, update_interval=12):
        """Download datasets from the recommended data servers using covsirphy.DataDownloader.

        Args:
            country(str or None): country name or None
            province(str or None): province / state / prefecture name or None
            databases(list[str] or None): refer to covsirphy.DataDownloader.layer()
            update_interval (int): update interval of downloading dataset
        """
        downloader = DataDownloader(directory=self._directory, update_interval=update_interval, verbose=self._verbose)
        df = downloader.layer(country=country, province=province, databases=databases)
        citations = downloader.citations()
        self.register(
            data=df, layers=[self.ISO3, self.PROVINCE, self.CITY], date=self.DATE, variables=None, citations=citations, convert_iso3=False)
        return self

    def layer(self, geo=None, start_date=None, end_date=None, variables=None):
        """Return the data at the selected layer in the date range.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to specify the layer or None (the top level)
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            variables (list[str] or None): list of variables to add or None (all available columns)

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
                    - Date (pandas.Timestamp): observation dates
                    - columns defined by @variables

        Note:
           Note that records with NAs as country names will be always removed.

        Note:
            Regarding @geo argument, please refer to covsirphy.GIS.layer().
        """
        return self._gis.layer(geo=geo, start_date=start_date, end_date=end_date, variables=variables, errors="raise")

    def choropleth(self, geo, variable, on=None, title="Choropleth map", filename="choropleth.jpg", logscale=True, natural_earth=None, **kwargs):
        """Create choropleth map.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to specify the layer or None (the top level)
            variable (str): variable name to show
            on (str or None): the date, like 22Jan2020, or None (the last date of each location)
            title (str): title of the map
            filename (str or None): filename to save the figure or None (display)
            logscale (bool): whether convert the value to log10 scale values or not
            natural_earth (str or None): title of GeoJSON file (without extension) of "Natural Earth" GitHub repository or None (automatically determined)
            kwargs: keyword arguments of the following classes and methods.
                - matplotlib.pyplot.savefig(), matplotlib.pyplot.legend(), and
                - pandas.DataFrame.plot()

        Note:
            Regarding @geo argument, please refer to covsirphy.GIS.layer().

        Note:
            GeoJSON files are listed in https://github.com/nvkelso/natural-earth-vector/tree/master/geojson
            https://www.naturalearthdata.com/
            https://github.com/nvkelso/natural-earth-vector
            Natural Earth (Free vector and raster map data at naturalearthdata.com, Public Domain)
        """
        layer_df = self.layer(geo=geo, variables=[variable])
        gis = GIS(**self._gis_kwargs)
        gis.register(data=layer_df, date=self.DATE)
        gis.choropleth(
            variable=variable, filename=filename, title=title, logscale=logscale,
            geo=geo, on=on, directory=[self._directory, "natural_earth"], natural_earth=natural_earth, **kwargs
        )

    def subset(self, geo=None, start_date=None, end_date=None, variables=None):
        """Return subset of the location and date range.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None (total at the top level)
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            variables (list[str] or None): list of variables to add or None (all available columns)

        Returns:
            pandas.DataFrame:
                Index:
                    reset index
                Columns
                    - (pandas.Timestamp): observation dates, column defined by covsirphy.GIS(date)
                    - columns defined by @variables

        Note:
            Regarding @geo argument, please refer to covsirphy.GIS.subset().
        """
        return self._gis.subset(geo=geo, start_date=start_date, end_date=end_date, variables=variables, errors="raise")

    def subset_alias(self, alias=None, update=False, **kwargs):
        """Set/get/list-up alias name(s) of subset.

        Args:
            alias (str or None): alias name or None (list-up alias names)
            update (bool): force updating the alias when @alias is not None
            **kwargs: keyword arguments of covsirphy.EDA.subset()

        Returns:
            pandas.DataFrame or dict[str, pandas.DataFrame]:
                - pandas.DataFrame: when @alias is not None, the subset of the alias
                - dict[str, pandas.DataFrame]: when @alias is None, dictionary of aliases and subsets

        Note:
            When the alias name was a new one, subset will be registered with covsirphy.EDA.subset(**kwargs).
        """
        if alias is None:
            return self._alias_dict["subset"]
        if update or alias not in self._alias_dict["subset"]:
            self._alias_dict["subset"][alias] = self.subset(**kwargs)
        return self._alias_dict["subset"][alias]

    def variables_alias(self, alias=None, variables=None):
        """Set/get/list-up alias name(s) of variables.

        Args:
            alias (str or None): alias name or None (list-up alias names)
            variables (list[str]): variables to register with the alias

        Returns:
            list[str] or dict[str, list[str]]:
                - list[str]: when @alias is not None, the variables of the alias
                - dict[str, list[str]]: when @alias is None, dictionary of aliases and variables

        Note:
            When @variables is not None, alias will be registered/updated.

        Note:
            Some aliases are preset. We can check them with EDA().variables_alias().
        """
        if alias is None:
            return self._alias_dict["variables"]
        if variables is not None:
            Validator(variables, "variables").sequence(candidates=self.all().columns.tolist())
            self._alias_dict["variables"][alias] = variables[:]
        return self._alias_dict["variables"][alias]
