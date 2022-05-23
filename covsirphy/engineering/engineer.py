#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import NotIncludedError, UnExecutedError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.gis.gis import GIS
from covsirphy.downloading.downloader import DataDownloader
from covsirphy.engineering.cleaner import _DataCleaner
from covsirphy.engineering.transformer import _DataTransformer
from covsirphy.engineering.complement import _ComplementHandler


class DataEngineer(Term):
    """Class for data engineering including loading, cleaning, transforming, complementing, EDA (explanatory data analysis).

    Args:
        layers (list[str] or None): list of layers of geographic information or None (["ISO3", "Province", "City"])
        country (str or None): layer name of countries or None (countries are not included in the layers)
        verbose (int): level of verbosity of stdout

        Raises:
            ValueError: @layers has duplicates

        Note:
            Country level data specified with @country will be stored with ISO3 codes.

        Note:
            If @verbose is 0, no descriptions will be shown.
            If @verbose is 1 or larger, details of layer adjustment will be shown.
    """

    def __init__(self, layers=None, country="ISO3", verbose=1):
        self._layers = Validator(layers, "layers").sequence(default=[self.ISO3, self.PROVINCE, self.CITY])
        self._country = str(country)
        self._verbose = Validator(verbose, "verbose").int(value_range=(0, None))
        self._gis_kwargs = dict(layers=self._layers, country=self._country, date=self.DATE, verbose=verbose)
        self._gis = GIS(**self._gis_kwargs)
        # Aliases
        _variable_preset_dict = {
            "N": [self.N], "S": [self.S], "T": [self.TESTS], "C": [self.C], "I": [self.CI], "F": [self.F], "R": [self.R],
            "CFR": [self.C, self.F, self.R],
            "CIFR": [self.C, self.CI, self.F, self.R],
            "CR": [self.C, self.R],
        }
        self._alias_dict = {"subset": {}, "variables": _variable_preset_dict.copy()}

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
            covsirphy.DataEngineer: self
        """
        self._gis.register(
            data=data, layers=self._layers, date=self.DATE, variables=None,
            citations=citations or ["my own dataset"], convert_iso3=(self._country in self._layers), **kwargs)
        return self

    def download(self, country=None, province=None, **kwargs):
        """Download datasets from the recommended data servers using covsirphy.DataDownloader.

        Args:
            country(str or None): country name or None
            province(str or None): province / state / prefecture name or None
            **kwargs: the other keyword arguments of covsirphy.DataDownloader() and covsirphy.DataDownloader.layer()

        Returns:
            covsirphy.DataEngineer: self

        Note:
            When @verbose is not included in **kwargs, covsirphy.DataEngineer(verbose) will be used.
        """
        default_dict = {"country": country, "province": province, "verbose": self._verbose}
        validator = Validator(kwargs, name="the other keyword arguments")
        downloader = DataDownloader(**validator.kwargs(DataEngineer, default=default_dict))
        df = downloader.layer(**validator.kwargs(DataDownloader.layer, default=default_dict))
        citations = downloader.citations()
        self.register(
            data=df, layers=[self.ISO3, self.PROVINCE, self.CITY], date=self.DATE, variables=None, citations=citations, convert_iso3=False)
        return self

    def all(self, variables=None):
        """Return all available data, converting dtypes with pandas.DataFrame.convert_dtypes().

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Raises:
            NotRegisteredError: No records have been registered yet

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Column
                    - columns defined by @layers of _DataEngineer()
                    - (pandas.Timestamp): observation dates defined by @date of _DataEngineer()
                    - the other columns
        """
        return self._gis.all(variables=self._alias_dict["variables"].get(variables, variables), errors="raise").convert_dtypes()

    def citations(self, variables=None):
        """
        Return citation list of the secondary data sources.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            list[str]: citation list
        """
        return self._gis.citations(variables=self._alias_dict["variables"].get(variables, variables))

    def clean(self, kinds=None, **kwargs):
        """Clean all registered data.

        Args:
            kinds (list[str] or None): kinds of data cleaning with order or None (all available kinds as follows)
                - "convert_date": Convert dtype of date column to pandas.Timestamp.
                - "resample": Resample records with dates.
                - "fillna": Fill NA values with '-' (layers) and the previous values and 0.
            **kwargs: keyword arguments of data cleaning refer to note

        Returns:
            covsirphy.DataEngineer: self

        Note:
            For "convert_date", keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not" can be used.
        """
        citations = self._gis.citations(variables=None)
        cleaner = _DataCleaner(data=self._gis.all(), layers=self._layers, date=self._date)
        kind_dict = {
            "convert_date": cleaner.convert_date,
            "resample": cleaner.resample,
            "fillna": cleaner.fillna,
        }
        all_kinds = list(kind_dict.keys())
        selected = Validator(kinds, "kind").sequence(default=all_kinds, candidates=all_kinds)
        for kind in selected:
            try:
                kind_dict[kind](**Validator(kwargs, "keyword arguments").kwargs(functions=kind_dict[kind], default=None))
            except UnExecutedError:
                raise UnExecutedError(
                    "DataEngineer.clean(kinds=['convert_date'])", details=f"Column {self._date} was not a column of date") from None
        self._gis = GIS(**self._gis_kwargs)
        self._gis.register(
            data=cleaner.all(), layers=self._layers, date=self._date, variables=None, citations=citations, convert_iso3=False)
        return self

    def transform(self, susceptible="Susceptible", infected="Infected", **kwargs):
        """Transform all registered data, calculating the number of susceptible and infected cases.

        Args:
            susceptible (str or None): the number of susceptible cases or None (will not be calculated)
            infected (str or None): the number of infected cases or None (will not be calculated)
            kwargs (dict[str, str]): dictionary of existed columns
                - population (str): total population, "Population" as default
                - confirmed (str): the number of confirmed cases, "Confirmed" as default
                - fatal (str): the number of fatal cases, "Fatal" as default
                - recovered (str): the number of recovered cases, "Recovered" as default

        Returns:
            covsirphy.DataEngineer: self

        Note:
            Susceptible = Population - Confirmed.

        Note:
            Infected = Confirmed - Fatal - Recovered.
        """
        default_dict = {"population": self.N, "confirmed": self.C, "fatal": self.F, "recovered": self.R}
        kwargs_dict = Validator(kwargs, "keyword arguments").dict(default=default_dict)
        all_df = self._gis.all()
        citations = self._gis.citations(variables=None)
        transformer = _DataTransformer(data=all_df, layers=self._layers, date=self._date)
        # Susceptible
        s_cols = [v for k, v in kwargs_dict.items() if k in ["population", "confirmed"]]
        if susceptible is not None and None not in set(s_cols):
            Validator(all_df, "all registered data").dataframe(columns=s_cols)
            transformer.susceptible(new=susceptible, **kwargs)
        # Infected
        i_cols = [v for k, v in kwargs_dict.items() if k in ["confirmed", "fatal", "recovered"]]
        if infected is not None and None not in set(i_cols):
            Validator(all_df, "all registered data").dataframe(columns=i_cols)
            transformer.infected(new=infected, **kwargs)
        self._gis = GIS(**self._gis_kwargs)
        self._gis.register(
            data=transformer.all(), layers=self._layers, date=self._date, variables=None, citations=citations, convert_iso3=False)
        return self

    def diff(self, column, suffix="_diff", freq="D"):
        """Calculate daily new cases with "f(x>0) = F(x) - F(x-1), x(0) = 0 when F is cumulative numbers".

        Args:
            column (str): column name of the cumulative numbers
            suffix (str): suffix if the column (new column name will be '{column}{suffix}')
            freq (str): offset aliases of shifting dates

        Returns:
            covsirphy.DataEngineer: self

        Note:
            Regarding @freq, refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Note:
            If the alias of values
        """
        citations = self._gis.citations(variables=None)
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self._date)
        transformer.diff(
            column=self.variables_alias(column, length=1)[
                0] if column in self._alias_dict["variables"].keys() else column,
            suffix=suffix, freq=freq)
        self._gis = GIS(**self._gis_kwargs)
        self._gis.register(
            data=transformer.all(), layers=self._layers, date=self._date, variables=None, citations=citations, convert_iso3=False)
        return self

    def div(self, numerator, denominator, new=None, fill_value=0):
        """Calculate element-wise floating division, numerator / denominator * 100.

        Args:
            numerator (str): numerator column
            denominator (str): denominator column
            new (str or None): column name of floating division or None (f"{numerator}_per_({denominator.replace(' ', '_')})")
            fill_value (float): value to fill in NAs

        Note:
            Positive rate could be calculated with Confirmed / Tested * 100 (%), `.div(numerator="Confirmed", denominator="Tested", new="Positive_rate")`
        """
        citations = self._gis.citations(variables=None)
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self._date)
        transformer.div(
            numerator=self.variables_alias(numerator, length=1)[
                0] if numerator in self._alias_dict["variables"].keys() else numerator,
            denominator=self.variables_alias(denominator, length=1)[
                0] if denominator in self._alias_dict["variables"].keys() else denominator,
            new=new or f"{numerator}_per_({denominator.replace(' ', '_')})", fill_value=fill_value)
        self._gis = GIS(**self._gis_kwargs)
        self._gis.register(
            data=transformer.all(), layers=self._layers, date=self._date, variables=None, citations=citations, convert_iso3=False)
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
        variables_converted = self.variables_alias(
            variables) if variables in self._alias_dict["variables"].keys() else variables
        return self._gis.layer(geo=geo, start_date=start_date, end_date=end_date, variables=variables_converted, errors="raise")

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

    def subset(self, geo=None, start_date=None, end_date=None, variables=None, complement=True, **kwargs):
        """Return subset of the location and date range.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None (total at the top level)
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            variables (list[str] or None): list of variables to add or None (all available columns)
            **Kwargs: keyword arguments for complement and default values
                recovery_period (int): expected value of recovery period [days], 17
                interval (int): expected update interval of the number of recovered cases [days], 2
                max_ignored (int): Max number of recovered cases to be ignored [cases], 100
                max_ending_unupdated (int): Max number of days to apply full complement, where max recovered cases are not updated [days], 14
                upper_limit_days (int): maximum number of valid partial recovery periods [days], 90
                lower_limit_days (int): minimum number of valid partial recovery periods [days], 7
                upper_percentage (float): fraction of partial recovery periods with value greater than upper_limit_days, 0.5
                lower_percentage (float): fraction of partial recovery periods with value less than lower_limit_days, 0.5

        Returns:
            tuple(pandas.DataFrame, str, dict):
                pandas.DataFrame
                    Index
                        reset index
                    Columns
                        - Date (pd.Timestamp): Observation date
                        - Confirmed (int): the number of confirmed cases
                        - Fatal (int): the number of fatal cases
                        - Recovered (int): the number of recovered cases
                str: status code: will be selected from
                    - '' (not complemented)
                    - 'monotonic increasing complemented confirmed data'
                    - 'monotonic increasing complemented fatal data'
                    - 'monotonic increasing complemented recovered data'
                    - 'fully complemented recovered data'
                    - 'partially complemented recovered data'
                dict[str, bool]: status for each complement type, keys are
                    - Monotonic_confirmed
                    - Monotonic_fatal
                    - Monotonic_recovered
                    - Full_recovered
                    - Partial_recovered

        Note:
            Regarding @geo argument, please refer to covsirphy.GIS.subset().
        """
        subset_df = self._gis.subset(
            geo=geo, start_date=start_date, end_date=end_date, variables=variables, errors="raise")
        if not complement:
            return subset_df.convert_dtypes(), "", {}
        default_kwargs = {
            "recovery_period": 17,
            "interval": 2,
            "max_ignored": 100,
            "max_ending_unupdated": 14,
            "upper_limit_days": 90,
            "lower_limit_days": 7,
            "upper_percentage": 0.5,
            "lower_percentage": 0.5,
        }
        handler = _ComplementHandler(
            **Validator(kwargs, "keyword arguments").kwargs(_ComplementHandler, default=default_kwargs))
        return handler.run()

    def subset_alias(self, alias=None, update=False, **kwargs):
        """Set/get/list-up alias name(s) of subset.

        Args:
            alias (str or None): alias name or None (list-up alias names)
            update (bool): force updating the alias when @alias is not None
            **kwargs: keyword arguments of covsirphy.DataEngineer.subset()

        Returns:
            tuple(pandas.DataFrame, str, dict) or dict[str, tuple(pandas.DataFrame, str, dict)]:
                - tuple(pandas.DataFrame, str, dict): when @alias is not None, the subset of the alias
                - dict[str, tuple(pandas.DataFrame, str, dict)]: when @alias is None, dictionary of aliases and subsets

        Note:
            When the alias name was a new one, subset will be registered with covsirphy.DataEngineer.subset(**kwargs).
        """
        if alias is None:
            return self._alias_dict["subset"]
        if update or alias not in self._alias_dict["subset"]:
            self._alias_dict["subset"][alias] = self.subset(**kwargs)
        return self._alias_dict["subset"][alias]

    def variables_alias(self, alias=None, variables=None, length=None):
        """Set/get/list-up alias name(s) of variables.

        Args:
            alias (str or None): alias name or None (list-up alias names)
            variables (list[str]): variables to register with the alias
            length (int or None): the number of the variables for validation when return or None (no validation)

        Raises:
            NotIncludedError: the alias is not None and un-registered
            UnExpectedLengthError: the number of elements is not the same as @length

        Returns:
            list[str] or dict[str, list[str]]:
                - list[str]: when @alias is not None, the variables of the alias
                - dict[str, list[str]]: when @alias is None, dictionary of aliases and variables

        Note:
            When @variables is not None, alias will be registered/updated.

        Note:
            Some aliases are preset. We can check them with covsirphy.DataEngineer().variables_alias().
        """
        if alias is None:
            return self._alias_dict["variables"]
        if variables is not None:
            Validator(variables, "variables").sequence(candidates=self._gis.all().columns.tolist())
            self._alias_dict["variables"][alias] = variables[:]
        try:
            selected = self._alias_dict["variables"][alias]
        except KeyError:
            raise NotIncludedError(alias, "keys of alias dictionary of variables") from None
        return Validator(selected, f"variables selected with alias '{alias}'").sequence(length=length)

    def recovery_period(self, **kwargs):
        raise NotImplementedError

    def subset_show(self, subset_alias, variables, plot_type="line", title=None, filename=None, **kwargs):
        raise NotImplementedError

    def layer_show(self, geo, variable, plot_type="line", title=None, filename=None, **kwargs):
        raise NotImplementedError
