from __future__ import annotations
import numpy as np
import pandas as pd
from typing_extensions import Self
from covsirphy.util.error import NotIncludedError
from covsirphy.util.alias import Alias
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.gis.gis import GIS
from covsirphy.downloading.downloader import DataDownloader
from covsirphy.engineering._cleaner import _DataCleaner
from covsirphy.engineering._transformer import _DataTransformer
from covsirphy.engineering._complement import _ComplementHandler


class DataEngineer(Term):
    """Class for data engineering including loading, cleaning, transforming, complementing, EDA (explanatory data analysis).

    Args:
        layers: list of layers of geographic information or None (`["ISO3", "Province", "City"]`)
        country: layer name of countries or None (countries are not included in the layers)

        Raises:
            ValueError: @layers has duplicates

        Note:
            Country level data specified with @country will be stored with ISO3 codes.
    """

    def __init__(self, layers: list[str] | None = None, country: str = "ISO3", **kwargs) -> None:
        self._layers = Validator(layers, "layers").sequence(default=[self.ISO3, self.PROVINCE, self.CITY])
        self._country = country
        self._gis_kwargs = dict(layers=self._layers, country=self._country, date=self.DATE)
        self._gis = GIS(**self._gis_kwargs)
        # Aliases
        self._var_alias = Alias.for_variables()
        self._subset_alias = Alias(target_class=tuple)

    def register(self, data: pd.DataFrame, citations: list[str] | str | None = None, **kwargs) -> Self:
        """Register new data.

        Args:
            data: new data
                Index
                    reset index
                Columns
                    - columns defined by covsirphy.DataEngineer(layer)
                    - Date (pandas.DataFrame): observation dates
                    - Population (int): total population, optional
                    - Tests (int): column of the number of tests, optional
                    - Confirmed (int): the number of confirmed cases, optional
                    - Fatal (int): the number of fatal cases, optional
                    - Recovered (int): the number of recovered cases, optional
                    - the other columns will be also registered
            citations: citations of the dataset or None (["my own dataset"])
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"

        Returns:
            updated `DataEngineer` instance
        """
        Validator(data, "data").dataframe(columns=[*self._layers, self.DATE])
        self._gis.register(
            data=data, layers=self._layers, date=self.DATE, variables=None,
            citations=citations or ["my own dataset"], convert_iso3=(self._country in self._layers), **kwargs)
        return self

    def download(self, **kwargs) -> Self:
        """Download datasets from the recommended data servers using covsirphy.DataDownloader.

        Args:
            **kwargs: keyword arguments of `covsirphy.DataDownloader()` and `covsirphy.DataDownloader.layer()`

        Returns:
            updated `DataEngineer` instance
        """
        validator = Validator(kwargs, name="keyword arguments")
        downloader = DataDownloader(**validator.kwargs(DataDownloader))
        df = downloader.layer(**validator.kwargs(DataDownloader.layer))
        citations = downloader.citations()
        self._gis.register(
            data=df, layers=[self.ISO3, self.PROVINCE, self.CITY], date=self.DATE, variables=None,
            citations=citations, convert_iso3=False, **kwargs)
        return self

    def all(self, variables: list[str] | str | None = None) -> pd.DataFrame:
        """Return all available data, converting dtypes with pandas.DataFrame.convert_dtypes().

        Args:
            variables: list of variables to collect or alias or None (all available variables)

        Raises:
            NotRegisteredError: No records have been registered yet

        Returns:
            Index
                reset index
            Column
                - columns defined by @layers of `DataEngineer()|
                - Date (pandas.Timestamp): observation dates defined by @date of `DataEngineer()`
                - the other columns
        """
        return self._gis.all(variables=self._var_alias.find(name=variables, default=variables), errors="raise").convert_dtypes()

    def citations(self, variables: list[str] | str | None = None) -> list[str]:
        """Return citation list of the secondary data sources.

        Args:
            variables: list of variables to collect or alias or None (all available variables)

        Returns:
            citations
        """
        return self._gis.citations(variables=self._var_alias.find(name=variables, default=variables))

    def clean(self, kinds: list[str] | None = None, **kwargs) -> Self:
        """Clean all registered data.

        Args:
            kinds: kinds of data cleaning with order or None (all available kinds as follows)

                - "convert_date": Convert dtype of date column to pandas.Timestamp.
                - "resample": Resample records with dates.
                - "fillna": Fill NA values with '-' (layers) and the previous values and 0.
            **kwargs: keyword arguments of data cleaning refer to note

        Returns:
            updated `DataEngineer` instance

        Note:
            When "convert_date" included, keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not" can be used.

        Note:
            When "resample" included, `date_range=<tuple of (str or None, str or None) or None>)` can be applied as keyword arguments to set the range.
        """
        cleaner = _DataCleaner(data=self._gis.all(), layers=self._layers, date=self.DATE)
        kind_dict = {
            "convert_date": cleaner.convert_date,
            "resample": cleaner.resample,
            "fillna": cleaner.fillna,
        }
        all_kinds = list(kind_dict.keys())
        selected = Validator(kinds, "kind").sequence(default=all_kinds, candidates=all_kinds)
        for kind in selected:
            kind_dict[kind](**Validator(kwargs, "keyword arguments").kwargs(functions=kind_dict[kind], default=None))
        return self._recreate_gis(cleaner)

    def transform(self) -> Self:
        """Transform all registered data, calculating the number of susceptible and infected cases.

        Returns:
            updated `DataEngineer` instance

        Note:
            - Susceptible = Population - Confirmed
            - Infected = Confirmed - Fatal - Recovered
        """
        all_df = self._gis.all()
        transformer = _DataTransformer(data=all_df, layers=self._layers, date=self.DATE)
        transformer.susceptible(new=self.S, population=self.N, confirmed=self.C)
        transformer.infected(new=self.CI, confirmed=self.C, fatal=self.F, recovered=self.R)
        return self._recreate_gis(transformer)

    def inverse_transform(self) -> Self:
        """Perform inverse transformation, calculating total population and confirmed.

        Returns:
            updated `DataEngineer` instance

        Note:
            - Population = Susceptible + Confirmed
            - Confirmed = Infected + Fatal + Recovered
        """
        Validator(self._gis.all(), "all registered data").dataframe(columns=[self.S, self.CI, self.F, self.R])
        self.add(columns=[self.CI, self.F, self.R], new=self.C)
        self.add(columns=[self.S, self.C], new=self.N)
        return self

    def diff(self, column: str, suffix: str = "_diff", freq: str = "D") -> Self:
        """Calculate daily new cases with "f(x>0) = F(x) - F(x-1), x(0) = 0 when F is cumulative numbers".

        Args:
            column: column name of the cumulative numbers
            suffix: suffix if the column (new column name will be '{column}{suffix}')
            freq: offset aliases of shifting dates

        Returns:
            updated `DataEngineer` instance

        Note:
            Regarding @freq, refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self.DATE)
        transformer.diff(
            column=Validator(
                self._var_alias.find(name=column, default=[column]), "column", accept_none=False).sequence(length=1)[0],
            suffix=suffix, freq=freq)
        return self._recreate_gis(transformer)

    def add(self, columns: list[str] | str, new: str | None = None, fill_value: float | int = 0) -> Self:
        """Calculate element-wise addition with `pandas.DataFrame.sum(axis=1)`, X1 + X2 + X3 +...

        Args:
            columns: columns (or alias) to add
            new: column name of addition or None (f"{X1}+{X2}+{X3}...")
            fill_value: value to fill in NAs

        Returns:
            updated `DataEngineer` instance
        """
        col_names = self._var_alias.find(name=columns, default=columns)
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self.DATE)
        transformer.add(columns=col_names, new=new or "+".join(col_names), fill_value=fill_value)
        return self._recreate_gis(transformer)

    def mul(self, columns: list[str] | str, new: str | None = None, fill_value: float | int = 0) -> Self:
        """Calculate element-wise multiplication with `pandas.DataFrame.product(axis=1)`, X1 * X2 * X3 *...

        Args:
            columns: columns (or alias) to multiply
            new: column name of multiplication or None (f"{X1}*{X2}*{X3}...")
            fill_value: value to fill in NAs

        Returns:
            updated `DataEngineer` instance
        """
        col_names = self._var_alias.find(name=columns, default=columns)
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self.DATE)
        transformer.mul(columns=col_names, new=new or "*".join(col_names), fill_value=fill_value)
        return self._recreate_gis(transformer)

    def sub(self, minuend: str, subtrahend: str, new: str | None = None, fill_value: float | int = 0) -> Self:
        """Calculate element-wise subtraction with `pandas.Series.sub()`, minuend - subtrahend.

        Args:
            minuend: numerator column
            subtrahend: subtrahend column
            new: column name of subtraction or None (f"{minuend}-{subtrahend}")
            fill_value: value to fill in NAs

        Returns:
            updated `DataEngineer` instance
        """
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self.DATE)
        transformer.sub(
            minuend=Validator(
                self._var_alias.find(name=minuend, default=[minuend]), "minuend", accept_none=False).sequence(length=1)[0],
            subtrahend=Validator(
                self._var_alias.find(name=subtrahend, default=[subtrahend]), "subtrahend", accept_none=False).sequence(length=1)[0],
            new=new or f"{minuend}-{subtrahend}", fill_value=fill_value)
        return self._recreate_gis(transformer)

    def div(self, numerator: str, denominator: str, new: str | None = None, fill_value: float | int = 0) -> Self:
        """Calculate element-wise floating division with `pandas.Series.div()`, numerator / denominator.

        Args:
            numerator: numerator column
            denominator: denominator column
            new: column name of floating division or None (f"{numerator}_per_({denominator.replace(' ', '_')})")
            fill_value: value to fill in NAs

        Returns:
            updated `DataEngineer` instance

        Note:
            Positive rate could be calculated with Confirmed / Tested, `.div(numerator="Confirmed", denominator="Tested", new="Positive_rate")`
        """
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self.DATE)
        transformer.div(
            numerator=Validator(
                self._var_alias.find(name=numerator, default=[numerator]), "numerator", accept_none=False).sequence(length=1)[0],
            denominator=Validator(
                self._var_alias.find(name=denominator, default=[denominator]), "denominator", accept_none=False).sequence(length=1)[0],
            new=new or f"{numerator}_per_({denominator.replace(' ', '_')})", fill_value=fill_value)
        return self._recreate_gis(transformer)

    def assign(self, **kwargs) -> Self:
        """Assign a new column with `pandas.DataFrame.assign()`.

        Args:
            **kwargs: dict of {str: callable or pandas.Series}

        Note:
            Refer to documentation of `pandas.DataFrame.assign()`, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html
        """
        transformer = _DataTransformer(data=self._gis.all(), layers=self._layers, date=self.DATE)
        transformer.assign(**kwargs)
        return self._recreate_gis(transformer)

    def _recreate_gis(self, transformer: _DataTransformer) -> Self:
        """Recreate GIS instance with transformer.

        Args:
            transformer
        """
        citations = self._gis.citations(variables=None)
        self._gis = GIS(**self._gis_kwargs)
        self._gis.register(
            data=transformer.all(), layers=self._layers, date=self.DATE,
            variables=None, citations=citations, convert_iso3=False)
        return self

    def layer(self, geo: tuple[list[str] | tuple[str] | str | None, ...] = None,
              start_date: str | None = None, end_date: str | None = None, variables: list[str] | None = None) -> pd.DataFrame:
        """Return the data at the selected layer in the date range.

        Args:
            geo: location names to specify the layer or None (the top level)
            start_date: start date, like 22Jan2020
            end_date: end date, like 01Feb2020
            variables: list of variables to add or None (all available columns)

        Raises:
            TypeError: @geo has un-expected types
            ValueError: the length of @geo is larger than the length of layers
            NotRegisteredError: No records have been registered at the layer yet

        Returns:
            Index:
                reset index
            Columns
                - (str): columns defined by covsirphy.GIS(layers)
                - Date (pandas.Timestamp): observation dates
                - columns defined by @ variables

        Note:
           Note that records with NAs as country names will be always removed.

        Note:
            Regarding @geo argument, please refer to `covsirphy.GIS.layer()`.
        """
        v_converted = self._var_alias.find(name=variables, default=variables)
        return self._gis.layer(geo=geo, start_date=start_date, end_date=end_date, variables=v_converted, errors="raise")

    def choropleth(self, geo: tuple[list[str] | tuple[str] | str | None, ...], variable: str,
                   on: str | None = None, title: str = "Choropleth map", filename: str = "choropleth.jpg",
                   logscale: bool = True, directory: str | None = None, natural_earth: str | None = None, **kwargs) -> None:
        """Create choropleth map.

        Args:
            geo: location names to specify the layer or None (the top level)
            variable: variable name to show
            on: the date, like 22Jan2020, or None (the last date of each location)
            title: title of the map
            filename: filename to save the figure or None (display)
            logscale: whether convert the value to log10 scale values or not
            directory: directory to save GeoJSON file of "Natural Earth" GitHub repository or None (the directory of GIS class script)
            natural_earth: title of GeoJSON file(without extension) of "Natural Earth" GitHub repository or None (automatically determined)
            **kwargs: keyword arguments of the following classes and methods.

                - `matplotlib.pyplot.savefig()`,
                - `matplotlib.pyplot.legend()`, and
                - pandas.DataFrame.plot()`

        Note:
            Regarding @ geo argument, please refer to `covsirphy.GIS.layer()`.

        Note:
            GeoJSON files are listed in

            - https://github.com/nvkelso/natural-earth-vector/tree/master/geojson
            - https://www.naturalearthdata.com/
            - https://github.com/nvkelso/natural-earth-vector
            - Natural Earth (Free vector and raster map data at naturalearthdata.com, Public Domain)
        """
        layer_df = self.layer(geo=geo, variables=[variable])
        gis = GIS(**self._gis_kwargs)
        gis.register(data=layer_df, date=self.DATE)
        gis.choropleth(
            variable=variable, filename=filename, title=title, logscale=logscale,
            geo=geo, on=on, directory=directory, natural_earth=natural_earth, **kwargs)

    def subset(self, geo: tuple[list[str] | tuple[str] | str | None, ...] = None,
               start_date: str | None = None, end_date: str | None = None,
               variables: list[str] | str | None = None, complement: bool = True, get_dummies: bool = True,
               **kwargs) -> tuple[pd.DataFrame, str, dict[str, bool]]:
        """Return subset of the location and date range.

        Args:
            geo: location names to filter or None (total at the top level)
            start_date: start date, like 22Jan2020
            end_date: end date, like 01Feb2020
            variables: list of variables to add or None (all available columns)
            complement: whether perform data complement or not, True as default
            get_dummies: whether convert categorical variable into dummy variables or not, True as default
            **Kwargs: keyword arguments for complement and default values

                - recovery_period (int): expected value of recovery period[days], 17
                - interval (int): expected update interval of the number of recovered cases[days], 2
                - max_ignored (int): Max number of recovered cases to be ignored[cases], 100
                - max_ending_unupdated (int): Max number of days to apply full complement, where max recovered cases are not updated[days], 14
                - upper_limit_days (int): maximum number of valid partial recovery periods[days], 90
                - lower_limit_days (int): minimum number of valid partial recovery periods[days], 7
                - upper_percentage (float): fraction of partial recovery periods with value greater than upper_limit_days, 0.5
                - lower_percentage (float): fraction of partial recovery periods with value less than lower_limit_days, 0.5

        Returns:
            - pandas.DataFrame
                Index
                    Date(pandas.DataFrame): observation dates
                Columns
                    Population(int): total population
                    Tests(int): column of the number of tests
                    Confirmed(int): the number of confirmed cases
                    Fatal(int): the number of fatal cases
                    Recovered(int): the number of recovered cases
                    the other columns registered

            - str: status code: will be selected from

                - '' (not complemented)
                - 'monotonic increasing complemented confirmed data'
                - 'monotonic increasing complemented fatal data'
                - 'monotonic increasing complemented recovered data'
                - 'fully complemented recovered data'
                - 'partially complemented recovered data'

            - dict[str, bool]: status for each complement type, keys are

                - Monotonic_confirmed
                - Monotonic_fatal
                - Monotonic_recovered
                - Full_recovered
                - Partial_recovered

        Note:
            Regarding @geo argument, please refer to `covsirphy.GIS.subset()`.

        Note:
            Re-calculation of Susceptible and Infected will be done automatically.
        """
        v_converted = self._var_alias.find(name=variables, default=variables)
        subset_df = self._gis.subset(geo=geo, start_date=start_date, end_date=end_date, variables=None, errors="raise")
        if not complement:
            df = subset_df.set_index(self.DATE)
            return df.loc[:, v_converted or df.columns].convert_dtypes(), "", {}
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
        c_df, status, status_dict = handler.run(data=subset_df)
        df = pd.concat([subset_df.drop([self.DATE, self.C, self.F, self.R], axis=1), c_df], axis=1)
        df["location"] = self.NA
        transformer = _DataTransformer(data=df, layers=["location"], date=self.DATE)
        transformer.susceptible(new=self.S, population=self.N, confirmed=self.C)
        transformer.infected(new=self.CI, confirmed=self.C, fatal=self.F, recovered=self.R)
        transformed_df = transformer.all().drop("location", axis=1).set_index(self.DATE)
        if get_dummies:
            transformed_df = pd.get_dummies(transformed_df, dtype=float).convert_dtypes()
        return transformed_df.loc[:, v_converted or transformed_df.columns], status, status_dict

    def subset_alias(self, alias: str | None = None, update: bool = False,
                     **kwargs) -> tuple[pd.DataFrame, str, dict[str, bool]] | dict[str, tuple[pd.DataFrame, str, dict[str, bool]]]:
        """Set/get/list-up alias name(s) of subset.

        Args:
            alias: alias name or None (list-up alias names)
            update: force updating the alias when @alias is not None
            **kwargs: keyword arguments of covsirphy.DataEngineer().subset()

        Returns:
            - tuple[pandas.DataFrame, str, dict]: when @alias is not None, the subset of the alias
            - dict[str, tuple[pandas.DataFrame, str, dict]]: when @alias is None, dictionary of aliases and subsets

        Note:
            When the alias name was a new one, subset will be registered with `covsirphy.DataEngineer.subset(**kwargs)`.
        """
        if alias is None:
            return self._subset_alias.all()
        result = self._subset_alias.find(alias, default=None)
        if update or result is None:
            self._subset_alias.update(name=alias, target=self.subset(**kwargs))
        return self._subset_alias.find(alias)

    def variables_alias(self, alias: str | None = None, variables: list[str] | None = None) -> list[str] | dict[str, list[str]]:
        """Set/get/list-up alias name(s) of variables.

        Args:
            alias: alias name or None (list - up alias names)
            variables: variables to register with the alias

        Raises:
            NotIncludedError: the alias is not None and un - registered

        Returns:
            - list[str]: when @alias is not None, the variables of the alias
            - dict[str, list[str]]: when @alias is None, dictionary of aliases and variables

        Note:
            When @variables is not None, alias will be registered/updated.

        Note:
            Some aliases are preset. We can check them with `covsirphy.DataEngineer().variables_alias()`.
        """
        if alias is None:
            return self._var_alias.all()
        if variables is not None:
            self._var_alias.update(name=alias, target=variables)
        elif alias not in self._var_alias.all():
            raise NotIncludedError(alias, "keys of alias dictionary of variables")
        return self._var_alias.find(name=alias)

    @classmethod
    def recovery_period(cls, data: pd.DataFrame) -> int:
        """Calculate mode value of recovery period of the data.

        Args:
            data: data for calculation
                Index
                    Date (pandas.Timestamp): observation dates
                Columns
                    - Confirmed (int): the number of confirmed cases, optional
                    - Fatal (int): the number of fatal cases, optional
                    - Recovered (int): the number of recovered cases, optional
                    - the other columns will be ignored

        Returns:
            mode value of recovery period [days]
        """
        df = Validator(data, "data").dataframe(time_index=True, columns=[cls.C, cls.F, cls.R], empty_ok=False)
        df = df.resample("D").sum()
        df["diff"] = df[cls.C] - df[cls.F]
        df = df.loc[:, ["diff", cls.R]].unstack().reset_index()
        df.columns = ["Variable", "Date", "Number"]
        df["Days"] = (df["Date"] - df["Date"].min()).dt.days
        df = df.pivot_table(values="Days", index="Number", columns="Variable")
        df = df.interpolate(limit_area="inside").dropna().astype(np.int64)
        df["Elapsed"] = df[cls.R] - df["diff"]
        df = df.loc[df["Elapsed"] > 0]
        return 0 if df.empty else round(df["Elapsed"].mode().mean())
