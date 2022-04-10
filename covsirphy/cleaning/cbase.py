#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from pathlib import Path
import warnings
import country_converter as coco
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.util.error import deprecate, SubsetNotFoundError, UnExpectedValueError
from covsirphy.util.term import Term
from covsirphy.visualization.colored_map import ColoredMap
from covsirphy.cleaning.geography import Geography


class CleaningBase(Term):
    """
    Basic class for data cleaning.

    Args:
        filename (str or None): CSV filename of the dataset
        data (pandas.DataFrame or None):
            Index
                reset index
            Columns
                - Date: Observation date
                - (str): location identifiers defined by @layers
                - (str): variable names defined by @variables
        citation (str or None): citation or None (empty)
        layers (list[str] or None): list of location identifiers or None (ISO3, Country, Province)
        variables (list[str] or None): variables to clean (not including date and location identifiers)

    Note:
        Either @filename (high priority) or @data must be specified.

    Note:
        - If @filename is None, geography information will be saved in "input" directory.
        - If @filename is not None, geography information will be saved in the directory which has the file.
        - The directory of geography information could be changed with .directory property.
    """
    _LOC = "Location_ID"
    _LOC_COLS = [Term.COUNTRY, Term.ISO3, Term.PROVINCE]

    def __init__(self, filename=None, data=None, citation=None, layers=None, variables=None):
        self._layers = self._ensure_list(layers or self._LOC_COLS[:], name="layers")
        # Columns of self._raw, self._clean_df and self.cleaned()
        self._raw_cols = [self.DATE] + self._layers + (variables or [])
        self._subset_cols = [self.DATE] + (variables or [])
        # Raw data
        self._raw = self._parse_raw(filename, data, self._raw_cols)
        # Location data
        loc_df = self._raw[self._layers].drop_duplicates(ignore_index=True)
        for layer in self._layers:
            loc_df.loc[:, layer] = loc_df[layer].fillna(self.NA)
        loc_df.loc[:, self._LOC] = "id" + loc_df.index.astype("str").str.zfill(len(str(len(loc_df))))
        self._loc_df = loc_df.copy()
        # Data cleaning
        if self._raw.empty:
            self._value_df = pd.DataFrame(columns=[self._LOC, *self._subset_cols])
        else:
            df = self._raw.merge(loc_df, how="left", on=self._layers)
            self._value_df = self._cleaning(raw=df.drop(self._layers, axis=1))
        # Citation
        self._citation = citation or ""
        # Directory that save the file
        if filename is None:
            self._dirpath = Path("input")
        else:
            Path(filename).parent.mkdir(exist_ok=True, parents=True)
            self._dirpath = Path(filename).resolve().parent

    def _parse_raw(self, filename, data, columns):
        """
        Parse raw data with a CSV file or a dataframe.

        Args:
            filename (str or None): CSV filename of the dataset
            data (pandas.DataFrame or None):
                Index
                    reset index
                Columns
                    columns defined by @required_cols
            columns (list[str]): column names to use

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    columns defined by @required_cols

        Note:
            Either @filename (high priority) or @data must be specified.

        Note:
            If some columns are not included in the dataset, values will be None.
        """
        if filename is None:
            if data is None:
                return pd.DataFrame(columns=columns)
            return data.reindex(columns=columns)
        dtype_dict = {
            self.PROVINCE: "object", "Province/State": "object",
            "key": "object", "key_alpha_2": "object"}
        return pd.read_csv(filename, dtype=dtype_dict).reindex(columns=columns)

    @property
    def raw(self):
        """
        pandas.DataFrame: raw data
        """
        return self._raw

    @raw.setter
    def raw(self, dataframe):
        """
        pandas.DataFrame: raw dataset
        """
        self._raw = self._ensure_dataframe(dataframe, name="dataframe")

    @property
    def directory(self):
        """
        str: directory name to save geography information
        """
        return str(self._dirpath)

    @directory.setter
    def directory(self, name):
        self._dirpath = Path(name)

    @staticmethod
    @deprecate(".load()", version="2.21.0-kappa-fu5")
    def load(urlpath, header=0, columns=None, dtype="object"):
        """
        Load a local/remote file.

        Args:
            urlpath (str or pathlib.Path): filename or URL
            header (int): row number of the header
            columns (list[str]): columns to use
            dtype (str or dict[str]): data type for the dataframe or specified columns

        Returns:
            pd.DataFrame: raw dataset
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        kwargs = {
            "low_memory": False, "dtype": dtype, "header": header, "usecols": columns}
        return pd.read_csv(urlpath, **kwargs)

    def cleaned(self):
        """
        Return the cleaned dataset with location information.

        Note:
            Cleaning method is defined by CleaningBase._cleaning() method.

        Returns:
            pandas.DataFrame: cleaned data
        """
        return self._loc_df.merge(self._value_df, how="right", on=self._LOC).drop(self._LOC, axis=1)

    def _cleaning(self, raw):
        """
        Perform data cleaning of the values of the raw data (without location information).

        Args:
            pandas.DataFrame: raw data

        Returns:
            pandas.DataFrame: cleaned data
        """
        raise NotImplementedError

    @property
    def citation(self):
        """
        str: citation/description of the dataset
        """
        return self._citation

    @citation.setter
    def citation(self, description):
        self._citation = str(description)

    def ensure_country_name(self, country, errors="raise"):
        """
        Ensure that the country name is correct.
        If not, the correct country name will be found.

        Args:
            country (str): country name
            errors (str): 'raise' or 'coerce'

        Returns:
            str: country name

        Raises:
            SubsetNotFoundError: no records were found for the country and @errors is 'raise'
        """
        df = self._loc_df.copy()
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.COUNTRY])
        selectable_set = set(df[self.COUNTRY].unique())
        # return country name as-is if selectable
        if country in selectable_set:
            return country
        # Convert country name
        warnings.simplefilter("ignore", FutureWarning)
        converted = coco.convert(country, to="name_short", not_found=None)
        # Additional abbr
        abbr_dict = {
            "Congo Republic": "Republic of the Congo",
            "DR Congo": "Democratic Republic of the Congo",
            "UK": "United Kingdom",
            "Vatican": "Holy See",
        }
        name = abbr_dict.get(converted, converted)
        # Return the name if registered in the dataset
        if name in selectable_set:
            return name
        if errors == "raise":
            raise SubsetNotFoundError(country=country, country_alias=name)

    def _to_location_identifiers(self, geo, country, province, method):
        """
        Convert geographic information to location identifiers.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): geographic information
            country (str or None): country name or ISO3 code
            province(str or None): province name
            method (str): "layer" or "filter", method name of Geometry class

        Raises:
            SubsetNotFoundError: no locations were found with the geographic information

        Returns:
            set(str): list of location identifiers (internal)

        Note:
            Please refer to the documentation of Geometry class for more information regarding @geo argument.
        """
        if {self.ISO3, self.COUNTRY}.issubset(self._layers):
            loc_columns = [col for col in self._layers if col != self.ISO3]
            loc_columns = sorted(set(loc_columns), key=loc_columns.index)
        else:
            loc_columns = self._layers[:]
        if geo is None and country is not None:
            country_alias = self.ensure_country_name(country, errors="raise")
            geo_all = [
                country_alias if col == self.COUNTRY else province if col == self.PROVINCE else None for col in loc_columns]
            geo_arranged = tuple(geo_all[:geo_all.index(None)] if None in geo_all else geo_all)
            warnings.warn(
                f"Argument @country and @province were deprecated and please use geo={geo_arranged}",
                DeprecationWarning, stacklevel=3)
        elif isinstance(geo, str):
            geo_arranged = (self.ensure_country_name(geo, errors="raise"),)
        else:
            geo_arranged = [
                [self.ensure_country_name(c, errors="raise") for c in (info if isinstance(info, list) else [info])]
                if col == self.COUNTRY else info
                for (info, col) in zip(geo or [], loc_columns) if info is not None]
        geography = Geography(layers=loc_columns)
        method_dict = {"layer": geography.layer, "filter": geography.filter}
        df = method_dict[method](data=self._loc_df, geo=geo_arranged)
        if df.empty:
            raise SubsetNotFoundError(geo=geo, country=country, province=province)
        return set(df[self._LOC].tolist())

    @deprecate("CleaningBase.iso3_to_country()", new="CleaningBase.ensure_country_name()")
    def iso3_to_country(self, iso3_code):
        """
        Convert ISO3 code to country name if records are available.

        Args:
            iso3_code (str): ISO3 code or country name

        Returns:
            str: country name

        Note:
            If ISO3 codes are not registered, return the string as-si @iso3_code.
        """
        return self.ensure_country_name(iso3_code)

    def country_to_iso3(self, country, check_data=True):
        """
        Convert country name to ISO3 code if records are available.

        Args:
            country (str): country name
            check_data (bool): whether validate the country name with the dataset

        Raises:
            KeyError: ISO3 code of the country is not registered

        Returns:
            str: ISO3 code or "---" (when unknown)
        """
        name = self.ensure_country_name(country) if check_data else country
        return coco.convert(name, to="ISO3", not_found="---")

    @classmethod
    def area_name(cls, geo=None, country=None, province=None):
        """
        Return area name of the country/province.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): geographic information
            country (str or None): country name or ISO3 code
            province (str or None): province name

        Returns:
            str: area name

        Note:
            If province is None or '-', return country name.
            If not, return the area name, like 'Japan/Tokyo'

        Note:
            Please refer to the documentation of Geometry class for more information regarding @geo argument.
        """
        if isinstance(geo, str):
            return geo
        if isinstance(geo, (list, tuple)):
            return cls.SEP.join(list(geo))
        if province in [None, cls.NA]:
            return country
        return f"{country}{cls.SEP}{province}"

    def layer(self, geo=None, country=None):
        """
        Return the cleaned data at the selected layer.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None (top-level layer)
            country (str or None): country name or None (country level data or country-specific dataset)

        Returns:
            pandas.DataFrame: as-is tue cleaned data at the given layer

        Raises:
            TypeError: @geo has un-expected types
            ValueError: the length of @geo is equal to or larger than the length of layers
            SubsetNotFoundError: no records were found for the country (when @country is not None)

        Note:
            Please refer to Geometry.layer() for more information regarding @geo argument.

        Note:
            @country was deprecated and please use @geo.
            When @country is None, country level data will be returned.
            When @country is a country name, province level data in the selected country will be returned.
        """
        try:
            loc_identifiers = self._to_location_identifiers(geo=geo, country=country, province=None, method="layer")
        except SubsetNotFoundError as e:
            raise e from None
        df = self._value_df.copy()
        df = df.loc[df[self._LOC].isin(loc_identifiers)]
        df = df.merge(self._loc_df, how="left", on=self._LOC).drop(self._LOC, axis=1)
        for col in self._layers:
            if df[col].nunique == 1:
                df = df.drop(col, axis=1)
        return df

    def subset(self, geo=None, country=None, province=None, start_date=None, end_date=None):
        """
        Return subset with country/province name and start/end date.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names for the layers to filter or None (all data at the top level)
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    without ISO3, Country, Province column

        Raises:
            SubsetNotFoundError: no records were found for the condition

        Note:
            Please refer to Geometry.filter() for more information regarding @geo argument.

        Note:
            @country and @province were deprecated and please use @geo.
        """
        try:
            loc_identifiers = self._to_location_identifiers(
                geo=geo, country=country, province=province, method="filter")
        except SubsetNotFoundError:
            raise SubsetNotFoundError(geo=geo, country=country, province=province) from None
        df = self._value_df.copy()
        df = df.loc[df[self._LOC].isin(loc_identifiers)]
        df = df.merge(self._loc_df, how="left", on=self._LOC).drop([self._LOC, *self._layers], axis=1)
        # Subset with start/end date
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.DATE])
        df = df.groupby(self.DATE).sum().reset_index()
        if start_date is None and end_date is None:
            return df
        series = df[self.DATE].copy()
        start_obj = self._ensure_date(start_date, default=series.min())
        end_obj = self._ensure_date(end_date, default=series.max())
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        if df.empty:
            raise SubsetNotFoundError(
                geo=geo, country=country, province=province, start_date=start_date, end_date=end_date) from None
        return df.reset_index(drop=True)

    def subset_complement(self, *args, **kwargs):
        """
        Return the subset. If necessary, complemention will be performed.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def records(self, geo=None, country=None, province=None, start_date=None, end_date=None,
                auto_complement=True, **kwargs):
        """
        Return the subset. If necessary, complemention will be performed.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names for the layers to filter or None (all data at the top level)
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            auto_complement (bool): if True and necessary, the number of cases will be complemented
            kwargs: the other arguments of complement

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    without location identifiers

        Note:
            Please refer to Geometry.filter() for more information regarding @geo argument.

        Note:
            @country and @province were deprecated and please use @geo.
        """
        subset_arg_dict = {
            "geo": geo, "country": country, "province": province, "start_date": start_date, "end_date": end_date}
        if auto_complement:
            with contextlib.suppress(NotImplementedError):
                df, is_complemented = self.subset_complement(**subset_arg_dict, **kwargs)
                if not df.empty:
                    return (df, is_complemented)
        try:
            return (self.subset(**subset_arg_dict), False)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(**subset_arg_dict) from None

    def countries(self):
        """
        Return names of countries where records are registered.

        Raises:
            KeyError: Country names are not registered in this dataset

        Returns:
            list[str]: list of country names
        """
        df = self._ensure_dataframe(self._cleaned_df, name="the cleaned dataset", columns=[self.COUNTRY])
        return list(df[self.COUNTRY].unique())

    def total(self):
        """
        Calculate total values of the cleaned dataset.
        """
        raise NotImplementedError

    def _colored_map(self, title, **kwargs):
        """
        Create global colored map to show the values.

        Args:
            title (str): title of the figure
            kwargs: arguments of ColoredMap() and ColoredMap.plot()
        """
        with ColoredMap(**find_args([plt.savefig, ColoredMap], **kwargs)) as cm:
            cm.title = title
            cm.directory = self._dirpath
            cm.plot(**find_args([gpd.GeoDataFrame.plot, ColoredMap.plot], **kwargs))

    def _colored_map_global(self, variable, title, date, **kwargs):
        """
        Create global colored map to show the values at country level.

        Args:
            variable (str): variable name to show
            title (str): title of the figure
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()
        """
        df = self._loc_df.merge(self._value_df, how="right", on=self._LOC)
        # Check variable name
        if variable not in df.columns:
            candidates = [col for col in df.columns if col not in self._LOC_COLS]
            raise UnExpectedValueError(name="variable", value=variable, candidates=candidates)
        # Remove cruise ships
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.COUNTRY])
        df = df.loc[df[self.COUNTRY] != self.OTHERS]
        # Recognize province as a region/country
        if self.PROVINCE in df:
            with contextlib.suppress(ValueError):
                df[self.ISO3] = df[self.ISO3].astype("category").cat.add_categories(["GRL"])
                df[self.COUNTRY] = df[self.COUNTRY].astype("category").cat.add_categories(["Greenland"])
                df.loc[df[self.PROVINCE] == "Greenland", self._LOC_COLS] = ["GRL", "Greenland", self.NA]
        # Select country level data
        if self.PROVINCE in df.columns:
            df = df.loc[df[self.PROVINCE] == self.NA]
        # Select date
        if date is not None:
            self._ensure_dataframe(df, name="cleaned dataset", columns=[self.DATE])
            df = df.loc[df[self.DATE] == pd.to_datetime(date)]
        df[self.COUNTRY] = df[self.COUNTRY].astype(str)
        df = df.groupby(self.COUNTRY).last().reset_index()
        # Plotting
        df.rename(columns={variable: "Value"}, inplace=True)
        self._colored_map(title=title, data=df, level=self.COUNTRY, **kwargs)

    def _colored_map_country(self, country, variable, title, date, **kwargs):
        """
        Create country-specific colored map to show the values at province level.

        Args:
            country (str): country name
            variable (str): variable name to show
            title (str): title of the figure
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of covsirphy.ColoredMap() and covsirphy.ColoredMap.plot()
        """
        df = self._loc_df.merge(self._value_df, how="right", on=self._LOC)
        country_alias = self.ensure_country_name(country)
        # Check variable name
        if variable not in df.columns:
            candidates = [col for col in df.columns if col not in self._LOC_COLS]
            raise UnExpectedValueError(name="variable", value=variable, candidates=candidates)
        # Select country-specific data
        self._ensure_dataframe(df, name="cleaned dataset", columns=[self.COUNTRY, self.PROVINCE])
        df = df.loc[df[self.COUNTRY] == country_alias]
        df = df.loc[df[self.PROVINCE] != self.NA]
        if df.empty:
            raise SubsetNotFoundError(country=country, country_alias=country_alias, message="at province level")
        # Select date
        if date is not None:
            self._ensure_dataframe(df, name="cleaned dataset", columns=[self.DATE])
            df = df.loc[df[self.DATE] == pd.to_datetime(date)]
        df = df.groupby(self.PROVINCE).last().reset_index()
        # Plotting
        df[self.COUNTRY] = country_alias
        df.rename(columns={variable: "Value"}, inplace=True)
        self._colored_map(title=title, data=df, level=self.PROVINCE, **kwargs)
