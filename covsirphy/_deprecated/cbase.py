#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from pathlib import Path
import warnings
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.util.error import deprecate, SubsetNotFoundError, UnExpectedValueError
from covsirphy.util.term import Term
from covsirphy._deprecated.colored_map import ColoredMap


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
                - ISO3: ISO3 code
                - Country: country/region name (optional)
                - Province: province/prefecture/state name
                - Confirmed: the number of confirmed cases
                - Fatal: the number of fatal cases
                - Recovered: the number of recovered cases
                - Population: population values (optional)
        citation (str or None): citation or None (empty)
        variables (list[str] or None): variables to clean (not including date and location identifiers)

    Note:
        Either @filename (high priority) or @data must be specified.

    Note:
        - If @filename is None, geography information will be saved in "input" directory.
        - If @filename is not None, geography information will be saved in the directory which has the file.
        - The directory of geography information could be changed with .directory property.
    """

    @deprecate("CleaningBase", version="2.27.0-zeta")
    def __init__(self, filename=None, data=None, citation=None, variables=None):
        # Columns of self._raw, self._clean_df and self.cleaned()
        self._raw_cols = [self.DATE, self.ISO3, self.COUNTRY, self.PROVINCE] + (variables or [])
        self._subset_cols = [self.DATE] + (variables or [])
        # Raw data
        self._raw = self._parse_raw(filename, data, self._raw_cols)
        # Data cleaning
        self._cleaned_df = pd.DataFrame(columns=self._raw_cols) if self._raw.empty else self._cleaning()
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
            "key": "object", "key_alpha_2": "object"
        }
        return pd.read_csv(filename, dtype=dtype_dict).reindex(columns=columns)

    @staticmethod
    def _ensure_instance(target, class_obj, name="target"):
        """
        Ensure the target is an instance of the class object.

        Args:
            target (instance): target to ensure
            parent (class): class object
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        s = f"@{name} must be an instance of {class_obj}, but {type(target)} was applied."
        if not isinstance(target, class_obj):
            raise TypeError(s)
        return target

    @staticmethod
    def _ensure_dataframe(target, name="df", time_index=False, columns=None, empty_ok=True):
        """
        Ensure the dataframe has the columns.

        Args:
            target (pandas.DataFrame): the dataframe to ensure
            name (str): argument name of the dataframe
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have
            empty_ok (bool): whether give permission to empty dataframe or not

        Returns:
            pandas.DataFrame:
                Index
                    as-is
                Columns:
                    columns specified with @columns or all columns of @target (when @columns is None)
        """
        if not isinstance(target, pd.DataFrame):
            raise TypeError(f"@{name} must be a instance of (pandas.DataFrame).")
        df = target.copy()
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if not empty_ok and target.empty:
            raise ValueError(f"@{name} must not be a empty dataframe.")
        if columns is None:
            return df
        if not set(columns).issubset(df.columns):
            cols_str = ", ".join(col for col in columns if col not in df.columns)
            included = ", ".join(df.columns.tolist())
            s1 = f"Expected columns were not included in {name} with {included}."
            raise KeyError(f"{s1} {cols_str} must be included.")
        return df.loc[:, columns]

    @staticmethod
    def _ensure_natural_int(target, name="number", include_zero=False, none_ok=False):
        """
        Ensure a natural (non-negative) number.

        Args:
            target (int or float or str or None): value to ensure
            name (str): argument name of the value
            include_zero (bool): include 0 or not
            none_ok (bool): None value can be applied or not.

        Returns:
            int: as-is the target

        Note:
            When @target is None and @none_ok is True, None will be returned.
            If the value is a natural number and the type was float or string,
            it will be converted to an integer.
        """
        if target is None and none_ok:
            return None
        s = f"@{name} must be a natural number, but {target} was applied"
        try:
            number = int(target)
        except TypeError as e:
            raise TypeError(f"{s} and not converted to integer.") from e
        if number != target:
            raise ValueError(f"{s}. |{target} - {number}| > 0")
        min_value = 0 if include_zero else 1
        if number < min_value:
            raise ValueError(f"{s}. This value is under {min_value}")
        return number

    @classmethod
    def _ensure_date(cls, target, name="date", default=None):
        """
        Ensure the format of the string.

        Args:
            target (str or pandas.Timestamp): string to ensure
            name (str): argument name of the string
            default (pandas.Timestamp or None): default value to return

        Returns:
            pandas.Timestamp or None: as-is the target or default value
        """
        if target is None:
            return default
        if isinstance(target, pd.Timestamp):
            return target.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            return pd.to_datetime(target).replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError as e:
            raise ValueError(f"{name} was not recognized as a date, {target} was applied.") from e

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
        Return the cleaned dataset.

        Note:
            Cleaning method is defined by CleaningBase._cleaning() method.

        Returns:
            pandas.DataFrame: cleaned data
        """
        return self._cleaned_df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            pandas.DataFrame: cleaned data
        """
        return self._raw.copy()

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
            str: ISO3 code

        Raises:
            SubsetNotFoundError: no records were found for the country and @errors is 'raise'
        """
        iso3 = self._to_iso3(country)[0]
        df = self._cleaned_df.copy()
        if self.ISO3 in df and iso3 in df[self.ISO3].unique():
            return iso3
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.COUNTRY])
        selectable_set = set(df[self.COUNTRY].unique())
        # return country name as-is if selectable
        if country in selectable_set:
            return iso3
        if errors == "raise":
            raise SubsetNotFoundError(country=country, country_alias=iso3)

    @deprecate("CleaningBase.iso3_to_country()")
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

    @deprecate("CleaningBase.country_to_iso3()", new="CleaningBase.ensure_country_name()", version="2.24.0-gamma")
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
        iso3 = self._to_iso3(name)[0]
        return self.NA * 3 if iso3 is None or iso3 == country else iso3

    @classmethod
    def area_name(cls, country, province=None):
        """
        Return area name of the province/country.

        Args:
            country (str): country name or ISO3 code
            province (str): province name

        Returns:
            str: area name

        Note:
            If province is None or '-', return country name.
            If not, return the area name, like 'Tokyo/Japan'
        """
        if province in [None, cls.NA]:
            return country
        return f"{province}{cls.SEP}{country}"

    def layer(self, country=None):
        """
        Return the cleaned data at the selected layer.

        Args:
            country (str or None): country name or None (country level data or country-specific dataset)

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                - Country (str): country names
                - Province (str): province names (or removed when country level data)
                - any other columns of the cleaned data

        Raises:
            SubsetNotFoundError: no records were found for the country (when @country is not None)
            KeyError: @country was None, but country names were not registered in the dataset

        Note:
            When @country is None, country level data will be returned.
            When @country is a country name, province level data in the selected country will be returned.
        """
        df = self._cleaned_df.copy()
        for col in [self.ISO3, self.COUNTRY, self.PROVINCE]:
            df[col] = df[col].astype(str, errors="ignore")
        # Country level data
        if country is None:
            if self.PROVINCE in df:
                df = df.loc[df[self.PROVINCE] == self.NA].drop(self.PROVINCE, axis=1)
            return df.reset_index(drop=True)
        # Province level data at the selected country
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.PROVINCE])
        iso3 = self.ensure_country_name(country)
        df = df.loc[df[self.ISO3] == iso3]
        if df.empty:
            raise SubsetNotFoundError(country=country, country_alias=iso3) from None
        if self.PROVINCE in df:
            df = df.loc[df[self.PROVINCE] != self.NA]
        return df.reset_index(drop=True)

    def _subset_by_area(self, country, province=None):
        """
        Return subset for the country/province.

        Args:
            country (str): country name
            province (str or None): province name or None (country level data)

        Returns:
            pandas.DataFrame: subset for the country/province, columns are not changed

        Raises:
            SubsetNotFoundError: no records were found for the condition
        """
        # Country level
        if province is None or province == self.NA:
            df = self.layer(country=None)
            iso3 = self.ensure_country_name(country)
            df = df.loc[df[self.ISO3] == iso3]
            return df.reset_index(drop=True)
        # Province level
        df = self.layer(country=country)
        df = df.loc[df[self.PROVINCE] == province]
        if df.empty:
            raise SubsetNotFoundError(country=country)
        return df.reset_index(drop=True)

    def subset(self, country, province=None, start_date=None, end_date=None):
        """
        Return subset with country/province name and start/end date.

        Args:
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
        """
        iso3 = self.ensure_country_name(country, errors="coerce")
        try:
            df = self._subset_by_area(country=country, province=province)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country, country_alias=iso3, province=province) from None
        df = df.drop([self.COUNTRY, self.ISO3, self.PROVINCE], axis=1, errors="ignore")
        # Subset with Start/end date
        if start_date is None and end_date is None:
            return df.reset_index(drop=True)
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.DATE])
        series = df[self.DATE].copy()
        start_obj = self._ensure_date(start_date, default=series.min())
        end_obj = self._ensure_date(end_date, default=series.max())
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=iso3, province=province,
                start_date=start_date, end_date=end_date) from None
        return df.reset_index(drop=True)

    def subset_complement(self, country, **kwargs):
        """
        Return the subset. If necessary, complementing will be performed.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def records(self, country, province=None, start_date=None, end_date=None,
                auto_complement=True, **kwargs):
        """
        Return the subset. If necessary, complementing will be performed.

        Args:
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
                    without ISO3, Country, Province column
        """
        iso3 = self.ensure_country_name(country)
        subset_arg_dict = {
            "country": country, "province": province, "start_date": start_date, "end_date": end_date}
        if auto_complement:
            with contextlib.suppress(NotImplementedError):
                df, is_complemented = self.subset_complement(
                    **subset_arg_dict, **kwargs)
                if not df.empty:
                    return (df, is_complemented)
        try:
            return (self.subset(**subset_arg_dict), False)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country, country_alias=iso3, province=province,
                start_date=start_date, end_date=end_date) from None

    def countries(self):
        """
        Return names of countries where records are registered.

        Raises:
            KeyError: Country names are not registered in this dataset

        Returns:
            list[str]: list of country names
        """
        df = self._ensure_dataframe(
            self._cleaned_df, name="the cleaned dataset", columns=[self.COUNTRY])
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
        df = self._cleaned_df.copy()
        # Check variable name
        if variable not in df.columns:
            candidates = [col for col in df.columns if col not in self.AREA_ABBR_COLS]
            raise UnExpectedValueError(name="variable", value=variable, candidates=candidates)
        # Remove cruise ships
        df = df.loc[df[self.COUNTRY] != self.OTHERS]
        # Recognize province as a region/country
        if self.PROVINCE in df:
            with contextlib.suppress(ValueError):
                df[self.ISO3] = df[self.ISO3].cat.add_categories(["GRL"])
                df[self.COUNTRY] = df[self.COUNTRY].cat.add_categories(["Greenland"])
                df.loc[df[self.PROVINCE] == "Greenland", self.AREA_ABBR_COLS] = ["GRL", "Greenland", self.NA]
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
        df = self._cleaned_df.copy()
        iso3 = self.ensure_country_name(country)
        # Check variable name
        if variable not in df.columns:
            candidates = [col for col in df.columns if col not in self.AREA_ABBR_COLS]
            raise UnExpectedValueError(name="variable", value=variable, candidates=candidates)
        # Select country-specific data
        self._ensure_dataframe(df, name="cleaned dataset", columns=[self.ISO3, self.PROVINCE])
        df = df.loc[df[self.ISO3] == iso3]
        df = df.loc[df[self.PROVINCE] != self.NA]
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=iso3, details="at province level")
        # Select date
        if date is not None:
            self._ensure_dataframe(df, name="cleaned dataset", columns=[self.DATE])
            df = df.loc[df[self.DATE] == pd.to_datetime(date)]
        df = df.groupby(self.PROVINCE).last().reset_index()
        # Plotting
        df[self.COUNTRY] = country
        df.rename(columns={variable: "Value"}, inplace=True)
        self._colored_map(title=title, data=df, level=self.PROVINCE, **kwargs)
