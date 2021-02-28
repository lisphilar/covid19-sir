#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import country_converter as coco
from dask import dataframe as dd
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
from covsirphy.util.argument import find_args
from covsirphy.util.error import deprecate, SubsetNotFoundError, UnExpectedValueError
from covsirphy.util.term import Term
from covsirphy.visualization.colored_map import ColoredMap


class CleaningBase(Term):
    """
    Basic class for data cleaning.

    Args:
        filename (str or None): CSV filename of the dataset
        citation (str or None): citation

    Note:
        - If @filename is None, empty dataframe will be set as raw data and geometry information will be saved in "input" directory.
        - If @filename is not None, geometry information will be saved in the directory which has the file.
        - The directory of geometry information could be changed with .directory property.
        - If @citation is None, citation will be empty string.
    """

    def __init__(self, filename, citation=None):
        warnings.simplefilter("ignore", DeprecationWarning)
        if filename is None:
            self._raw = pd.DataFrame()
            self._cleaned_df = pd.DataFrame()
        else:
            Path(filename).parent.mkdir(exist_ok=True, parents=True)
            self._raw = dd.read_csv(
                filename, dtype={"Province/State": "object"}
            ).compute()
            self._cleaned_df = self._cleaning()
        self._citation = citation or ""
        # Directory that save the file
        if filename is None:
            self._dirpath = Path("input")
        else:
            self._dirpath = Path(filename).resolve().parent

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
        str: directory name to save geometry information
        """
        return str(self._dirpath)

    @directory.setter
    def directory(self, name):
        self._dirpath = Path(name)

    @staticmethod
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
        try:
            return dd.read_csv(urlpath, blocksize=None, **kwargs).compute()
        except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError):
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
            str: country name

        Raises:
            SubsetNotFoundError: no records were found for the country and @errors is 'raise'
        """
        df = self._cleaned_df.copy()
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
    def area_name(cls, country, province=None):
        """
        Return area name of the country/province.

        Args:
            country (str): country name or ISO3 code
            province (str): province name

        Returns:
            str: area name

        Note:
            If province is None or '-', return country name.
            If not, return the area name, like 'Japan/Tokyo'
        """
        if province in [None, cls.UNKNOWN]:
            return country
        return f"{country}{cls.SEP}{province}"

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
        self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.COUNTRY])
        if self.PROVINCE not in df:
            df[self.PROVINCE] = self.UNKNOWN
        df[self.AREA_COLUMNS] = df[self.AREA_COLUMNS].astype(str)
        # Country level data
        if country is None:
            df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
            return df.drop(self.PROVINCE, axis=1).reset_index(drop=True)
        # Province level data at the selected country
        country_alias = self.ensure_country_name(country, errors="coerce")
        df = df.loc[df[self.COUNTRY] == country_alias]
        if df.empty:
            raise SubsetNotFoundError(country=country, country_alias=country_alias) from None
        df = df.loc[df[self.PROVINCE] != self.UNKNOWN]
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
        if province is None or province == self.UNKNOWN:
            df = self.layer(country=None)
            country_alias = self.ensure_country_name(country)
            df = df.loc[df[self.COUNTRY] == country_alias]
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
        country_alias = self.ensure_country_name(country, errors="coerce")
        try:
            df = self._subset_by_area(country=country, province=province)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province) from None
        df = df.drop([self.COUNTRY, self.ISO3, self.PROVINCE], axis=1, errors="ignore")
        # Subset with Start/end date
        if start_date is None and end_date is None:
            return df.reset_index(drop=True)
        df = self._ensure_dataframe(df, name="the cleaned dataset", columns=[self.DATE])
        series = df[self.DATE].copy()
        start_obj = self.date_obj(date_str=start_date, default=series.min())
        end_obj = self.date_obj(date_str=end_date, default=series.max())
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date) from None
        return df.reset_index(drop=True)

    def subset_complement(self, country, **kwargs):
        """
        Return the subset. If necessary, complemention will be performed.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def records(self, country, province=None, start_date=None, end_date=None,
                auto_complement=True, **kwargs):
        """
        Return the subset. If necessary, complemention will be performed.

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
        country_alias = self.ensure_country_name(country)
        subset_arg_dict = {
            "country": country, "province": province, "start_date": start_date, "end_date": end_date}
        if auto_complement:
            try:
                df, is_complemented = self.subset_complement(
                    **subset_arg_dict, **kwargs)
                if not df.empty:
                    return (df, is_complemented)
            except NotImplementedError:
                pass
        try:
            return (self.subset(**subset_arg_dict), False)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
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
        # Select country level data
        if self.PROVINCE in df.columns:
            df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
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
        country_alias = self.ensure_country_name(country)
        # Check variable name
        if variable not in df.columns:
            candidates = [col for col in df.columns if col not in self.AREA_ABBR_COLS]
            raise UnExpectedValueError(name="variable", value=variable, candidates=candidates)
        # Select country-specific data
        self._ensure_dataframe(df, name="cleaned dataset", columns=[self.COUNTRY, self.PROVINCE])
        df = df.loc[df[self.COUNTRY] == country_alias]
        df = df.loc[df[self.PROVINCE] != self.UNKNOWN]
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, message="at province level")
        # Select date
        if date is not None:
            self._ensure_dataframe(df, name="cleaned dataset", columns=[self.DATE])
            df = df.loc[df[self.DATE] == pd.to_datetime(date)]
        df = df.groupby(self.PROVINCE).last().reset_index()
        # Plotting
        df[self.COUNTRY] = country_alias
        df.rename(columns={variable: "Value"}, inplace=True)
        self._colored_map(title=title, data=df, level=self.PROVINCE, **kwargs)
