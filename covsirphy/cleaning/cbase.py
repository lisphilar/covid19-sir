#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import country_converter as coco
from dask import dataframe as dd
import pandas as pd
from covsirphy.util.error import deprecate, SubsetNotFoundError
from covsirphy.cleaning.term import Term


class CleaningBase(Term):
    """
    Basic class for data cleaning.

    Args:
        filename (str or None): CSV filename of the dataset
        citation (str or None): citation

    Returns:
        If @filename is None, empty dataframe will be set as raw data.
        If @citation is None, citation will be empty string.
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
        self._raw = self.ensure_dataframe(dataframe, name="dataframe")

    @staticmethod
    def load(urlpath, header=0, columns=None):
        """
        Load a local/remote file.

        Args:
            urlpath (str or pathlib.Path): filename or URL
            header (int): row number of the header

        Returns:
            pd.DataFrame: raw dataset
        """
        kwargs = {
            "low_memory": False, "dtype": "object", "header": header, "names": columns}
        try:
            return dd.read_csv(urlpath, blocksize=None, **kwargs).compute()
        except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError):
            return pd.read_csv(urlpath, **kwargs)

    def cleaned(self):
        """
        Return the cleaned dataset.

        Notes:
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

    def ensure_country_name(self, country):
        """
        Ensure that the country name is correct.
        If not, the correct country name will be found.

        Args:
            country (str): country name

        Returns:
            str: country name
        """
        df = self.ensure_dataframe(
            self._cleaned_df, name="the cleaned dataset", columns=[self.COUNTRY])
        selectable_set = set(df[self.COUNTRY].unique())
        # return country name as-is if selectable
        if country in selectable_set:
            return country
        # Convert country name
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
        raise SubsetNotFoundError(country=country, country_alias=name)

    @deprecate("CleaningBase.iso3_to_country()", new="CleaningBase.ensure_country_name()")
    def iso3_to_country(self, iso3_code):
        """
        Convert ISO3 code to country name if records are available.

        Args:
            iso3_code (str): ISO3 code or country name

        Returns:
            str: country name

        Notes:
            If ISO3 codes are not registered, return the string as-si @iso3_code.
        """
        return self.ensure_country_name(iso3_code)

    def country_to_iso3(self, country):
        """
        Convert country name to ISO3 code if records are available.

        Args:
            country (str): country name

        Raises:
            KeyError: ISO3 code of the country is not registered

        Returns:
            str: ISO3 code or "---" (when unknown)
        """
        name = self.ensure_country_name(country)
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

        Notes:
            If province is None or '-', return country name.
            If not, return the area name, like 'Japan/Tokyo'
        """
        if province in [None, cls.UNKNOWN]:
            return country
        return f"{country}{cls.SEP}{province}"

    def _subset_by_area(self, country, province=None):
        """
        Return subset for the country/province.

        Args:
            country (str): country name
            province (str or None): province name

        Returns:
            pandas.DataFrame: subset for the country/province
        """
        # Country level
        country = self.ensure_country_name(country)
        df = self.ensure_dataframe(
            self._cleaned_df, name="the cleaned dataset", columns=[self.COUNTRY])
        df = df.loc[df[self.COUNTRY] == country, :]
        # Province level
        province = province or self.UNKNOWN
        if self.PROVINCE not in df.columns and province == self.UNKNOWN:
            return df
        df = self.ensure_dataframe(
            df, "the cleaned dataset", columns=[self.PROVINCE])
        return df.loc[df[self.PROVINCE] == province, :]

    def subset(self, country, province=None, start_date=None, end_date=None):
        """
        Return subset of the country/province and start/end date.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index: reset index
                Columns: without ISO3, Country, Province column
        """
        country_alias = self.ensure_country_name(country)
        df = self._subset_by_area(country=country, province=province)
        df = df.drop(
            [self.COUNTRY, self.ISO3, self.PROVINCE], axis=1, errors="ignore")
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province)
        # Subset with Start/end date
        if start_date is None and end_date is None:
            return df.reset_index(drop=True)
        df = self.ensure_dataframe(
            df, name="the cleaned dataset", columns=[self.DATE])
        series = df[self.DATE].copy()
        start_obj = self.date_obj(date_str=start_date, default=series.min())
        end_obj = self.date_obj(date_str=end_date, default=series.max())
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date)
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
                Index: reset index
                Columns:
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
        df = self.ensure_dataframe(
            self._cleaned_df, name="the cleaned dataset", columns=[self.COUNTRY])
        return list(df[self.COUNTRY].unique())

    def total(self):
        """
        Calculate total values of the cleaned dataset.
        """
        raise NotImplementedError
