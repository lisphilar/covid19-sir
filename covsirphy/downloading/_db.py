#!/usr/bin/env python
# -*- coding: utf-8 -*-


from covsirphy.util.filer import Filer
from covsirphy.util.term import Term
from covsirphy.downloading._provider import _DataProvider


class _DataBase(Term):
    """Basic class for databases.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        update_interval (int): update interval of downloading dataset
    """
    # File title without extensions and suffix
    TITLE = ""
    # Dictionary of column names
    COL_DICT = {}
    # Stdout when downloading (shown at most one time)
    STDOUT = None
    # Citation
    CITATION = ""

    def __init__(self, directory, update_interval):
        self._filer = Filer(directory=directory)
        self._update_interval = update_interval
        self._provider = _DataProvider(update_interval=self._update_interval, stdout=self.STDOUT)

    def layer(self, country, province):
        """Return the data at the selected layer.

        Args:
            country (str or None): country name or None
            province (str or None): province/state/prefecture name or None

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (str): province/state/prefecture names
                    - City (str): city names
                    - the other available columns

        Note:
            When @country is None, country-level data will be returned.

        Note:
            When @country is a string and @province is None, province-level data in the country will be returned.

        Note:
            When @country and @province are strings, city-level data in the province will be returned.
        """
        if country is None:
            return self._country()
        if province is None:
            return self._province(country=country)
        return self._city(country=country, province=province)

    def _country(self):
        """Returns country-level data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (object): country names
                    - Province (object): NAs
                    - City (object): NAs
                    - the other available columns
        """
        raise NotImplementedError

    def _province(self, country):
        """Returns province-level data.

        Args:
            country (str): country name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (object): country names
                    - Province (object): province/state/prefecture names
                    - City (object): NAs
                    - the other available columns
        """
        raise NotImplementedError

    def _city(self, country, province):
        """Returns city-level data.

        Args:
            country (str): country name
            province (str): province/state/prefecture name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (object): country names
                    - Province (object): province/state/prefecture names
                    - City (object): city names
                    - the other available columns
        """
        raise NotImplementedError

    def _provide(self, url, suffix, columns, date=None, date_format="%Y-%m-%d"):
        """Provide the latest data and rename with class variable .COL_DICT.

        Args:
            url (str): URL of the data
            suffix (str): suffix of the file title
            columns (list[str]): columns to use
            date (str or None): column name of date
            date_format (str): format of date column, like %Y-%m-%d

        Returns:
            pandas.DataFrame

        Note:
            File will be downloaded to '/{self._directory}/{title}{suffix}.csv'.
        """
        filename = self._filer.csv(title=f"{self.TITLE}{suffix}")["path_or_buf"]
        df = self._provider.latest(filename=filename, url=url, columns=columns, date=date, date_format=date_format)
        return df.rename(columns=self.COL_DICT)
