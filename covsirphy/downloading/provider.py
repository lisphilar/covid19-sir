#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from covsirphy.util.filer import Filer
from covsirphy.util.term import Term


class _DataProvider(Term):
    """Extract datasets and save it locally.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        title (str): file title without extensions
        columns (list[str]): column names the dataset must have
        update_interval (int): update interval of downloading dataset
    """

    def __init__(self, directory, title, columns, update_interval):
        self._filename = Filer(directory=directory).csv(title=title)["path_or_buf"]
        self._update_interval = self._ensure_natural_int(update_interval, include_zero=True, name="update_interval")

    def provide(self, url, verbose):
        """Provide the last dataset as a dataframe, downloading remote files or reading local files.

        Args:
            url (str): URL of the dataset
            verbose (int): level of verbosity when downloading

        Note:
            If @verbose is 0, no descriptions will be shown.
            If @verbose is 1 or larger, URL and database name will be shown.
        """
        verbose = self._ensure_natural_int(verbose, include_zero=True, name="verbose")

    @staticmethod
    def _last_updated_local(path):
        """
        Return the date last updated of local file/directory.

        Args:
            path (str or pathlib.Path): name of the file/directory

        Returns:
            datetime.datetime: time last updated (UTC)
        """
        m_time = Path(path).stat().st_mtime
        date = datetime.fromtimestamp(m_time)
        return date.astimezone(timezone.utc).replace(tzinfo=None)

    def _download_necessity(self, filename):
        """
        Return whether we need to get the data from remote servers or not,
        comparing the last update of the files.
        Args:
            filename (str): filename of the local file
        Returns:
            (bool): whether we need to get the data from remote servers or not
        Note:
            If the last updated date is unknown, returns True.
            If @update_interval (of _Recommended) hours have passed and the remote file was updated, return True.
        """
        if not Path(filename).exists():
            return True
        date_local = self._last_updated_local(filename)
        time_limit = date_local + timedelta(hours=self._update_interval)
        return datetime.now() > time_limit

    @staticmethod
    def _read_csv(path, columns):
        """Read the CSV file and return as a dataframe.

        Args:
            columns (list[str]): column names the dataset must have
        """
        kwargs = {"low_memory": False, "header": 0, "usecols": columns}
        return pd.read_csv(path, **kwargs)
