#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
import warnings
import numpy as np
import pandas as pd
from unidecode import unidecode
from covsirphy.util.config import config
from covsirphy.util.term import Term


class _DataProvider(Term):
    """Extract datasets and save it locally.

    Args:
        update_interval (int or None): update interval of downloading dataset or None (do not update if we have files in local)
        stdout: stdout when downloading if the logger level > info
    """

    def __init__(self, update_interval, stdout):
        self._update_interval = update_interval
        self._stdout = stdout

    def latest(self, filename, url, columns, date, date_format):
        """Provide the last dataset as a dataframe, downloading remote files or reading local files.

        Args:
            filename (str or pathlib.Path): filename to save/read the download
            url (str): URL of the dataset
            columns (list[str]): column names the dataset must have
            date (str or None): column name of date
            date_format (str): format of date column, like %Y-%m-%d

        Returns:
            pandas.DataFrame
        """
        if not self.download_necessity(filename):
            with contextlib.suppress(ValueError):
                return self.read_csv(filename, columns, date=date, date_format=date_format)
        config.info(self._stdout)
        df = self.read_csv(url, columns, date=date, date_format=date_format)
        df.to_csv(filename, index=False)
        return df

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

    def download_necessity(self, filename):
        """
        Return whether we need to get the data from remote servers or not,
        comparing the last update of the files.

        Args:
            filename (str): filename of the local file

        Returns:
            bool: whether we need to get the data from remote servers or not

        Note:
            If the last updated date is unknown, returns True.
            If update interval was set as None, return False.
            If @update_interval (of _DataProvider) hours have passed and the remote file was updated, return True.
        """
        if not Path(filename).exists():
            return True
        if self._update_interval is None:
            return False
        date_local = self._last_updated_local(filename)
        time_limit = date_local + timedelta(hours=self._update_interval)
        return datetime.now() >= time_limit

    @staticmethod
    def read_csv(path, columns, date, date_format):
        """Read the CSV file and return as a dataframe.

        Args:
            columns (list[str] or None): column names the dataset must have
            date (str or None): column name of date
            date_format (str): format of date column, like %Y-%m-%d

        Returns:
            pandas.DataFrame: downloaded data
        """
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        kwargs = {
            "header": 0, "usecols": columns, "encoding": "utf-8", "engine": "pyarrow",
            "parse_dates": None if date is None else [date], "date_parser": lambda x: datetime.strptime(x, date_format)
        }
        if urlparse(path).scheme:
            kwargs["storage_options"] = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(path, **kwargs)
        for col in df:
            with contextlib.suppress(TypeError):
                df[col] = df[col].apply(lambda x: unidecode(x) if len(x) else np.nan)
        return df
