#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from pathlib import Path
import pandas as pd
from covsirphy.util.term import Term


class _RemoteDatabase(Term):
    """
    Base class to access remote databases.

    Args:
        filename (str): CSV filename to save records
        iso3 (str or None): ISO3 code of the country which must be included in the dataset or None (all available countries)
    """
    # Citation
    CITATION = ""
    # Column names and data types
    # {"name in database": "name defined in Term class"}
    COL_DICT = {}

    def __init__(self, filename, iso3):
        # Filepath to save files
        try:
            self.filepath = Path(filename)
        except TypeError as e:
            raise TypeError(f"@filename should be a path-like object, but {filename} was applied.") from e

        self.filepath.parent.mkdir(exist_ok=True, parents=True)
        # Country
        self._iso3 = None if iso3 is None else str(iso3)
        # List of primary sources
        self.primary_list = []
        # List of column names (defined in Term class)
        self.saved_cols = list(self.COL_DICT.values())

    def to_dataframe(self, force, verbose):
        """
        Load the dataset and return it as a dataframe.

        Args:
            force (bool): whether download the dataset from server or not when we have saved dataset
            verbose (int): level of verbosity

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by the first values of self.COL_DICT.values()
        """
        # Read local file if available and usable
        if not force and self.filepath.exists():
            with contextlib.suppress(ValueError):
                df = self._ensure_dataframe(self.read(), columns=self.saved_cols)
                if self._iso3 is None and df[self.ISO3].nunique() > 1:
                    return df
                if self._iso3 is not None and self._value_included(df, value_dict={self.ISO3: self._iso3}):
                    return df
        # Download dataset from server
        df = self.download(verbose=verbose)
        df = df.rename(columns=self.COL_DICT)
        df = self._ensure_dataframe(df, columns=self.saved_cols)
        df.to_csv(self.filepath, index=False)
        return df

    def read(self):
        """
        Load the dataset with a local file.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by .COL_DICT.values()
        """
        kwargs = {"low_memory": False, "header": 0, "usecols": self.saved_cols}
        return pd.read_csv(self.filepath, **kwargs)

    def download(self, verbose):
        """
        Download the dataset from the server and set the list of primary sources.

        Args:
            verbose (int): level of verbosity

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by .COL_DICT.values()
        """
        raise NotImplementedError

    @property
    def primary(self):
        """
        str: the list of primary sources.
        """
        return "\n".join(self.primary_list)
